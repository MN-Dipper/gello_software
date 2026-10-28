"""启动时对齐 GELLO，随后使用 franky 异步关节轨迹遥操作，可映射原装 Franka Hand 夹爪。

python test/teleop_franka.py --config test/gello_calibration.json
python test/teleop_franka.py --config test/gello_calibration.json --preview

标定文件使用 test/1.py 生成的 schema=1 格式。参考姿态必须在两台机械臂
物理形状对应时采集；随意将两个不同姿态标为参考，不能得到形状对应关系。
对齐期间请保持 GELLO 静止。Ctrl+C 请求减速停止。
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the serial reader without renaming the user's existing 1.py.
_spec = importlib.util.spec_from_file_location(
    "gello_calibration_reader", Path(__file__).with_name("1.py")
)
_reader_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader_module)
GelloReader = _reader_module.GelloReader


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是 7 个有限数值")
    return result


def load_config(path):
    with path.open(encoding="utf-8") as stream:
        c = json.load(stream)
    if c.get("schema") != 1:
        raise ValueError("需要 schema=1 的 GELLO 标定文件")
    signs = vector(c["signs"], "signs")
    if not np.all(np.isin(signs, [-1, 1])):
        raise ValueError("signs 必须为 +1/-1")
    ids = c["ids"]
    if (len(ids) != 7 or len(set(ids)) != 7
            or any(type(i) is not int or not 0 <= i <= 252 for i in ids)
            or type(c["baudrate"]) is not int or c["baudrate"] <= 0):
        raise ValueError("IDs 或波特率无效")
    return (c, signs, vector(c["encoder_reference"], "encoder_reference"),
            vector(c["robot_reference"], "robot_reference"))


class Targets:
    def __init__(self, reader, signs, encoder_ref, robot_ref, max_speed):
        self.reader = reader
        self.signs = signs
        self.encoder_ref = encoder_ref
        self.robot_ref = robot_ref
        self.max_speed = max_speed
        self.previous = None
        self.timestamp = None

    def read(self):
        raw = np.asarray(self.reader.read(), dtype=float)
        encoder = vector(raw[:7], "GELLO 读数")
        self.gripper_angle = float(raw[7]) if len(raw) == 8 else None
        if self.gripper_angle is not None and not np.isfinite(self.gripper_angle):
            raise RuntimeError("GELLO 夹爪读数无效")
        now = time.monotonic()
        if self.previous is not None:
            dt = now - self.timestamp
            if dt > 0.3:
                raise RuntimeError("GELLO 数据间隔超过 0.3 秒，停止跟随")
            # Do not silently unwrap: a wrap may indicate an incorrect calibration.
            if np.max(np.abs(encoder - self.previous)) > max(0.04, self.max_speed * dt):
                raise RuntimeError("GELLO 跨圈、读数突变或移动过快，停止跟随")
        self.previous, self.timestamp = encoder.copy(), now
        return self.robot_ref + self.signs * (encoder - self.encoder_ref)


def gripper_fraction(angle, config):
    opened, closed = float(config["open_angle"]), float(config["closed_angle"])
    if not all(np.isfinite(v) for v in (angle, opened, closed)) or abs(opened - closed) < 0.05:
        raise ValueError("夹爪标定无效：开闭角度差至少为 0.05 rad")
    if not min(opened, closed) - 0.15 <= angle <= max(opened, closed) + 0.15:
        raise RuntimeError("GELLO 夹爪超出标定范围或编码器跨圈，请重新标定")
    return float(np.clip((angle - closed) / (opened - closed), 0.0, 1.0))


def calibrate_gripper(reader, config, path, motor_id):
    print("只采集 GELLO 夹爪，不连接或驱动 Franka。")
    samples = []
    for label in ("完全张开", "完全闭合"):
        input(f"将 GELLO 夹爪{label}并保持静止，按回车采样：")
        values = []
        for _ in range(15):
            values.append(reader.read()[-1])
            time.sleep(0.02)
        if not np.all(np.isfinite(values)) or np.ptp(values) > 0.02:
            raise RuntimeError("夹爪采样不稳定，请重新运行标定")
        samples.append(float(np.median(values)))
    grip = {"id": motor_id, "open_angle": samples[0], "closed_angle": samples[1]}
    if not 0.05 <= abs(samples[0] - samples[1]) <= 2 * np.pi:
        raise ValueError("夹爪开闭差太小或跨圈，请检查读数")
    config["gripper"] = grip
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
    print("夹爪标定已保存，下次启动自动启用：", grip)


class GripperFollower:
    """Send current targets promptly; stop obsolete in-flight moves asynchronously."""
    def __init__(self, gripper, speed, debug=False, mode="proportional"):
        self.gripper = gripper
        self.speed = speed
        self.debug = debug
        self.mode = mode
        self.binary_open = None
        self.max_width = float(gripper.max_width)
        if not np.isfinite(self.max_width) or not 0 < self.max_width <= 0.2:
            raise RuntimeError("Franka Hand 最大开度无效，请先在设备端完成夹爪 homing")
        self.future = None
        self.stop_future = None
        self.last_width = None
        self.sent_at = 0.0
        self.debug_at = 0.0
        self.debug_width = None

    def log(self, message):
        if self.debug:
            print(f"[夹爪 {time.monotonic():.3f}] {message}", flush=True)

    def update(self, fraction):
        if self.mode == "binary":
            if self.binary_open is None:
                self.binary_open = fraction >= 0.5
            elif self.binary_open and fraction <= 0.35:
                self.binary_open = False
            elif not self.binary_open and fraction >= 0.65:
                self.binary_open = True
            fraction = float(self.binary_open)
        width = self.max_width * fraction
        # Encoder noise near fully open/closed must not start a new stroke.
        if width < 0.001:
            width = 0.0
        elif self.max_width - width < 0.001:
            width = self.max_width
        now = time.monotonic()
        if (self.debug and now - self.debug_at >= 0.25
                and (self.debug_width is None or abs(width - self.debug_width) >= 0.0005)):
            self.log(f"GELLO 当前目标 {width * 1000:.1f} mm")
            self.debug_width, self.debug_at = width, now
        # stop_async may finish before the interrupted move future: drain both.
        if self.stop_future is not None:
            if not self.stop_future.wait(timeout=0):
                return
            if self.future is not None and not self.future.wait(timeout=0):
                return
            if not self.stop_future.get():
                raise RuntimeError("夹爪取消旧动作失败")
            if self.future is not None:
                self.future.get()  # False is expected for an intentionally stopped move.
            self.future = self.stop_future = None
            self.last_width = None
            self.log("旧动作已停止，准备发送最新目标")
        if self.future is not None:
            if self.future.wait(timeout=0):
                if not self.future.get():
                    raise RuntimeError("Franka 夹爪未完成开合，可能遇到物体或未初始化")
                self.future = None
                self.log(f"动作完成，用时 {(now - self.sent_at) * 1000:.0f} ms")
            else:
                # Do not wait a full stroke for an obsolete command, including
                # the automatic initial command sent before the user's first press.
                if abs(width - self.last_width) >= 0.004 and now - self.sent_at >= 0.15:
                    self.log(f"新目标 {width * 1000:.1f} mm，取消旧目标 {self.last_width * 1000:.1f} mm")
                    self.stop_future = self.gripper.stop_async()
                return
        # 0.5 mm deadband avoids delaying the start of a slow squeeze by 2 mm.
        if self.last_width is None or abs(width - self.last_width) >= 0.0005:
            started = time.monotonic()
            self.future = self.gripper.move_async(width, self.speed)
            self.last_width, self.sent_at = width, started
            self.log(f"已下发 {width * 1000:.1f} mm，调用耗时 {(time.monotonic() - started) * 1000:.1f} ms")


def gripper_worker(ip, speed, debug, latest, shutdown, status, factory=None, mode="proportional"):
    """Own all gripper SDK calls in a separate interpreter (including their GIL)."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    hand = None
    try:
        if factory is None:
            from franky import Gripper
            factory = Gripper
        hand = factory(ip)
        follower = GripperFollower(hand, speed, debug, mode)
        status.send(("ready", follower.max_width))
        while not shutdown.is_set():
            stamp = latest[1]
            fraction = latest[0]
            if stamp > 0:
                if time.monotonic() - stamp > 0.3:
                    raise RuntimeError("主循环夹爪目标超过 0.3 秒未更新")
                follower.update(fraction)
            shutdown.wait(0.01)
    except BaseException as exc:
        status.send(("error", str(exc)))
    finally:
        if hand is not None:
            try:
                hand.stop()
            except Exception as exc:
                try:
                    status.send(("error", f"夹爪停止失败: {exc}"))
                except (BrokenPipeError, OSError):
                    pass
        status.close()


class GripperProcess:
    """Main loop only publishes latest targets; it never calls the gripper SDK."""
    def __init__(self, ip, speed, debug=False, factory=None, mode="proportional"):
        ctx = mp.get_context("spawn")  # Never fork an active robot control thread.
        self.latest = ctx.Array("d", [0.0, 0.0], lock=False)
        self.shutdown = ctx.Event()
        self.status, sender = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=gripper_worker,
                                   args=(ip, speed, debug, self.latest,
                                         self.shutdown, sender, factory, mode),
                                   name="franka-gripper")
        self.process.start()
        sender.close()
        try:
            if not self.status.poll(10):
                raise RuntimeError("连接夹爪超时")
            kind, value = self.status.recv()
            if kind != "ready":
                raise RuntimeError(f"夹爪初始化失败: {value}")
            self.max_width = value
        except BaseException:
            self.close()
            raise

    def update(self, fraction):
        while self.status.poll():
            try:
                kind, value = self.status.recv()
            except EOFError:
                raise RuntimeError("夹爪控制进程已退出") from None
            if kind == "error":
                raise RuntimeError(f"夹爪控制失败: {value}")
        if not self.process.is_alive():
            raise RuntimeError("夹爪控制进程意外退出")
        self.latest[0] = fraction
        self.latest[1] = time.monotonic()

    def close(self):
        self.shutdown.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            # Closing a socket does not itself stop an ongoing hardware move.
            print("夹爪进程未能在 5 秒内退出，请检查设备状态。", file=sys.stderr)
            self.process.terminate()
            self.process.join(timeout=1)
        self.status.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("gello_calibration.json"))
    parser.add_argument("--ip", help="覆盖标定文件中的 Franka IP")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--dynamics", type=float, default=0.20,
                        help="遥操作速度比例，范围 (0, 0.2]")
    parser.add_argument("--acceleration", type=float, default=0.05,
                        help="遥操作加速度比例，范围 (0, 0.2]")
    parser.add_argument("--jerk", type=float, default=0.02,
                        help="遥操作 jerk 比例，范围 (0, 0.2]")
    parser.add_argument("--alignment-dynamics", type=float, default=0.05,
                        help="启动对齐的动态比例，范围 (0, 0.2]")
    parser.add_argument("--max-gello-speed", type=float, default=2.0, help="rad/s")
    parser.add_argument("--max-tracking-error", type=float, default=0.5, help="rad")
    parser.add_argument("--preview", action="store_true", help="只读 GELLO，不连接 Franka")
    parser.add_argument("--calibrate-gripper", action="store_true",
                        help="仅采集 GELLO 夹爪开闭角度，保存后退出，不连接 Franka")
    parser.add_argument("--gripper-id", type=int, default=None, help="夹爪 ID，首次默认 8")
    parser.add_argument("--gripper-speed", type=float, default=0.08, help="Franka Hand 开合速度 m/s")
    parser.add_argument("--gripper-mode", choices=("proportional", "binary"),
                        default="proportional", help="连续开度映射，或带阈值的开/合切换")
    parser.add_argument("--gripper-debug", action="store_true", help="打印夹爪目标、下发和完成时刻以定位延迟")
    parser.add_argument("--no-gripper", action="store_true", help="禁用已标定的夹爪映射")
    a = parser.parse_args()
    if not (np.isfinite(a.hz) and 10 <= a.hz <= 100
            and np.isfinite(a.dynamics) and 0 < a.dynamics <= 0.2
            and np.isfinite(a.acceleration) and 0 < a.acceleration <= 0.2
            and np.isfinite(a.jerk) and 0 < a.jerk <= 0.2
            and np.isfinite(a.alignment_dynamics) and 0 < a.alignment_dynamics <= 0.2
            and np.isfinite(a.max_gello_speed) and a.max_gello_speed > 0
            and np.isfinite(a.max_tracking_error) and a.max_tracking_error > 0):
        parser.error("频率范围 10~100 Hz，各动态比例范围 (0, 0.2]，其余阈值须为有限正数")
    c, signs, encoder_ref, robot_ref = load_config(a.config)
    if not np.isfinite(a.gripper_speed) or not 0 < a.gripper_speed <= 0.5:
        parser.error("gripper-speed 范围 (0, 0.5] m/s")
    if a.calibrate_gripper and a.no_gripper:
        parser.error("标定夹爪时不能同时指定 --no-gripper")
    grip_config = c.get("gripper")
    use_gripper = a.calibrate_gripper or (grip_config is not None and not a.no_gripper)
    motor_id = a.gripper_id if a.gripper_id is not None else (grip_config or {}).get("id", 8)
    if use_gripper:
        if type(motor_id) is not int or not 0 <= motor_id <= 252 or motor_id in c["ids"]:
            parser.error("夹爪 ID 必须在 0~252 内且不同于七个手臂电机 ID")
        if not a.calibrate_gripper and motor_id != grip_config["id"]:
            parser.error("夹爪 ID 变化后需要重新运行 --calibrate-gripper")
        if not a.calibrate_gripper:
            gripper_fraction(float(grip_config["open_angle"]), grip_config)
    robot = None
    hand = None
    follower = None
    motion_started = False
    reader = GelloReader(c["port"], c["baudrate"], c["ids"] + ([motor_id] if use_gripper else []))
    try:
        if a.calibrate_gripper:
            calibrate_gripper(reader, c, a.config, motor_id)
            return
        targets = Targets(reader, signs, encoder_ref, robot_ref, a.max_gello_speed)
        target = targets.read()
        if a.preview:
            print("预览模式；不会连接 Franka。Ctrl+C 退出。")
            while True:
                fraction = (gripper_fraction(targets.gripper_angle, grip_config)
                            if use_gripper else None)
                print("目标(rad):", np.round(target, 4), "夹爪张开比例:", fraction, flush=True)
                time.sleep(1 / a.hz)
                target = targets.read()
        from franky import JointMotion, JointState, JointStopMotion, RelativeDynamicsFactor, Robot
        robot = Robot(a.ip or c["ip"])
        if use_gripper:
            hand = GripperProcess(a.ip or c["ip"], a.gripper_speed, a.gripper_debug, mode=a.gripper_mode)
            follower = hand
            print(f"启用原装 Franka Hand 映射，GELLO ID={motor_id}，最大开度 {follower.max_width:.3f} m，模式 {a.gripper_mode}")
        else:
            print("夹爪未启用；首次请运行 --calibrate-gripper。")
        robot.relative_dynamics_factor = a.alignment_dynamics
        print("启动对齐，保持 GELLO 静止。目标(rad):", np.round(target, 4), flush=True)
        initial_target = target.copy()
        motion_started = True
        robot.move(JointMotion(JointState(initial_target)), asynchronous=True, limit_rate=True)
        while not robot.join_motion(timeout=0):
            time.sleep(1 / a.hz)
            target = targets.read()
            if np.max(np.abs(target - initial_target)) > 0.04:
                raise RuntimeError("启动对齐期间 GELLO 移动，请保持静止后重新启动")
        current = vector(robot.state.q, "Franka 状态")
        if np.max(np.abs(current - initial_target)) > 0.05:
            raise RuntimeError("Franka 未到达启动目标")
        # Keep velocity responsive without scaling acceleration and jerk up with it.
        robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity=a.dynamics, acceleration=a.acceleration, jerk=a.jerk
        )
        print(f"对齐完成，开始遥操作（{a.hz:g} Hz，速度 {a.dynamics:g}，"
              f"加速度 {a.acceleration:g}，jerk {a.jerk:g}）；"
              "Ctrl+C 减速停止。", flush=True)
        while True:
            started = time.monotonic()
            target = targets.read()
            current = vector(robot.state.q, "Franka 状态")
            if np.max(np.abs(target - current)) > a.max_tracking_error:
                raise RuntimeError("Franka 跟随误差过大，停止遥操作")
            if follower is not None:
                follower.update(gripper_fraction(targets.gripper_angle, grip_config))
            # Keep the 1 kHz franky control loop alive between Python updates.
            # Otherwise a reached target can finish the motion between samples.
            robot.move(JointMotion(JointState(target), return_when_finished=False),
                       asynchronous=True, limit_rate=True)
            time.sleep(max(0.0, 1 / a.hz - (time.monotonic() - started)))
    except BaseException:
        if robot is not None:
            try:
                state = robot.state
                print(f"Franka 模式: {state.robot_mode}\n"
                      f"当前错误: {state.current_errors}\n"
                      f"上次运动错误: {state.last_motion_errors}", file=sys.stderr)
            except Exception as diagnostic_error:
                print(f"无法读取诊断状态: {diagnostic_error}", file=sys.stderr)
        raise
    finally:
        original_error = sys.exc_info()[1]
        try:
            if robot is not None and motion_started:
                from franky import JointStopMotion
                try:
                    robot.move(JointStopMotion(), asynchronous=True, limit_rate=True)
                    robot.join_motion()
                except Exception as stop_error:
                    if original_error is None:
                        raise
                    print(f"停止命令未执行成功（保留原始异常）: {stop_error}", file=sys.stderr)
        finally:
            try:
                if hand is not None:
                    try:
                        hand.close()
                    except Exception as stop_error:
                        print(f"夹爪停止命令失败: {stop_error}", file=sys.stderr)
            finally:
                reader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已请求停止。")
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
