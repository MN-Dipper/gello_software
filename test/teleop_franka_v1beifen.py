"""启动时对齐 GELLO，随后使用 franky 异步关节轨迹遥操作（不控制夹爪）。

python test/teleop_franka.py --config test/gello_calibration.json
python test/teleop_franka.py --config test/gello_calibration.json --preview

标定文件使用 test/1.py 生成的 schema=1 格式。参考姿态必须在两台机械臂
物理形状对应时采集；随意将两个不同姿态标为参考，不能得到形状对应关系。
对齐期间请保持 GELLO 静止。Ctrl+C 请求减速停止。
"""
import argparse
import importlib.util
import json
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
        encoder = vector(self.reader.read(), "GELLO 读数")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("gello_calibration.json"))
    parser.add_argument("--ip", help="覆盖标定文件中的 Franka IP")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--dynamics", type=float, default=0.10,
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
    robot = None
    motion_started = False
    reader = GelloReader(c["port"], c["baudrate"], c["ids"])
    try:
        targets = Targets(reader, signs, encoder_ref, robot_ref, a.max_gello_speed)
        target = targets.read()
        if a.preview:
            print("预览模式；不会连接 Franka。Ctrl+C 退出。")
            while True:
                print("目标(rad):", np.round(target, 4), flush=True)
                time.sleep(1 / a.hz)
                target = targets.read()
        from franky import JointMotion, JointState, JointStopMotion, RelativeDynamicsFactor, Robot
        robot = Robot(a.ip or c["ip"])
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
            reader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已请求停止。")
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
