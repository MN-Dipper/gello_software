import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS, GroupSyncRead, PacketHandler, PortHandler,
)

DEFAULT_PORT = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FTAOWIWQ-if00-port0"
)
DEFAULT_CONFIG = Path(__file__).with_name("gello_calibration.json")


class GelloReader:
    """Protocol 2.0 X-series, position register 132, 4096 ticks/rev."""

    def __init__(self, port, baudrate, ids):
        self.port = PortHandler(port)
        self.packet = PacketHandler(2.0)
        self.reader = GroupSyncRead(self.port, self.packet, 132, 4)
        self.ids = list(ids)
        try:
            if not self.port.openPort():
                raise RuntimeError("无法打开 GELLO 串口")
            if not self.port.setBaudRate(baudrate):
                raise RuntimeError("无法设置电脑端串口速率")
            for motor_id in self.ids:
                _, result, error = self.packet.ping(self.port, motor_id)
                if result != COMM_SUCCESS:
                    raise RuntimeError(
                        f"ID {motor_id}: {self.packet.getTxRxResult(result)}"
                    )
                if error:
                    raise RuntimeError(
                        f"ID {motor_id}: {self.packet.getRxPacketError(error)}"
                    )
                if not self.reader.addParam(motor_id):
                    raise RuntimeError(f"无法添加 ID {motor_id}")
        except BaseException:
            self.close()
            raise

    def read(self):
        started = time.monotonic()
        result = self.reader.txRxPacket()
        if result != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(result))
        if time.monotonic() - started > 0.3:
            raise RuntimeError("串口读取耗时超过 0.3 秒")
        values = []
        for motor_id in self.ids:
            if not self.reader.isAvailable(motor_id, 132, 4):
                raise RuntimeError(f"ID {motor_id} 数据缺失")
            raw = self.reader.getData(motor_id, 132, 4)
            if raw >= 2**31:
                raw -= 2**32
            values.append(raw * 2 * math.pi / 4096)
        return np.array(values, dtype=float)

    def close(self):
        self.reader.clearParam()
        if self.port.ser is not None:
            self.port.closePort()


def stable_sample(reader):
    samples = []
    for _ in range(15):
        samples.append(reader.read())
        time.sleep(0.02)
    samples = np.array(samples)
    if np.max(np.ptp(samples, axis=0)) > 0.02:
        raise RuntimeError("采样时 GELLO 移动或编码器跨圈，请保持静止重新标定")
    return np.median(samples, axis=0)


def sample_pair(reader, robot):
    """Read corresponding stationary poses without commanding either device."""
    before = np.asarray(robot.state.q, dtype=float)
    encoder = stable_sample(reader)
    state = robot.state
    joints = np.asarray(state.q, dtype=float)
    velocity = np.asarray(state.dq, dtype=float)
    if (before.shape != (7,) or joints.shape != (7,) or velocity.shape != (7,)
            or not all(np.all(np.isfinite(v)) for v in (before, joints, velocity))
            or np.max(np.abs(joints - before)) > 0.01
            or np.max(np.abs(velocity)) > 0.02):
        raise RuntimeError("Franka 状态无效或正在运动，请静止后重试")
    return encoder, joints


def input_signs():
    print("静态标定：无需移动任何机械臂。")
    print("两台机械臂保持形状一致；程序会自动计算零位偏移。")
    print("方向无法由单个静态姿态判断，需要设备配置中的 J1~J7 方向。")
    while True:
        text = input("输入七个方向（空格分隔的 1/-1；Ctrl+C 取消）：")
        try:
            signs = [int(item) for item in text.replace(",", " ").split()]
        except ValueError:
            signs = []
        if len(signs) == 7 and all(sign in (-1, 1) for sign in signs):
            return signs
        print("请输入恰好七个 1 或 -1，程序不会使用未经确认的默认方向。")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baudrate", type=int, default=57600)
    p.add_argument("--ip", default="172.16.0.2")
    p.add_argument("--ids", type=int, nargs=7, default=list(range(1, 8)))
    p.add_argument("--signs", type=int, nargs=7, default=None,
                   help="可选：已知方向时传入七个 +1/-1；省略则在终端输入")
    p.add_argument("--signs-from", type=Path, help="从已有标定 JSON 复用方向，无需重新输入")
    p.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    a = p.parse_args()
    if (any(s not in (-1, 1) for s in (a.signs or []))
            or len(set(a.ids)) != 7
            or any(i < 0 or i > 252 for i in a.ids)
            or a.baudrate <= 0):
        p.error("请检查 signs、IDs 和波特率")
    if a.output.exists():
        p.error(f"{a.output} 已存在；如需重新标定，请先重命名保存旧文件")

    if a.signs is not None and a.signs_from is not None:
        p.error("--signs 与 --signs-from 只能选一个")
    signs = a.signs
    if a.signs_from is not None:
        with a.signs_from.open(encoding="utf-8") as f:
            signs = json.load(f)["signs"]
        if (not isinstance(signs, list) or len(signs) != 7
                or any(type(v) is not int or v not in (-1, 1) for v in signs)):
            p.error("已有配置中的 signs 无效")
    if signs is None:
        signs = input_signs()

    from franky import Robot
    robot = Robot(a.ip)  # 连接并读状态，不发送运动命令
    reader = GelloReader(a.port, a.baudrate, a.ids)
    try:
        print("\n零位采样：将两台机械臂摆成物理形状对应的参考姿态。")
        print("不要求机械零位；这一对姿态用于计算编码器到 Franka 的偏移。")
        input("两边保持静止后，按回车保存标定：")
        encoder, q_robot = sample_pair(reader, robot)
        config = {
            "schema": 1,
            "ip": a.ip,
            "port": a.port,
            "baudrate": a.baudrate,
            "ids": a.ids,
            "signs": signs,
            "encoder_reference": encoder.tolist(),
            "robot_reference": q_robot.tolist(),
        }
        # Exclusive creation prevents accidentally overwriting calibration.
        with a.output.open("x", encoding="utf-8") as f:
            json.dump(config, f, indent=2, allow_nan=False)
        print("Franka 参考角度(rad):", np.round(q_robot, 4))
        print("GELLO 参考角度(rad): ", np.round(encoder, 4))
        print("方向映射:", signs)
        print("已保存:", a.output)
        print("标定只需做一次；之后直接启动 teleop_franka.py。")
    finally:
        reader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消")