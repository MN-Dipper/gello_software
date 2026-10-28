import argparse
import math
import time

from dynamixel_sdk import (
    COMM_SUCCESS, GroupSyncRead, PacketHandler, PortHandler,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        default="/dev/serial/by-id/"
                "usb-FTDI_USB__-__Serial_Converter_FTAOWIWQ-if00-port0",
    )
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--ids", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6, 7])
    args = parser.parse_args()
    if (len(set(args.ids)) != len(args.ids)
            or any(i < 0 or i > 252 for i in args.ids)
            or args.baudrate <= 0):
        parser.error("电机 ID 必须唯一且在 0~252 内，波特率必须大于零")

    port = PortHandler(args.port)
    packet = PacketHandler(2.0)
    reader = GroupSyncRead(port, packet, 132, 4)

    try:
        if not port.openPort():
            raise RuntimeError("无法打开串口，请检查路径、权限和串口占用")
        # 仅设置电脑端串口速率，不修改电机波特率。
        if not port.setBaudRate(args.baudrate):
            raise RuntimeError("无法设置串口波特率")

        for motor_id in args.ids:
            model, result, error = packet.ping(port, motor_id)
            if result != COMM_SUCCESS:
                raise RuntimeError(
                    f"ID {motor_id}: {packet.getTxRxResult(result)}；"
                    "请检查电源、电机 ID 和波特率"
                )
            if error:
                raise RuntimeError(
                    f"ID {motor_id}: {packet.getRxPacketError(error)}"
                )
            print(f"已连接 ID={motor_id}, model={model}")
            if not reader.addParam(motor_id):
                raise RuntimeError(f"无法添加电机 ID {motor_id}")

        print("显示未经零位和方向标定的编码器角度，Ctrl+C 退出。")
        while True:
            result = reader.txRxPacket()
            if result != COMM_SUCCESS:
                raise RuntimeError(packet.getTxRxResult(result))

            values = []
            for motor_id in args.ids:
                if not reader.isAvailable(motor_id, 132, 4):
                    raise RuntimeError(f"电机 ID {motor_id} 无有效读数")
                ticks = reader.getData(motor_id, 132, 4)
                if ticks >= 2**31:
                    ticks -= 2**32
                radians = ticks * 2 * math.pi / 4096
                values.append(
                    f"ID{motor_id}: {radians:+.4f}rad"
                    f" ({math.degrees(radians):+.2f}°)"
                )

            print(" | ".join(values), flush=True)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n已停止读取。")
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"读取失败：{exc}")
    finally:
        reader.clearParam()
        port.closePort()


if __name__ == "__main__":
    main()