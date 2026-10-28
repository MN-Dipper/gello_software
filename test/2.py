import argparse
import json
import time

import numpy as np

from calibrate_gello import DEFAULT_CONFIG, GelloReader


def vector(config, key):
    result = np.asarray(config[key], dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"标定文件字段 {key} 无效")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        c = json.load(f)
    if c.get("schema") != 1:
        raise ValueError("不支持的标定文件版本")
    signs = vector(c, "signs")
    reference = vector(c, "robot_reference")
    encoder_reference = vector(c, "encoder_reference")
    if not np.all(np.isin(signs, [-1, 1])):
        raise ValueError("signs 必须为 +1/-1")
    if (len(c["ids"]) != 7 or len(set(c["ids"])) != 7
            or any(type(i) is not int or not 0 <= i <= 252 for i in c["ids"])
            or type(c["baudrate"]) is not int or c["baudrate"] <= 0):
        raise ValueError("标定文件 IDs 或波特率无效")

    reader = GelloReader(c["port"], c["baudrate"], c["ids"])
    try:
        robot = None
        if a.execute:
            from franky import JointMotion, Robot
            robot = Robot(c["ip"])
            robot.relative_dynamics_factor = 0.03
            print("将实际控制 Franka。请清空工作区域，保持急停可用。")
            print("关闭其他 Franka 控制程序；保持 GELLO 与标定姿态接近。")
            if input("输入 MOVE 启动：").strip() != "MOVE":
                return

        previous_encoder = None
        first = True
        print("模式:", "实际低速分段跟随" if a.execute else "仅预览，不连接 Franka")
        while True:
            # Always acquire a fresh sample; never integrate across failed reads.
            encoder = reader.read()
            if previous_encoder is not None:
                if np.max(np.abs(encoder - previous_encoder)) > 0.15:
                    raise RuntimeError("GELLO 角度突变、跨圈或移动过快，退出")
            previous_encoder = encoder.copy()
            target = reference + signs * (encoder - encoder_reference)
            if not np.all(np.isfinite(target)):
                raise RuntimeError("无效目标")
            if np.max(np.abs(target - reference)) > 0.2:
                raise RuntimeError("超过参考姿态 ±0.2 rad 测试范围，请减小动作")
            print("目标(rad):", np.round(target, 4), flush=True)

            if robot is not None:
                current = np.array(robot.state.q, dtype=float)
                if current.shape != (7,) or not np.all(np.isfinite(current)):
                    raise RuntimeError("机器人关节状态无效")
                error = target - current
                threshold = 0.05 if first else 0.15
                if np.max(np.abs(error)) > threshold:
                    raise RuntimeError(
                        "目标与机器人姿态差过大。请先退出并人工对齐，"
                        "不要用程序强行同步"
                    )
                first = False
                step = current + np.clip(error, -0.01, 0.01)
                if np.max(np.abs(step - current)) > 0.001:
                    # Blocking call: finish a bounded segment before reading again.
                    # It deliberately does not queue or stream async targets.
                    robot.move(JointMotion(step.tolist()))
            time.sleep(0.05)
    finally:
        reader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出；若正在执行同步运动，当前小段可能先完成。")