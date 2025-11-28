# 简单 RTDE 自检脚本：每秒打印一次当前 TCP 末端位姿
# 收集到csv，自行简单实现，不用rtde的csv模块

# RTDE 采集脚本：
# - 每次运行生成一个带时间戳命名的 CSV + JSON 配置文件
# - CSV: controller_time_s + 6 维 TCP pose
# - JSON: 本次采集的配置（host, port, 频率, 字段等）

import sys
import os
import csv
import json
from datetime import datetime

# 把 rtde 包所在目录加进 Python 搜索路径
sys.path.append("/home/zhangw/UR5e_DataCollection/rtde-2.7.12-release/rtde-2.7.12")

import rtde.rtde as rtde  # 官方 RTDE Python 客户端

ROBOT_HOST = "192.168.0.3"   # 你的 UR 控制箱 IP
ROBOT_PORT = 30004           # RTDE 默认端口
FREQUENCY_HZ = 1             # 采样频率（当前是 1 Hz）

# 数据保存目录
DATA_DIR = "/home/zhangw/UR5e_DataCollection/action_data"


def main():
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. 生成本次采集的时间戳字符串（用于文件名）
    # 例如：20251128_175512
    run_time = datetime.now()
    ts_str = run_time.strftime("%Y%m%d_%H%M%S")

    # 只作为文件名
    csv_filename = f"rtde_tcp_{ts_str}.csv"
    cfg_filename = f"rtde_tcp_{ts_str}.json"

    # 拼成完整路径
    csv_path = os.path.join(DATA_DIR, csv_filename)
    cfg_path = os.path.join(DATA_DIR, cfg_filename)

    # 2. 写配置文件（JSON）
    config = {
        "run_start_time": run_time.isoformat(),
        "file_timestamp": ts_str,
        "robot_host": ROBOT_HOST,
        "robot_port": ROBOT_PORT,
        "frequency_hz": FREQUENCY_HZ,
        "rtde_fields": ["timestamp", "actual_TCP_pose"],
        "data_format": {
            "csv_columns": [
                "controller_time_s",
                "tcp_x", "tcp_y", "tcp_z",
                "tcp_rx", "tcp_ry", "tcp_rz",
            ],
            "tcp_pose_unit": {
                "position": "meters",
                "orientation": "axis-angle radians",
            },
        },
        "notes": "TCP pose stream via RTDE; timestamp is controller time in seconds.",
    }

    with open(cfg_path, "w") as f_cfg:
        json.dump(config, f_cfg, indent=2)

    print(f"Config written to {cfg_path}")

    # 3. 建立 RTDE 连接（按官方示例写法，不检查返回值）
    print(f"Connecting to {ROBOT_HOST}:{ROBOT_PORT} ...")
    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    con.connect()

    # 可选：确认控制器版本
    con.get_controller_version()

    # 4. 配置输出字段：timestamp + actual_TCP_pose
    if not con.send_output_setup(["timestamp", "actual_TCP_pose"], frequency=FREQUENCY_HZ):
        print("Failed to setup RTDE output (timestamp, actual_TCP_pose).")
        con.disconnect()
        sys.exit(1)

    # 5. 开始数据同步
    if not con.send_start():
        print("Failed to start RTDE data synchronization.")
        con.disconnect()
        sys.exit(1)

    print(f"RTDE started. Recording TCP pose at {FREQUENCY_HZ} Hz.")
    print(f"CSV will be written to {csv_path}")
    print("Press Ctrl+C to stop.\n")

    # 6. 打开 CSV，写表头 + 持续写数据
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        # 表头
        writer.writerow([
            "controller_time_s",
            "tcp_x", "tcp_y", "tcp_z",
            "tcp_rx", "tcp_ry", "tcp_rz",
        ])

        try:
            while True:
                state = con.receive()
                if state is None:
                    print("RTDE connection closed by robot.")
                    break

                ts = state.timestamp          # 控制器时间，单位：秒（浮点）
                tcp = state.actual_TCP_pose   # [x, y, z, rx, ry, rz]

                # 写入一行：时间戳 + 6D pose
                writer.writerow([ts] + list(tcp))

                # 控制台也可以简单 echo 一下（可选）
                print(f"{ts:8.3f} s | TCP pose = {tcp}")

        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C).")
        finally:
            try:
                con.send_pause()
            except Exception:
                pass
            con.disconnect()
            print(f"CSV saved to {csv_path}")
            print(f"Config saved to {cfg_path}")


if __name__ == "__main__":
    main()
