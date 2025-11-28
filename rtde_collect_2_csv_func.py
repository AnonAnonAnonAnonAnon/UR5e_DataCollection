# 规范的函数封装

# 简单 RTDE 自检脚本：每秒打印一次当前 TCP 末端位姿
# 收集到csv，自行简单实现，不用rtde的csv模块
#
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

# 文件操作方面的准备
def prepare_rtde_run(data_dir, robot_host, robot_port, frequency_hz):
    """
    准备一次 RTDE 采集：
    - 确保目录存在
    - 生成时间戳命名
    - 写配置 JSON
    - 返回 csv_path, cfg_path
    """
    os.makedirs(data_dir, exist_ok=True)

    run_time = datetime.now()
    ts_str = run_time.strftime("%Y%m%d_%H%M%S")

    csv_filename = f"rtde_tcp_{ts_str}.csv"
    cfg_filename = f"rtde_tcp_{ts_str}.json"

    csv_path = os.path.join(data_dir, csv_filename)
    cfg_path = os.path.join(data_dir, cfg_filename)

    config = {
        "run_start_time": run_time.isoformat(),
        "file_timestamp": ts_str,
        "robot_host": robot_host,
        "robot_port": robot_port,
        "frequency_hz": frequency_hz,
        "rtde_fields": ["timestamp", "actual_TCP_pose"],
        "data_format": {
            "csv_columns": [
                "controller_time_s",
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "tcp_rx",
                "tcp_ry",
                "tcp_rz",
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

    return csv_path, cfg_path

# 网络链接和 RTDE 配置
def connect_rtde(robot_host, robot_port, frequency_hz):
    """
    建立 RTDE 连接并配置输出：
    - 连接控制箱
    - 获取版本
    - 配置 timestamp + actual_TCP_pose 输出
    - 启动同步
    返回连接对象 con
    """
    print(f"Connecting to {robot_host}:{robot_port} ...")
    con = rtde.RTDE(robot_host, robot_port)
    con.connect()

    # 可选：确认控制器版本
    con.get_controller_version()

    # 配置输出字段
    if not con.send_output_setup(["timestamp", "actual_TCP_pose"], frequency=frequency_hz):
        print("Failed to setup RTDE output (timestamp, actual_TCP_pose).")
        con.disconnect()
        sys.exit(1)

    # 开始数据同步
    if not con.send_start():
        print("Failed to start RTDE data synchronization.")
        con.disconnect()
        sys.exit(1)

    return con

# 文件操作
def record_rtde_tcp_to_csv(con, csv_path):
    """
    从 RTDE 持续读取 TCP pose，写入 csv_path。
    按 Ctrl+C 终止。
    """
    print(f"RTDE started. Recording TCP pose.")
    print(f"CSV will be written to {csv_path}")
    print("Press Ctrl+C to stop.\n")

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "controller_time_s",
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "tcp_rx",
                "tcp_ry",
                "tcp_rz",
            ]
        )

        try:
            while True:
                state = con.receive()
                if state is None:
                    print("RTDE connection closed by robot.")
                    break

                ts = state.timestamp          # 控制器时间，单位：秒（浮点）
                tcp = state.actual_TCP_pose   # [x, y, z, rx, ry, rz]

                writer.writerow([ts] + list(tcp))

                # 控制台 echo（方便调试）
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


def main():
    # 在 main 里统一定义参数，方便之后联合相机脚本
    ROBOT_HOST = "192.168.0.3"   # UR 控制箱 IP
    ROBOT_PORT = 30004           # RTDE 默认端口
    FREQUENCY_HZ = 1             # 采样频率（当前是 1 Hz）
    DATA_DIR = "/home/zhangw/UR5e_DataCollection/action_data"

    csv_path, cfg_path = prepare_rtde_run(
        data_dir=DATA_DIR,
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        frequency_hz=FREQUENCY_HZ,
    )

    con = connect_rtde(
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        frequency_hz=FREQUENCY_HZ,
    )

    record_rtde_tcp_to_csv(con, csv_path)

    print(f"Config saved to {cfg_path}")


if __name__ == "__main__":
    main()
