#!/usr/bin/env python
# 简单 RTDE 自检脚本：每秒打印一次当前 TCP 末端位姿

import sys

# 把 rtde 包所在目录加进 Python 搜索路径
sys.path.append("/home/zhangw/UR5e_DataCollection/rtde-2.7.12-release/rtde-2.7.12")

import rtde.rtde as rtde  # 官方 RTDE Python 客户端

ROBOT_HOST = "192.168.0.3"   # 你的 UR 控制箱 IP
ROBOT_PORT = 30004           # RTDE 默认端口


def main():
    print(f"Connecting to {ROBOT_HOST}:{ROBOT_PORT} ...")

    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)

    # 官方示例就是直接调用，不检查返回值
    con.connect()

    # 可选：确认控制器版本
    con.get_controller_version()

    # 配置输出字段：
    #   - timestamp: 控制器时间戳
    #   - actual_TCP_pose: 当前末端位姿 [x, y, z, rx, ry, rz]
    if not con.send_output_setup(["timestamp", "actual_TCP_pose"], frequency=1):
        print("Failed to setup RTDE output (timestamp, actual_TCP_pose).")
        con.disconnect()
        sys.exit(1)

    # 开始数据同步
    if not con.send_start():
        print("Failed to start RTDE data synchronization.")
        con.disconnect()
        sys.exit(1)

    print("RTDE started. Printing TCP pose at 1 Hz. Press Ctrl+C to stop.\n")

    try:
        while True:
            state = con.receive()
            if state is None:
                print("RTDE connection closed by robot.")
                break

            ts = state.timestamp
            tcp = state.actual_TCP_pose  # [x, y, z, rx, ry, rz]

            print(f"{ts:8.3f} s | TCP pose = {tcp}")

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
    finally:
        try:
            con.send_pause()
        except Exception:
            pass
        con.disconnect()


if __name__ == "__main__":
    main()
