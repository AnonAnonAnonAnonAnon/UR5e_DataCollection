#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小 RTDE 控制脚本：配合官方 rtde_control_loop.urp
效果：在两个 hardcode 的 TCP pose 之间来回 MoveL

依赖：UniversalRobots RTDE_Python_Client_Library (rtde-2.7.12)
"""

# =======================
# CONFIG（只改这里）
# =======================
import sys

RTDE_PKG_DIR = "/home/zhangw/UR5e_DataCollection/rtde-2.7.12-release/rtde-2.7.12"
ROBOT_HOST   = "192.168.0.3"
ROBOT_PORT   = 30004

# [-0.38872729368772857, -0.46542313240396643, 0.3381279652796633, 1.060678209976973, 2.9434795530453237, -0.11223619309082138]

# 两个 TCP pose（单位：m / rad；格式：[x, y, z, rx, ry, rz]）
POSE_A = [0.38, -0.46, 0.33,  1.06, 2.94, -0.11]
POSE_B = [0.45, -0.46, 0.33,  1.06, 2.94, -0.11]
# =======================

sys.path.append(RTDE_PKG_DIR)

import rtde.rtde as rtde


def main():
    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    con.connect()
    con.get_controller_version()

    # 只订阅握手位即可（URP 会写 output_int_register_0）
    con.send_output_setup(["output_int_register_0"], ["INT32"], frequency=10)

    # 一个输入 recipe：6 个 setpoint + 1 个 watchdog/ack
    inp = con.send_input_setup(
        [
            "input_double_register_0",
            "input_double_register_1",
            "input_double_register_2",
            "input_double_register_3",
            "input_double_register_4",
            "input_double_register_5",
            "input_int_register_0",
        ],
        ["DOUBLE"] * 6 + ["INT32"],
    )

    # 初值：watchdog=0；setpoint 随便（反正 URP 会等到 output==1 才触发）
    inp.input_int_register_0 = 0

    # ===== 关键修复：把所有 input_* 都初始化，否则 pack() 会报 Uninitialized parameter =====
    inp.input_double_register_0 = POSE_A[0]
    inp.input_double_register_1 = POSE_A[1]
    inp.input_double_register_2 = POSE_A[2]
    inp.input_double_register_3 = POSE_A[3]
    inp.input_double_register_4 = POSE_A[4]
    inp.input_double_register_5 = POSE_A[5]
    inp.input_int_register_0 = 0
    # ==============================================================================

    con.send_start()

    next_pose = POSE_A
    sent = False  # 是否已对本轮 output==1 发送过 setpoint

    try:
        while True:
            state = con.receive()
            if state is None:
                break

            req = state.output_int_register_0

            # URP 请求下一点：output==1
            if (req == 1) and (not sent):
                inp.input_double_register_0 = next_pose[0]
                inp.input_double_register_1 = next_pose[1]
                inp.input_double_register_2 = next_pose[2]
                inp.input_double_register_3 = next_pose[3]
                inp.input_double_register_4 = next_pose[4]
                inp.input_double_register_5 = next_pose[5]
                inp.input_int_register_0 = 1   # 告诉 URP：新点已下发（并作为 watchdog kick）

                print("Send pose:", next_pose)

                next_pose = POSE_B if next_pose == POSE_A else POSE_A
                sent = True

            # URP 已执行完 moveL：output==0（此时 URP 在 while 等你把 input_int_register_0 置回 0）
            elif (req == 0) and sent:
                inp.input_int_register_0 = 0
                sent = False

            # 每轮都发一次，既是 watchdog kick，也把 ack/当前 setpoint 持续刷新给控制器
            con.send(inp)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            con.send_pause()
        except Exception:
            pass
        con.disconnect()


if __name__ == "__main__":
    main()
