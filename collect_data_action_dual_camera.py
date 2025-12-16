# 联合采集脚本：同时记录 RTDE TCP pose + 双 RealSense 彩色图像（Head + Wrist）
# - 按 q 退出
# - 依赖 rtde_collect_2_csv_func.py 与 realsense_dual_collect_2_folder_func.py

import os
import time
import csv

import numpy as np
import cv2

from rtde_collect_2_csv_func import (
    prepare_rtde_run,
    connect_rtde,
)

from realsense_dual_collect_2_folder_func import (
    prepare_dual_camera_run,
    start_realsense_pipeline,
    list_serials,   # 如果你的 dual func 文件里没有这个函数，把这一行删掉即可
)


def main():
    # ===== 统一参数设置 =====
    # 机械臂 / RTDE
    ROBOT_HOST = "192.168.0.3"
    ROBOT_PORT = 30004
    RTDE_HZ = 10
    ACTION_DATA_DIR = "/home/zhangw/UR5e_DataCollection/action_data"

    # 相机（双）
    CAMERA_BASE_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"
    CAM_SAVE_HZ = 10.0
    COLOR_WIDTH = 640
    COLOR_HEIGHT = 480
    COLOR_FPS = 30

    # 双相机序列号（按你的实际情况填写）
    HEAD_SERIAL = "243522072333"
    WRIST_SERIAL = "233522079334"

    # =====（可选）启动前检查相机是否都在 =====
    try:
        serials = list_serials()
        print("检测到相机序列号：", serials)
        if HEAD_SERIAL not in serials:
            print(f"未检测到 HEAD_SERIAL={HEAD_SERIAL}，请检查序列号或USB连接。")
            return
        if WRIST_SERIAL not in serials:
            print(f"未检测到 WRIST_SERIAL={WRIST_SERIAL}，请检查序列号或USB连接。")
            return
    except Exception:
        # 你的 realsense_dual_collect_2_folder_func.py 如果没提供 list_serials，就忽略检查
        pass

    # ===== 准备文件 / 目录 =====
    csv_path, cfg_rtde_path = prepare_rtde_run(
        data_dir=ACTION_DATA_DIR,
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        frequency_hz=RTDE_HZ,
    )

    run_dir, head_dir, wrist_dir, cfg_cam_path = prepare_dual_camera_run(
        base_dir=CAMERA_BASE_DIR,
        save_hz=CAM_SAVE_HZ,
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
        head_serial=HEAD_SERIAL,
        wrist_serial=WRIST_SERIAL,
    )

    # ===== 建立 RTDE 连接 & 启动双相机 =====
    con = connect_rtde(
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        frequency_hz=RTDE_HZ,
    )

    pipe_head = start_realsense_pipeline(
        serial=HEAD_SERIAL,
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )
    pipe_wrist = start_realsense_pipeline(
        serial=WRIST_SERIAL,
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )

    # ===== 联合采集循环 =====
    next_save_rtde = time.time()
    next_save_cam = time.time()
    frame_idx = 0

    print("联合采集开始（双相机），按 'q' 退出。")
    print(f"RTDE CSV: {csv_path}")
    print(f"Camera folder: {run_dir}")

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
                # ------ 1) RTDE 读取当前 TCP pose ------
                state = con.receive()
                if state is None:
                    print("RTDE connection closed by robot.")
                    break

                ts = state.timestamp
                tcp = state.actual_TCP_pose  # [x, y, z, rx, ry, rz]

                # ------ 2) 双相机读取当前画面 ------
                frames_head = pipe_head.wait_for_frames()
                frames_wrist = pipe_wrist.wait_for_frames()

                color_head = frames_head.get_color_frame()
                color_wrist = frames_wrist.get_color_frame()
                if (not color_head) or (not color_wrist):
                    continue

                img_head = np.asanyarray(color_head.get_data())
                img_wrist = np.asanyarray(color_wrist.get_data())

                cv2.imshow("RealSense HEAD Color", img_head)
                cv2.imshow("RealSense WRIST Color", img_wrist)

                now = time.time()

                # ------ 3) 按 RTDE_HZ 频率保存 TCP pose ------
                if now >= next_save_rtde:
                    writer.writerow([ts] + list(tcp))
                    print(f"[RTDE] {ts:8.3f} s | TCP pose = {tcp}")
                    next_save_rtde += 1.0 / RTDE_HZ

                # ------ 4) 按 CAM_SAVE_HZ 频率保存双相机图片 ------
                if now >= next_save_cam:
                    frame_idx += 1

                    head_name = f"frame_{frame_idx:05d}.png"
                    wrist_name = f"frame_{frame_idx:05d}.png"

                    head_path = os.path.join(head_dir, head_name)
                    wrist_path = os.path.join(wrist_dir, wrist_name)

                    cv2.imwrite(head_path, img_head)
                    cv2.imwrite(wrist_path, img_wrist)

                    print(f"[CAM ] [{frame_idx:05d}] 已保存: {head_path} | {wrist_path}")
                    next_save_cam += 1.0 / CAM_SAVE_HZ

                # ------ 5) 键盘监听：按 q 退出 ------
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            # 收尾：暂停 RTDE / 断开连接 / 关相机窗口
            try:
                con.send_pause()
            except Exception:
                pass
            con.disconnect()

            pipe_head.stop()
            pipe_wrist.stop()
            cv2.destroyAllWindows()

            print("联合采集结束。")
            print(f"RTDE config:   {cfg_rtde_path}")
            print(f"Camera config: {cfg_cam_path}")


if __name__ == "__main__":
    main()
