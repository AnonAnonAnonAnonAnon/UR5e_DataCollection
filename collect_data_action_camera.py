# 联合采集脚本：同时记录 RTDE TCP pose 和 RealSense 彩色图像
# - 按 q 退出
# - 依赖 rtde_collect_2_csv_func.py 与 realsense_collect_2_folder_func.py 中的函数

import os
import time
import csv

import numpy as np
import cv2

from rtde_collect_2_csv_func import (
    prepare_rtde_run,
    connect_rtde,
)

from realsense_collect_2_folder_func import (
    prepare_camera_run,
    start_realsense_pipeline,
)


def main():
    # ===== 统一参数设置 =====
    # 机械臂 / RTDE
    ROBOT_HOST = "192.168.0.3"
    ROBOT_PORT = 30004
    RTDE_HZ = 1               # RTDE 输出频率，也作为保存频率
    ACTION_DATA_DIR = "/home/zhangw/UR5e_DataCollection/action_data"

    # 相机
    CAMERA_BASE_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"
    CAM_SAVE_HZ = 1.0
    COLOR_WIDTH = 640
    COLOR_HEIGHT = 480
    COLOR_FPS = 30

    # ===== 准备文件 / 目录（沿用各自模块的函数） =====
    csv_path, cfg_rtde_path = prepare_rtde_run(
        data_dir=ACTION_DATA_DIR,
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        frequency_hz=RTDE_HZ,
    )

    cam_run_dir, cfg_cam_path = prepare_camera_run(
        base_dir=CAMERA_BASE_DIR,
        save_hz=CAM_SAVE_HZ,
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )

    # ===== 建立 RTDE 连接 & 启动相机 =====
    con = connect_rtde(
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        frequency_hz=RTDE_HZ,
    )

    pipeline = start_realsense_pipeline(
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )

    # ===== 联合采集循环 =====
    next_save_rtde = time.time()
    next_save_cam = time.time()
    frame_idx = 0

    print("联合采集开始，按 'q' 退出。")
    print(f"RTDE CSV: {csv_path}")
    print(f"Camera folder: {cam_run_dir}")

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

                # ------ 2) 相机读取当前画面 ------
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow("RealSense Color", color_image)

                now = time.time()

                # ------ 3) 按 RTDE_HZ 频率保存 TCP pose ------
                if now >= next_save_rtde:
                    writer.writerow([ts] + list(tcp))
                    print(f"[RTDE] {ts:8.3f} s | TCP pose = {tcp}")
                    next_save_rtde += 1.0 / RTDE_HZ

                # ------ 4) 按 CAM_SAVE_HZ 频率保存图片 ------
                if now >= next_save_cam:
                    frame_idx += 1
                    img_name = f"frame_{frame_idx:05d}.png"
                    img_path = os.path.join(cam_run_dir, img_name)
                    cv2.imwrite(img_path, color_image)
                    print(f"[CAM ] [{frame_idx:05d}] 已保存: {img_path}")
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

            pipeline.stop()
            cv2.destroyAllWindows()

            print("联合采集结束。")
            print(f"RTDE config:   {cfg_rtde_path}")
            print(f"Camera config: {cfg_cam_path}")


if __name__ == "__main__":
    main()
