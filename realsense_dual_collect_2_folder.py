# 双相机录制脚本（Head + Wrist）
# - 实时显示两路彩色画面
# - 自动按 SAVE_HZ 频率保存图片到一个以时间戳命名的文件夹
# - 按 q 退出

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json
from datetime import datetime

# =========================
# 配置区：按你的实际安装修改
# =========================
HEAD_SERIAL  = "243522072333"   # 头部相机
WRIST_SERIAL = "233522079334"   # 手腕相机

# 数据保存的大目录
BASE_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"

# 自动保存频率（Hz）
SAVE_HZ = 10.0

# 彩色相机参数（方便在文件前部直接改）
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30


def list_serials():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for d in devices:
        serials.append(d.get_info(rs.camera_info.serial_number))
    return serials


def start_pipeline(serial: str):
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_device(serial)
    config_rs.enable_stream(
        rs.stream.color,
        COLOR_WIDTH,
        COLOR_HEIGHT,
        rs.format.bgr8,
        COLOR_FPS,
    )
    pipeline.start(config_rs)
    return pipeline


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # 检查设备是否都在线（避免“跑起来才发现少一台”）
    serials = list_serials()
    print("检测到相机序列号：", serials)
    if HEAD_SERIAL not in serials:
        print(f"未检测到 HEAD_SERIAL={HEAD_SERIAL}，请检查序列号或USB连接。")
        return
    if WRIST_SERIAL not in serials:
        print(f"未检测到 WRIST_SERIAL={WRIST_SERIAL}，请检查序列号或USB连接。")
        return

    # 本次录制的时间戳，用于子文件夹命名
    run_time = datetime.now()
    ts_str = run_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, f"cam_dual_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)

    # 两个子目录分别存 head / wrist
    head_dir = os.path.join(run_dir, "head")
    wrist_dir = os.path.join(run_dir, "wrist")
    os.makedirs(head_dir, exist_ok=True)
    os.makedirs(wrist_dir, exist_ok=True)

    # 写 config.json
    cfg_path = os.path.join(run_dir, "config.json")
    config = {
        "run_start_time": run_time.isoformat(),
        "folder_name": f"cam_dual_{ts_str}",
        "save_hz": SAVE_HZ,
        "color_resolution": [COLOR_WIDTH, COLOR_HEIGHT],
        "color_fps": COLOR_FPS,
        "head_serial": HEAD_SERIAL,
        "wrist_serial": WRIST_SERIAL,
        "notes": "Dual RealSense color streams; images saved at SAVE_HZ into head/ and wrist/.",
    }
    with open(cfg_path, "w") as f_cfg:
        json.dump(config, f_cfg, indent=2)
    print(f"配置已保存到: {cfg_path}")

    # 启动两台相机
    pipe_head = start_pipeline(HEAD_SERIAL)
    pipe_wrist = start_pipeline(WRIST_SERIAL)

    print("双相机已启动。")
    print(f"当前录制目录: {run_dir}")
    print("按 'q' 退出。")

    next_save_time = time.time()
    frame_idx = 0

    try:
        while True:
            # 读取两路 frames
            frames_head = pipe_head.wait_for_frames()
            frames_wrist = pipe_wrist.wait_for_frames()

            color_head = frames_head.get_color_frame()
            color_wrist = frames_wrist.get_color_frame()
            if (not color_head) or (not color_wrist):
                continue

            img_head = np.asanyarray(color_head.get_data())
            img_wrist = np.asanyarray(color_wrist.get_data())

            # 显示
            cv2.imshow("RealSense HEAD Color", img_head)
            cv2.imshow("RealSense WRIST Color", img_wrist)

            # 自动保存
            now = time.time()
            if now >= next_save_time:
                frame_idx += 1

                head_name = f"frame_{frame_idx:05d}.png"
                wrist_name = f"frame_{frame_idx:05d}.png"

                head_path = os.path.join(head_dir, head_name)
                wrist_path = os.path.join(wrist_dir, wrist_name)

                cv2.imwrite(head_path, img_head)
                cv2.imwrite(wrist_path, img_wrist)

                print(f"[{frame_idx:05d}] 已保存: {head_path} | {wrist_path}")
                next_save_time += 1.0 / SAVE_HZ

            # 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipe_head.stop()
        pipe_wrist.stop()
        cv2.destroyAllWindows()
        print("已退出。")


if __name__ == "__main__":
    main()
