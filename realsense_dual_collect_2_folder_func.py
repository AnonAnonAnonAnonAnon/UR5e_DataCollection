# 双相机调试 + 录制脚本（函数化版本）
# - 连接 2 个相机（Head + Wrist，通过序列号区分）
# - 实时显示两路彩色画面
# - 自动按 SAVE_HZ 频率保存图片到一个以时间戳命名的文件夹
#   目录结构：
#     cam_dual_YYYYMMDD_HHMMSS/
#       config.json
#       head/frame_00001.png ...
#       wrist/frame_00001.png ...
# - 按 q 退出

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json
from datetime import datetime


def list_serials():
    """枚举当前可用的 RealSense 设备序列号列表。"""
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for d in devices:
        serials.append(d.get_info(rs.camera_info.serial_number))
    return serials


def prepare_dual_camera_run(base_dir, save_hz, width, height, fps, head_serial, wrist_serial):
    """
    准备一次双相机采集：
    - 确保大目录存在
    - 建子目录 cam_dual_YYYYMMDD_HHMMSS
      - head/
      - wrist/
    - 写 config.json
    - 返回 run_dir, head_dir, wrist_dir, cfg_path
    """
    os.makedirs(base_dir, exist_ok=True)

    run_time = datetime.now()
    ts_str = run_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"cam_dual_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)

    head_dir = os.path.join(run_dir, "head")
    wrist_dir = os.path.join(run_dir, "wrist")
    os.makedirs(head_dir, exist_ok=True)
    os.makedirs(wrist_dir, exist_ok=True)

    cfg_path = os.path.join(run_dir, "config.json")
    config = {
        "run_start_time": run_time.isoformat(),
        "folder_name": f"cam_dual_{ts_str}",
        "save_hz": save_hz,
        "color_resolution": [width, height],
        "color_fps": fps,
        "head_serial": head_serial,
        "wrist_serial": wrist_serial,
        "notes": "Dual RealSense color streams; images saved at SAVE_HZ into head/ and wrist/.",
    }
    with open(cfg_path, "w") as f_cfg:
        json.dump(config, f_cfg, indent=2)

    print(f"配置已保存到: {cfg_path}")
    print(f"当前录制目录: {run_dir}")
    print(f"head_dir: {head_dir}")
    print(f"wrist_dir: {wrist_dir}")

    return run_dir, head_dir, wrist_dir, cfg_path


def start_realsense_pipeline(serial, width, height, fps):
    """
    创建并启动指定序列号的 RealSense 彩色流，返回 pipeline。
    """
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_device(serial)
    config_rs.enable_stream(
        rs.stream.color,
        width,
        height,
        rs.format.bgr8,
        fps,
    )
    pipeline.start(config_rs)
    print(f"RealSense 已启动: {serial}")
    return pipeline


def record_dual_realsense_color(pipe_head, pipe_wrist, head_dir, wrist_dir, save_hz, head_serial, wrist_serial):
    """
    使用给定的两个 pipeline 采集彩色图像，显示并按 save_hz 频率保存到 head_dir / wrist_dir。
    按 'q' 退出。
    """
    next_save_time = time.time()
    frame_idx = 0

    try:
        print("按 'q' 退出。")
        while True:
            frames_head = pipe_head.wait_for_frames()
            frames_wrist = pipe_wrist.wait_for_frames()

            color_head = frames_head.get_color_frame()
            color_wrist = frames_wrist.get_color_frame()
            if (not color_head) or (not color_wrist):
                continue

            img_head = np.asanyarray(color_head.get_data())
            img_wrist = np.asanyarray(color_wrist.get_data())

            cv2.imshow(f"RealSense HEAD  [{head_serial}]", img_head)
            cv2.imshow(f"RealSense WRIST [{wrist_serial}]", img_wrist)

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
                next_save_time += 1.0 / save_hz

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipe_head.stop()
        pipe_wrist.stop()
        cv2.destroyAllWindows()
        print("已退出。")


def main():
    # ========== 参数区 ==========
    BASE_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"
    SAVE_HZ = 10.0

    COLOR_WIDTH = 640
    COLOR_HEIGHT = 480
    COLOR_FPS = 30

    HEAD_SERIAL = "243522072333"
    WRIST_SERIAL = "233522079334"

    # （可选但很实用）启动前检查两台相机是否都在
    serials = list_serials()
    print("检测到相机序列号：", serials)
    if HEAD_SERIAL not in serials:
        print(f"未检测到 HEAD_SERIAL={HEAD_SERIAL}，请检查序列号或USB连接。")
        return
    if WRIST_SERIAL not in serials:
        print(f"未检测到 WRIST_SERIAL={WRIST_SERIAL}，请检查序列号或USB连接。")
        return

    run_dir, head_dir, wrist_dir, cfg_path = prepare_dual_camera_run(
        base_dir=BASE_DIR,
        save_hz=SAVE_HZ,
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
        head_serial=HEAD_SERIAL,
        wrist_serial=WRIST_SERIAL,
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

    record_dual_realsense_color(
        pipe_head=pipe_head,
        pipe_wrist=pipe_wrist,
        head_dir=head_dir,
        wrist_dir=wrist_dir,
        save_hz=SAVE_HZ,
        head_serial=HEAD_SERIAL,
        wrist_serial=WRIST_SERIAL,
    )


if __name__ == "__main__":
    main()
