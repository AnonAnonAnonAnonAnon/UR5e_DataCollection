# 规范的函数封装

# 相机调试 + 录制脚本（函数化版本）
# - 只连接 1 个相机
# - 实时显示彩色画面
# - 自动按 SAVE_HZ 频率保存图片到一个以时间戳命名的文件夹
# - 按 q 退出

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json
from datetime import datetime

# 文件操作的准备
def prepare_camera_run(base_dir, save_hz, width, height, fps):
    """
    准备一次相机采集：
    - 确保大目录存在
    - 建子目录 cam_YYYYMMDD_HHMMSS
    - 写 config.json
    - 返回 run_dir, cfg_path
    """
    os.makedirs(base_dir, exist_ok=True)

    run_time = datetime.now()
    ts_str = run_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"cam_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)

    cfg_path = os.path.join(run_dir, "config.json")
    config = {
        "run_start_time": run_time.isoformat(),
        "folder_name": f"cam_{ts_str}",
        "save_hz": save_hz,
        "color_resolution": [width, height],
        "color_fps": fps,
        "notes": "RealSense color stream; images saved at SAVE_HZ.",
    }
    with open(cfg_path, "w") as f_cfg:
        json.dump(config, f_cfg, indent=2)

    print(f"配置已保存到: {cfg_path}")
    print(f"当前录制目录: {run_dir}")

    return run_dir, cfg_path

# 相机启动
def start_realsense_pipeline(width, height, fps):
    """
    创建并启动 RealSense 彩色流，返回 pipeline。
    """
    pipeline = rs.pipeline()
    config_rs = rs.config()

    config_rs.enable_stream(
        rs.stream.color,
        width,
        height,
        rs.format.bgr8,
        fps,
    )

    pipeline.start(config_rs)
    print("RealSense 已启动。")

    return pipeline

# 显示和自动保存
def record_realsense_color(pipeline, run_dir, save_hz):
    """
    使用给定的 pipeline 采集彩色图像，显示并按 save_hz 频率保存到 run_dir。
    按 'q' 退出。
    """
    next_save_time = time.time()
    frame_idx = 0

    try:
        print("按 'q' 退出。")
        while True:
            # 1. 等待一帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 2. 转成 numpy 数组 (BGR)
            color_image = np.asanyarray(color_frame.get_data())

            # 3. 显示
            cv2.imshow("RealSense Color", color_image)

            # 4. 自动按 save_hz 保存一张图
            now = time.time()
            if now >= next_save_time:
                frame_idx += 1
                img_name = f"frame_{frame_idx:05d}.png"
                img_path = os.path.join(run_dir, img_name)
                cv2.imwrite(img_path, color_image)
                print(f"[{frame_idx:05d}] 已保存: {img_path}")
                next_save_time += 1.0 / save_hz

            # 5. 键盘监听：按 q 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("已退出。")


def main():
    # 在 main 中统一设置参数（方便以后和 RTDE 联合）
    BASE_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"
    SAVE_HZ = 1.0
    COLOR_WIDTH = 640
    COLOR_HEIGHT = 480
    COLOR_FPS = 30

    run_dir, cfg_path = prepare_camera_run(
        base_dir=BASE_DIR,
        save_hz=SAVE_HZ,
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )

    pipeline = start_realsense_pipeline(
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )

    record_realsense_color(
        pipeline=pipeline,
        run_dir=run_dir,
        save_hz=SAVE_HZ,
    )


if __name__ == "__main__":
    main()
