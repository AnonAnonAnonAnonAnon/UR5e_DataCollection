# 相机调试 + 录制脚本
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

# 数据保存的大目录
BASE_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"

# 自动保存频率（Hz）
SAVE_HZ = 1.0

# 彩色相机参数（方便在文件前部直接改）
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    # 本次录制的时间戳，用于子文件夹命名
    run_time = datetime.now()
    ts_str = run_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, f"cam_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)

    # 写一个简单的配置文件，记录这次拍摄信息
    cfg_path = os.path.join(run_dir, "config.json")
    config = {
        "run_start_time": run_time.isoformat(),
        "folder_name": f"cam_{ts_str}",
        "save_hz": SAVE_HZ,
        "color_resolution": [COLOR_WIDTH, COLOR_HEIGHT],
        "color_fps": COLOR_FPS,
        "notes": "RealSense color stream; images saved at SAVE_HZ.",
    }
    with open(cfg_path, "w") as f_cfg:
        json.dump(config, f_cfg, indent=2)
    print(f"配置已保存到: {cfg_path}")

    # 1. 创建 RealSense pipeline
    pipeline = rs.pipeline()
    config_rs = rs.config()

    # 2. 启用彩色流（使用上面的参数）
    config_rs.enable_stream(
        rs.stream.color,
        COLOR_WIDTH,
        COLOR_HEIGHT,
        rs.format.bgr8,
        COLOR_FPS,
    )

    # 3. 开始采集
    pipeline.start(config_rs)
    print("RealSense 已启动。")
    print(f"当前录制目录: {run_dir}")
    print("按 'q' 退出。")

    # 用于 SAVE_HZ 保存的计时
    next_save_time = time.time()
    frame_idx = 0

    try:
        while True:
            # 4. 等待一帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 5. 转成 numpy 数组 (BGR)
            color_image = np.asanyarray(color_frame.get_data())

            # 6. 显示
            cv2.imshow("RealSense Color", color_image)

            # 7. 自动按 SAVE_HZ 保存一张图
            now = time.time()
            if now >= next_save_time:
                frame_idx += 1
                img_name = f"frame_{frame_idx:05d}.png"
                img_path = os.path.join(run_dir, img_name)
                cv2.imwrite(img_path, color_image)
                print(f"[{frame_idx:05d}] 已保存: {img_path}")
                next_save_time += 1.0 / SAVE_HZ

            # 8. 键盘监听：按 q 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        # 9. 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("已退出。")


if __name__ == "__main__":
    main()
