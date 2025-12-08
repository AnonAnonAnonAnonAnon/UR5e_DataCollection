#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 3: RTDE + RealSense 在线前向，但不控制机械臂
- 每 1/CTRL_HZ 秒：
  1) RTDE 读 actual_TCP_pose
  2) 构造 qpos = [tcp, 0, tcp, 0] (14 维)
  3) RealSense 读一帧彩色图
  4) 归一化 + 打包成 tensor
  5) 喂给 ACT，打印一行 action
  6) 按 'q' 退出

先验证「真机传感 → ACT」这条链路是否稳定。
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import pickle

import numpy as np
import cv2
import torch
import sys

# 把 UR5e_DataCollection 根目录加入 sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # .../RoboTwin/policy/ACT
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
# 现在 PROJECT_ROOT = .../UR5e_DataCollection
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 复用 Stage1 里的加载函数和常量
from real_eval_stage1_load_act import (
    load_act_from_ckpt,
    TASK_NAME,
    TASK_CONFIG,
    EXPERT_DATA_NUM,
)

# RTDE / RealSense 工具（在 UR5e_DataCollection 根目录）
from rtde_collect_2_csv_func import connect_rtde
from realsense_collect_2_folder_func import start_realsense_pipeline


print("[DEBUG] real_eval_stage3_online_no_ctrl.py imported")

# ========= 根据你的实际情况改一下 =========
ROBOT_HOST = "192.168.0.3"
ROBOT_PORT = 30004
CTRL_HZ = 10.0          # 控制频率，和采集时尽量保持一致

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30

SHOW_WINDOW = False  # 暂时关闭 OpenCV 窗口，避免卡在 imshow
# =========================================


def load_stats():
    """
    从 ckpt 目录读取 dataset_stats.pkl，
    返回 stats, pre_process, post_process
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(
        base_dir,
        "act_ckpt",
        f"act-{TASK_NAME}",
        f"{TASK_CONFIG}-{EXPERT_DATA_NUM}",
    )
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    print(f"[INFO] Loading stats from: {stats_path}")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    def pre_process(qpos_np: np.ndarray) -> np.ndarray:
        return (qpos_np - stats["qpos_mean"]) / stats["qpos_std"]

    def post_process(action_np: np.ndarray) -> np.ndarray:
        return action_np * stats["action_std"] + stats["action_mean"]

    return stats, pre_process, post_process


def build_qpos_from_tcp(tcp: np.ndarray) -> np.ndarray:
    """
    和 convert_2_hdf5_output_log.py 完全一致：
        right_arm = tcp
        left_arm = tcp
        right_gripper = 0
        left_gripper = 0

    tcp: shape (6,)
    返回: shape (14,)
    """
    tcp = tcp.astype(np.float32)
    left_arm = tcp
    right_arm = tcp
    left_gripper = np.array([0.0], dtype=np.float32)
    right_gripper = np.array([0.0], dtype=np.float32)
    qpos = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=0)
    return qpos


def build_image_tensor(color_image, camera_names, device):
    """
    把一张 HxWx3 的图像复制成 len(camera_names) 路，
    转成 (1, N_cam, 3, H, W) 的 float32 tensor。
    """
    H, W = 480, 640
    img = cv2.resize(color_image, (W, H))
    img = img.astype(np.float32) / 255.0

    imgs = []
    for _ in camera_names:
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)               # (N_cam, H, W, 3)
    imgs = np.transpose(imgs, (0, 3, 1, 2))     # (N_cam, 3, H, W)

    image_tensor = torch.from_numpy(imgs).to(device).unsqueeze(0)   # (1, N_cam, 3, H, W)
    return image_tensor



def main():
    # 1) 加载 ACT + stats
    policy, camera_names, device = load_act_from_ckpt()
    stats, pre_process, post_process = load_stats()

    # 2) 建立 RTDE 连接 & 启动 RealSense
    con = connect_rtde(
        robot_host=ROBOT_HOST,
        robot_port=ROBOT_PORT,
        frequency_hz=CTRL_HZ,
    )
    pipeline = start_realsense_pipeline(
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )

    print("[REAL] Start online forward (no control). Press Ctrl+C to stop.")

    next_step_time = time.time()
    step_idx = 0

    try:
        while True:
            # 2.1 读 RTDE
            state = con.receive()
            if state is None:
                print("[REAL] RTDE connection closed.")
                break

            tcp = np.array(state.actual_TCP_pose, dtype=np.float32)   # (6,)
            qpos_np = build_qpos_from_tcp(tcp)                         # (14,)
            qpos_norm = pre_process(qpos_np)
            qpos_tensor = (
                torch.from_numpy(qpos_norm)
                .float()
                .to(device)
                .unsqueeze(0)                                         # (1, 14)
            )

            # 2.2 读相机
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            if SHOW_WINDOW:
                cv2.imshow("RealSense Color (ACT online)", color_image)

            image_tensor = build_image_tensor(color_image, camera_names, device)

            # 2.3 前向 ACT，得到 raw_action
            with torch.no_grad():
                a_hat = policy(qpos_tensor, image_tensor)             # (1, 50, 14)
                if isinstance(a_hat, torch.Tensor):
                    raw_action = a_hat[0, 0].cpu().numpy()           # (14,)
                else:
                    raw_action = a_hat[0][0].cpu().numpy()

            # 2.4 反归一化
            action_real = post_process(raw_action)                    # (14,)

            print(
                f"[STEP {step_idx:04d}] tcp[0:3]={tcp[:3]}, "
                f"action_real[0:5]={action_real[:5]}"
            )
            step_idx += 1

            # 2.5 维持 CTRL_HZ
            next_step_time += 1.0 / CTRL_HZ
            sleep_t = next_step_time - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        try:
            con.send_pause()
        except Exception:
            pass
        con.disconnect()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[REAL] Online forward finished, connections closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[REAL] KeyboardInterrupt, user aborted.")
