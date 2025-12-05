#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 2: 从一个 episode*.hdf5 中
  - 读取一帧 qpos (14 维) + 图像
  - 做归一化
  - 喂给 ACT 前向一把

依然不连真机，只验证「HDF5 → ACT」这条 pipeline。
"""

import os
import pickle

import cv2
import h5py
import numpy as np
import torch

# 复用 Stage1 里的加载函数和常量
from real_eval_stage1_load_act import (
    load_act_from_ckpt,
    TASK_NAME,
    TASK_CONFIG,
    EXPERT_DATA_NUM,
)

# ========= 这里改成你真实存在的 episode 路径 =========
EPISODE_PATH = "/home/zhangw/UR5e_DataCollection/RoboTwin_like_data/run_20251201_193912/torch_cube/simple/data/episode0.hdf5"
# ====================================================

print("[DEBUG] real_eval_stage2_hdf5_forward.py imported")


def load_stats():
    """
    从 ckpt 目录读取 dataset_stats.pkl，
    返回 stats 和 qpos 归一化函数 pre_process。
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

    return stats, pre_process


def load_one_step_from_hdf5(h5_path, t_index=0):
    """
    从 episode.hdf5 里取一个时间步：
      - qpos: 14 维 (left_arm, left_gripper, right_arm, right_gripper)
      - color_image: H x W x 3 (BGR)
    """
    with h5py.File(h5_path, "r") as f:
        # 1) joint_action -> qpos
        ja = f["joint_action"]
        left_arm = ja["left_arm"][t_index].astype(np.float32)       # (6,)
        left_gripper = np.array([ja["left_gripper"][t_index]], np.float32)   # (1,)
        right_arm = ja["right_arm"][t_index].astype(np.float32)     # (6,)
        right_gripper = np.array([ja["right_gripper"][t_index]], np.float32) # (1,)

        qpos = np.concatenate(
            [left_arm, left_gripper, right_arm, right_gripper], axis=0
        )  # (14,)

        # 2) observation -> 一帧图像
        obs = f["observation"]
        cam_key = "head_camera"
        if cam_key not in obs:
            cam_key = list(obs.keys())[0]
        cam_grp = obs[cam_key]
        buf = cam_grp["rgb"][t_index]  # vlen uint8，一帧 jpg

        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # H x W x 3, BGR

    return qpos, img


def build_image_tensor(color_image, camera_names, device):
    """
    把一张 HxWx3 的图像，复制成 len(camera_names) 路，
    然后转成 (1, N_cam, 3, H, W) 的 float32 tensor。
    """
    H, W = 480, 640
    img = cv2.resize(color_image, (W, H))
    img = img.astype(np.float32) / 255.0  # [0,1]

    imgs = []
    for _ in camera_names:
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)               # (N_cam, H, W, 3)
    imgs = np.transpose(imgs, (0, 3, 1, 2))     # (N_cam, 3, H, W)

    image_tensor = torch.from_numpy(imgs).to(device).unsqueeze(0)   # (1, N_cam, 3, H, W)
    return image_tensor


def main():
    # 1) 先把 ACT + stats 都加载好
    policy, camera_names, device = load_act_from_ckpt()
    stats, pre_process = load_stats()

    print(f"[INFO] Using episode: {EPISODE_PATH}")

    # 2) 从 episode.hdf5 里读一帧
    qpos_np, img = load_one_step_from_hdf5(EPISODE_PATH, t_index=0)
    print(f"[INFO] qpos_np shape = {qpos_np.shape}, first 5 = {qpos_np[:5]}")
    print(f"[INFO] image shape = {img.shape}")

    # 3) 归一化 + 转 tensor
    qpos_norm = pre_process(qpos_np)
    qpos_tensor = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)  # (1, 14)
    image_tensor = build_image_tensor(img, camera_names, device)               # (1, N_cam, 3, H, W)

    # 4) 喂给 ACT 前向一把
    with torch.no_grad():
        a_hat = policy(qpos_tensor, image_tensor)

    print(f"[RESULT] a_hat type = {type(a_hat)}")
    if isinstance(a_hat, torch.Tensor):
        print(f"[RESULT] a_hat.shape = {a_hat.shape}")
    else:
        try:
            print(f"[RESULT] a_hat[0].shape = {a_hat[0].shape}")
        except Exception:
            pass

    print("[DONE] Stage 2 hdf5 forward finished.")


if __name__ == "__main__":
    main()
