#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把实机数据 (rtde_tcp_*.csv + cam_*/frame_*.png 或 cam_dual_*/head|wrist/frame_*.png)
转换成 RoboTwin 风格的 episode*.hdf5
"""

import os
import numpy as np
import cv2
import h5py
from datetime import datetime

# =========================
# CONFIG
# =========================
ACTION_DIR = "/home/zhangw/UR5e_DataCollection/action_data"
CAMERA_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"

OUT_ROOT = "/home/zhangw/UR5e_DataCollection/RoboTwin_like_data"
TASK_NAME = "torch_cube"
TASK_CONFIG = "simple"

# 你的双相机采集目录前缀：cam_dual_<ts>；单相机：cam_<ts>
CAM_DUAL_PREFIX = "cam_dual_"
CAM_SINGLE_PREFIX = "cam_"

# 双相机子目录名
HEAD_SUBDIR = "head"
WRIST_SUBDIR = "wrist"

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = os.path.join(OUT_ROOT, f"run_{RUN_TAG}")
DEBUG_DIR = os.path.join(RUN_ROOT, "_debug")

# RoboTwin observation 里相机组名映射（尽量不改下游）
# - head_camera 用 head 图
# - right_camera 用 wrist 图
# - left_camera 先复制 wrist 图占位
CAMERA_GROUP_MAP = {
    "head_camera": "head",
    "right_camera": "wrist",
    "left_camera": "wrist",
}


def find_episode_pairs():
    """
    找到 (csv_path, cam_mode, cam_path) 三元组：
      cam_mode in {"single", "dual"}
      cam_path:
        - single: camera_data/cam_<ts>
        - dual:   camera_data/cam_dual_<ts>
    """
    pairs = []
    for name in os.listdir(ACTION_DIR):
        if not (name.startswith("rtde_tcp_") and name.endswith(".csv")):
            continue
        ts = name[len("rtde_tcp_"):-4]  # 20251128_200251

        csv_path = os.path.join(ACTION_DIR, name)

        cam_dual = os.path.join(CAMERA_DIR, f"{CAM_DUAL_PREFIX}{ts}")
        cam_single = os.path.join(CAMERA_DIR, f"{CAM_SINGLE_PREFIX}{ts}")

        if os.path.isdir(cam_dual):
            pairs.append((csv_path, "dual", cam_dual))
        elif os.path.isdir(cam_single):
            pairs.append((csv_path, "single", cam_single))

    pairs.sort()
    return pairs


def load_csv_tcp(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    # data: [time, x, y, z, rx, ry, rz]
    tcp = data[:, 1:].astype(np.float32)  # (T, 6)
    return tcp


def load_images_from_dir(img_dir):
    files = [f for f in os.listdir(img_dir) if f.startswith("frame_") and f.endswith(".png")]
    files.sort()
    imgs = []
    for f in files:
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            continue
        imgs.append(img)
    return imgs


def save_episode_hdf5(out_path, tcp, imgs_head, imgs_wrist, episode_idx):
    # 1) 对齐长度：TCP、head、wrist 三者取最小
    T_raw_tcp = len(tcp)
    T_raw_head = len(imgs_head)
    T_raw_wrist = len(imgs_wrist)
    T = min(T_raw_tcp, T_raw_head, T_raw_wrist)

    tcp = tcp[:T]
    imgs_head = imgs_head[:T]
    imgs_wrist = imgs_wrist[:T]

    print(f"  -> T_raw_tcp   = {T_raw_tcp}")
    print(f"  -> T_raw_head  = {T_raw_head}")
    print(f"  -> T_raw_wrist = {T_raw_wrist}")
    print(f"  -> T_used      = {T}")

    if T == 0:
        print("  -> 跳过：对齐后 T=0")
        return

    # 2) 简单起见：把 TCP 当作 right_arm & left_arm，gripper 全 0
    right_arm = tcp
    left_arm = tcp
    right_gripper = np.zeros((T,), dtype=np.float32)
    left_gripper = np.zeros((T,), dtype=np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 2.5) debug
    epi_debug_dir = os.path.join(DEBUG_DIR, f"episode{episode_idx}")
    os.makedirs(epi_debug_dir, exist_ok=True)
    np.save(os.path.join(epi_debug_dir, "tcp_used.npy"), tcp)
    cv2.imwrite(os.path.join(epi_debug_dir, "head_img0_used.png"), imgs_head[0])
    cv2.imwrite(os.path.join(epi_debug_dir, "wrist_img0_used.png"), imgs_wrist[0])
    with open(os.path.join(epi_debug_dir, "info.txt"), "w") as f_info:
        f_info.write(f"T_raw_tcp   = {T_raw_tcp}\n")
        f_info.write(f"T_raw_head  = {T_raw_head}\n")
        f_info.write(f"T_raw_wrist = {T_raw_wrist}\n")
        f_info.write(f"T_used      = {T}\n")

    # 3) 写入 hdf5
    with h5py.File(out_path, "w") as f:
        # joint_action
        ja = f.create_group("joint_action")
        ja.create_dataset("left_arm", data=left_arm)
        ja.create_dataset("left_gripper", data=left_gripper)
        ja.create_dataset("right_arm", data=right_arm)
        ja.create_dataset("right_gripper", data=right_gripper)

        # observation / cameras (vlen uint8, jpg bitstream)
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        obs = f.create_group("observation")

        # 先建相机组与 dataset
        cam_dsets = {}
        for cam_name in CAMERA_GROUP_MAP.keys():
            g = obs.create_group(cam_name)
            dset = g.create_dataset("rgb", (T,), dtype=vlen_uint8)
            cam_dsets[cam_name] = dset

        # 写每一帧
        for t in range(T):
            # 选择 head/wrist 源图
            head_img = imgs_head[t]
            wrist_img = imgs_wrist[t]

            # 逐个相机组写入
            for cam_name, src in CAMERA_GROUP_MAP.items():
                img = head_img if src == "head" else wrist_img
                ok, buf = cv2.imencode(".jpg", img)
                if not ok:
                    # 极少数情况下编码失败，直接跳过这帧（保持最小逻辑）
                    continue
                arr = np.frombuffer(buf.tobytes(), dtype=np.uint8)
                cam_dsets[cam_name][t] = arr


def main():
    pairs = find_episode_pairs()
    print(f"找到 {len(pairs)} 条 episode 配对。")

    out_data_dir = os.path.join(RUN_ROOT, TASK_NAME, TASK_CONFIG, "data")

    for idx, (csv_path, cam_mode, cam_path) in enumerate(pairs):
        print(f"[{idx}] csv = {csv_path}")
        print(f"    cam_mode = {cam_mode}")
        print(f"    cam_path = {cam_path}")

        tcp = load_csv_tcp(csv_path)

        if len(tcp) == 0:
            print("  -> 跳过：tcp 为空\n")
            continue

        if cam_mode == "single":
            # 单相机：沿用旧逻辑，把同一套图当作 head + wrist（占位）
            imgs = load_images_from_dir(cam_path)
            imgs_head = imgs
            imgs_wrist = imgs
            print(f"  -> loaded tcp shape = {tcp.shape}, imgs(single) = {len(imgs)} frames")

        else:
            # 双相机：分别从 head/ wrist/ 读取
            head_dir = os.path.join(cam_path, HEAD_SUBDIR)
            wrist_dir = os.path.join(cam_path, WRIST_SUBDIR)
            imgs_head = load_images_from_dir(head_dir) if os.path.isdir(head_dir) else []
            imgs_wrist = load_images_from_dir(wrist_dir) if os.path.isdir(wrist_dir) else []
            print(f"  -> loaded tcp shape = {tcp.shape}, head = {len(imgs_head)} frames, wrist = {len(imgs_wrist)} frames")

        if len(imgs_head) == 0 or len(imgs_wrist) == 0:
            print("  -> 跳过：head 或 wrist 图片为空\n")
            continue

        out_path = os.path.join(out_data_dir, f"episode{idx}.hdf5")
        save_episode_hdf5(out_path, tcp, imgs_head, imgs_wrist, episode_idx=idx)
        print(f"  -> episode{idx} 保存为 {out_path}\n")

    print("全部转换完成。")


if __name__ == "__main__":
    main()
