#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把当前实机数据 (rtde_tcp_*.csv + cam_*/frame_*.png)
转换成 RoboTwin 风格的 episode*.hdf5
"""

# 转换文件夹中所有的数据

import os
import numpy as np
import cv2
import h5py

ACTION_DIR = "/home/zhangw/UR5e_DataCollection/action_data"
CAMERA_DIR = "/home/zhangw/UR5e_DataCollection/camera_data"

# 输出目录：直接对齐 RoboTwin 的 data 结构
OUT_ROOT = "/home/zhangw/UR5e_DataCollection/RoboTwin_like_data"
TASK_NAME = "ur5e_real_tcp"
TASK_CONFIG = "demo_real"


def find_episode_pairs():
    pairs = []
    for name in os.listdir(ACTION_DIR):
        if not (name.startswith("rtde_tcp_") and name.endswith(".csv")):
            continue
        ts = name[len("rtde_tcp_"):-4]        # 20251128_200251
        cam_dir = os.path.join(CAMERA_DIR, f"cam_{ts}")
        if os.path.isdir(cam_dir):
            csv_path = os.path.join(ACTION_DIR, name)
            pairs.append((csv_path, cam_dir))
    pairs.sort()
    return pairs


def load_csv_tcp(csv_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    # data: [time, x, y, z, rx, ry, rz]
    tcp = data[:, 1:].astype(np.float32)   # (T, 6)
    return tcp


def load_images(cam_dir):
    files = [f for f in os.listdir(cam_dir) if f.startswith("frame_") and f.endswith(".png")]
    files.sort()
    imgs = []
    for f in files:
        img = cv2.imread(os.path.join(cam_dir, f))
        if img is None:
            continue
        imgs.append(img)
    return imgs


def save_episode_hdf5(out_path, tcp, imgs):
    T = min(len(tcp), len(imgs))
    tcp = tcp[:T]
    imgs = imgs[:T]

    # 简单起见：把 TCP 当作 “right_arm & left_arm”，gripper 全 0
    right_arm = tcp
    left_arm = tcp
    right_gripper = np.zeros((T, 1), dtype=np.float32)
    left_gripper = np.zeros((T, 1), dtype=np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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

        cams = {}
        for cam_name in ["head_camera", "right_camera", "left_camera"]:
            g = obs.create_group(cam_name)
            dset = g.create_dataset("rgb", (T,), dtype=vlen_uint8)
            cams[cam_name] = dset

        for t in range(T):
            img = imgs[t]
            ok, buf = cv2.imencode(".jpg", img)
            arr = np.frombuffer(buf.tobytes(), dtype=np.uint8)
            for dset in cams.values():
                dset[t] = arr


def main():
    pairs = find_episode_pairs()
    print(f"找到 {len(pairs)} 条 episode 配对。")

    out_data_dir = os.path.join(
        OUT_ROOT,
        TASK_NAME,
        TASK_CONFIG,
        "data",
    )

    for idx, (csv_path, cam_dir) in enumerate(pairs):
        print(f"[{idx}] csv = {csv_path}")
        print(f"    cam = {cam_dir}")

        tcp = load_csv_tcp(csv_path)
        imgs = load_images(cam_dir)

        if len(tcp) == 0 or len(imgs) == 0:
            print("  -> 跳过：tcp 或图片为空")
            continue

        out_path = os.path.join(out_data_dir, f"episode{idx}.hdf5")
        save_episode_hdf5(out_path, tcp, imgs)
        print(f"  -> 保存为 {out_path}")

    print("全部转换完成。")


if __name__ == "__main__":
    main()
