#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单 HDF5 预览工具：
- 打印 HDF5 结构
- 打印动作信息
- 展示动作 heatmap
- 显示多帧相机图像（默认 right_camera 的前 4 帧）
"""

import argparse
import os
import math

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt


def print_tree(f):
    """打印 HDF5 的层级结构"""
    print("\n=== HDF5 structure ===")

    def show(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[Group ] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"[Dataset] {name:40s} shape={obj.shape}, dtype={obj.dtype}")

    f.visititems(show)


def show_actions(f):
    """打印 joint_action 中的基本信息"""
    if "joint_action" not in f:
        print("\n[WARN] no 'joint_action' group in this file.")
        return

    ja = f["joint_action"]
    print("\n=== joint_action info ===")
    for key in ja.keys():
        ds = ja[key]
        print(f"- {key:12s}: shape={ds.shape}, dtype={ds.dtype}")

    if "right_arm" in ja:
        right_arm = ja["right_arm"][()]
        print("\nright_arm sample:")
        print("  shape:", right_arm.shape)
        if right_arm.shape[0] > 0:
            print("  first:", right_arm[0])
            print("  last :", right_arm[-1])


def plot_action_heatmap(f, joint_name="right_arm", max_steps=200):
    """画一个简单的动作 heatmap"""
    if "joint_action" not in f:
        print("\n[WARN] no 'joint_action' group in this file, skip heatmap.")
        return

    ja = f["joint_action"]
    if joint_name not in ja:
        print(
            f"\n[WARN] joint_action/'{joint_name}' not found, "
            f"available: {list(ja.keys())}"
        )
        return

    data = ja[joint_name][()]  # (T, D)
    if data.ndim != 2:
        print(f"\n[WARN] {joint_name} is not 2D, skip heatmap. shape={data.shape}")
        return

    T, D = data.shape
    if T == 0:
        print(f"\n[WARN] {joint_name} is empty, skip heatmap.")
        return

    used_T = min(T, max_steps)
    if used_T < T:
        print(f"\n[INFO] heatmap only uses first {used_T}/{T} steps.")

    data_show = data[:used_T]

    print(f"\n=== action heatmap ({joint_name}) ===")
    print(f"shape used: {data_show.shape} (T, D)")

    plt.figure()
    im = plt.imshow(data_show, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("DoF (dimension index)")
    plt.ylabel("Timestep")
    plt.title(f"{joint_name} heatmap (first {used_T} steps)")
    plt.tight_layout()
    plt.show()


def show_images(f, cam_name="right_camera", start_idx=0, num_frames=4):
    """解码并显示多帧图像（网格展示）"""
    obs = f.get("observation", None)
    if obs is None:
        print("\n[WARN] no 'observation' group in this file.")
        return

    if cam_name not in obs:
        print(f"\n[WARN] camera '{cam_name}' not found, available cameras:")
        print("      ", list(obs.keys()))
        return

    cam_grp = obs[cam_name]
    if "rgb" not in cam_grp:
        print(f"\n[WARN] 'rgb' dataset not found under observation/{cam_name}/")
        return

    rgb = cam_grp["rgb"]
    n = rgb.shape[0]

    if n == 0:
        print(f"\n[WARN] observation/{cam_name}/rgb is empty.")
        return

    if start_idx < 0 or start_idx >= n:
        print(f"\n[WARN] start_idx {start_idx} out of range [0, {n-1}]")
        return

    end_idx = min(n, start_idx + num_frames)
    indices = list(range(start_idx, end_idx))
    if not indices:
        print("\n[WARN] no frames to show.")
        return

    print(f"\n=== show images ===")
    print(f"camera = {cam_name}, frames = {indices}")

    k = len(indices)
    cols = min(4, k)
    rows = math.ceil(k / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # 统一成一维数组

    for ax_idx, frame_idx in enumerate(indices):
        bits = rgb[frame_idx]
        arr = np.frombuffer(bits, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            axes[ax_idx].set_title(f"t={frame_idx} (decode fail)")
            axes[ax_idx].axis("off")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[ax_idx].imshow(img_rgb)
        axes[ax_idx].set_title(f"t={frame_idx}")
        axes[ax_idx].axis("off")

    # 多余子图隐藏
    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{cam_name} frames [{indices[0]}..{indices[-1]}]", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Preview RoboTwin-style HDF5 file.")
    parser.add_argument("hdf5_path", type=str, help="path to hdf5 file")
    parser.add_argument("--cam", type=str, default="right_camera", help="camera name")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="start frame index for image preview",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=10,
        help="number of frames to show (from start index)",
    )
    parser.add_argument(
        "--joint",
        type=str,
        default="right_arm",
        help="joint_action dataset to plot heatmap for",
    )
    parser.add_argument(
        "--maxT",
        type=int,
        default=200,
        help="max timesteps used in heatmap (for long episodes)",
    )
    args = parser.parse_args()

    path = args.hdf5_path
    if not os.path.isfile(path):
        print(f("[ERROR] file not found: {path}"))
        return

    with h5py.File(path, "r") as f:
        print(f"== Preview: {path} ==")
        print_tree(f)
        show_actions(f)
        plot_action_heatmap(f, joint_name=args.joint, max_steps=args.maxT)
        show_images(
            f,
            cam_name=args.cam,
            start_idx=args.frame,
            num_frames=args.nframes,
        )


if __name__ == "__main__":
    main()
