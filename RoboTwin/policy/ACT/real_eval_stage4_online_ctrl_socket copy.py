#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 4: RTDE + RealSense 在线前向 + socket 速度控制
- 每 1/CTRL_HZ 秒：
  1) RTDE 读 actual_TCP_pose (6 维)
  2) 构造 qpos = [tcp, 0, tcp, 0] (14 维) 并归一化
  3) RealSense 读一帧彩色图，拼成 (1, N_cam, 3, H, W)
  4) ACT 前向，得到 action_real (14 维)，取前 6 维作为目标 TCP
  5) 根据 tcp_des - tcp_cur 计算速度 v_tcp，发送 speedl 控制
  6) Ctrl+C 退出

⚠️ 真实会动机械臂，请确保现场安全。
"""

import os
import time
import pickle
import socket
import sys

import numpy as np
import cv2
import torch

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

print("[DEBUG] real_eval_stage4_online_ctrl_socket.py imported")

# ========= 根据你的实际情况改一下 =========
ROBOT_HOST = "192.168.0.3"
RTDE_PORT = 30004          # 读状态（在 connect_rtde 里用）
SOCKET_PORT = 30001        # socket 控制（你之前测试 speedl 用的端口）

CTRL_HZ = 10.0             # 控制频率（和采集、Stage 3 一致）

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30

SHOW_WINDOW = False        # 如需看实时图像，可改成 True
# =========================================


# ====================== 归一化相关 ======================

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
    与 convert_2_hdf5_output_log.py 一致：
        right_arm = tcp
        left_arm  = tcp
        right_gripper = 0
        left_gripper  = 0

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


# ====================== socket 控制相关 ======================

def connect_socket(robot_host: str, robot_port: int = SOCKET_PORT) -> socket.socket:
    """建立到机器人的 TCP 连接，用于发送 URScript 控制命令。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    print(f"[INFO] Connecting control socket to {robot_host}:{robot_port} ...")
    s.connect((robot_host, robot_port))
    print("[INFO] Control socket connected.")
    return s


def send_speedl(sock: socket.socket, v_tcp: np.ndarray, a: float, dt: float):
    """
    发送 speedl 命令。
    v_tcp: shape (6,) -> [vx, vy, vz, wx, wy, wz]
    a: 加速度
    dt: 持续时间（秒），需 <= 0.2，且每个周期都要刷新一次
    """
    vx, vy, vz, wx, wy, wz = v_tcp.tolist()
    cmd = (
        f"speedl([{vx:.4f}, {vy:.4f}, {vz:.4f}, "
        f"{wx:.4f}, {wy:.4f}, {wz:.4f}], a={a:.2f}, t={dt:.3f})\n"
    )
    try:
        sock.sendall(cmd.encode("utf-8"))
    except Exception as e:
        print(f"[WARN] send_speedl failed: {e}")


def send_stop(sock: socket.socket):
    """发送 stopl 停止线速度控制。"""
    cmd = "stopl(0.5)\n"
    try:
        sock.sendall(cmd.encode("utf-8"))
    except Exception as e:
        print(f"[WARN] send_stop failed: {e}")


# ====================== 主循环 ======================

def main():
    # 1) 加载 ACT + stats
    policy, camera_names, device = load_act_from_ckpt()
    stats, pre_process, post_process = load_stats()

    # 2) RTDE + RealSense + socket
    con = connect_rtde(
        robot_host=ROBOT_HOST,
        robot_port=RTDE_PORT,
        frequency_hz=CTRL_HZ,
    )
    pipeline = start_realsense_pipeline(
        width=COLOR_WIDTH,
        height=COLOR_HEIGHT,
        fps=COLOR_FPS,
    )
    sock = connect_socket(ROBOT_HOST, SOCKET_PORT)

    print("[REAL] Stage 4: online forward + socket control.")
    print("       Press Ctrl+C 终止（脚本里会自动 stopl / 断连）。")

    next_step_time = time.time()
    step_idx = 0

    # 速度限制（安全起见）
    MAX_LIN_VEL = 0.15   # m/s
    MAX_ANG_VEL = 0.5    # rad/s
    ACC = 0.5            # speedl 的 a
    dt_cmd = 1.0 / CTRL_HZ

    try:
        while True:
            # 2.1 RTDE 读取当前 TCP
            state = con.receive()
            if state is None:
                print("[REAL] RTDE connection closed.")
                break

            tcp = np.array(state.actual_TCP_pose, dtype=np.float32)   # (6,)

            # 2.2 构造 qpos、归一化
            qpos_np = build_qpos_from_tcp(tcp)                         # (14,)
            qpos_norm = pre_process(qpos_np)
            qpos_tensor = (
                torch.from_numpy(qpos_norm)
                .float()
                .to(device)
                .unsqueeze(0)                                         # (1, 14)
            )

            # 2.3 RealSense 取一帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            if SHOW_WINDOW:
                cv2.imshow("RealSense Color (ACT online ctrl)", color_image)
                cv2.waitKey(1)

            image_tensor = build_image_tensor(color_image, camera_names, device)

            # 2.4 ACT 前向，取 1 个 action
            with torch.no_grad():
                a_hat = policy(qpos_tensor, image_tensor)             # (1, 50, 14)
                if isinstance(a_hat, torch.Tensor):
                    raw_action = a_hat[0, 0].cpu().numpy()           # (14,)
                else:
                    raw_action = a_hat[0][0].cpu().numpy()

            action_real = post_process(raw_action)                    # (14,)

            # 取前 6 维作为目标 TCP（与训练时构造 qpos 的 tcp 对应）
            tcp_des = action_real[:6].astype(np.float32)

            # 2.5 计算 TCP 速度并限幅
            delta = tcp_des - tcp                                     # (6,)
            v_tcp = delta / max(dt_cmd, 1e-3)

            # 线速度限幅
            v_tcp[:3] = np.clip(v_tcp[:3], -MAX_LIN_VEL, MAX_LIN_VEL)
            # 角速度限幅
            v_tcp[3:] = np.clip(v_tcp[3:], -MAX_ANG_VEL, MAX_ANG_VEL)

            # 2.6 发送速度控制命令
            send_speedl(sock, v_tcp, a=ACC, dt=dt_cmd)

            # 打一点 log
            print(
                f"[STEP {step_idx:04d}] "
                f"tcp_cur[0:3]={tcp[:3]}, "
                f"tcp_des[0:3]={tcp_des[:3]}, "
                f"v_tcp[0:3]={v_tcp[:3]}"
            )
            step_idx += 1

            # 2.7 保持 CTRL_HZ
            next_step_time += dt_cmd
            sleep_t = next_step_time - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        print("[REAL] Cleaning up ...")
        try:
            send_stop(sock)
        except Exception:
            pass

        try:
            sock.close()
        except Exception:
            pass

        try:
            con.send_pause()
        except Exception:
            pass

        try:
            con.disconnect()
        except Exception:
            pass

        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("[REAL] Stage 4 finished, connections closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[REAL] KeyboardInterrupt, user aborted.")
