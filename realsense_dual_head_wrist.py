# 双相机调试：Head / Wrist
# - 显示两路彩色画面
# - 按 s 同时截图保存（区分 head/wrist）
# - 按 q 退出

import pyrealsense2 as rs
import numpy as np
import cv2
import time


# =========================
# 配置区：按你的实际安装修改
# =========================
HEAD_SERIAL  = "243522072333"   # 头部相机
WRIST_SERIAL = "233522079334"   # 手腕相机

WIDTH  = 640
HEIGHT = 480
FPS    = 30

OUT_DIR = "."  # 截图保存目录，"." 表示当前目录


def list_serials():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for d in devices:
        serials.append(d.get_info(rs.camera_info.serial_number))
    return serials


def start_pipeline(serial, w=640, h=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)  # 绑定到指定序列号相机
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def main():
    serials = list_serials()
    print("检测到相机序列号：", serials)

    if HEAD_SERIAL not in serials:
        print(f"未检测到 HEAD_SERIAL={HEAD_SERIAL}，请检查序列号或USB连接。")
        return
    if WRIST_SERIAL not in serials:
        print(f"未检测到 WRIST_SERIAL={WRIST_SERIAL}，请检查序列号或USB连接。")
        return

    print("Head camera :", HEAD_SERIAL)
    print("Wrist camera:", WRIST_SERIAL)

    pipe_head = start_pipeline(HEAD_SERIAL,  WIDTH, HEIGHT, FPS)
    pipe_wrist = start_pipeline(WRIST_SERIAL, WIDTH, HEIGHT, FPS)

    print("双相机已启动：按 's' 截图（head+wrist），按 'q' 退出。")

    try:
        while True:
            frames_head = pipe_head.wait_for_frames()
            frames_wrist = pipe_wrist.wait_for_frames()

            c_head = frames_head.get_color_frame()
            c_wrist = frames_wrist.get_color_frame()
            if not c_head or not c_wrist:
                continue

            img_head = np.asanyarray(c_head.get_data())
            img_wrist = np.asanyarray(c_wrist.get_data())

            cv2.imshow(f"RealSense HEAD  [{HEAD_SERIAL}]", img_head)
            cv2.imshow(f"RealSense WRIST [{WRIST_SERIAL}]", img_wrist)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                f_head  = f"{OUT_DIR}/realsense_head_{HEAD_SERIAL}_{ts}.png"
                f_wrist = f"{OUT_DIR}/realsense_wrist_{WRIST_SERIAL}_{ts}.png"
                cv2.imwrite(f_head, img_head)
                cv2.imwrite(f_wrist, img_wrist)
                print("已保存截图：", f_head, f_wrist)

            if key == ord('q'):
                break

    finally:
        pipe_head.stop()
        pipe_wrist.stop()
        cv2.destroyAllWindows()
        print("已退出。")


if __name__ == "__main__":
    main()
