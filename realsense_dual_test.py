# 双相机调试：显示两路彩色画面
# 按 s 同时截图保存；按 q 退出

import pyrealsense2 as rs
import numpy as np
import cv2
import time


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
    config.enable_device(serial)  # 关键：绑定到指定序列号相机
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def main():
    serials = list_serials()
    if len(serials) < 2:
        print(f"只检测到 {len(serials)} 台 RealSense，相机数不足 2。")
        print("可用序列号：", serials)
        return

    # 你也可以把这里改成：serials = ["xxxx", "yyyy"] 来固定左右相机
    serial1, serial2 = serials[0], serials[1]
    print("检测到相机序列号：", serials)
    print("使用前两台：", serial1, serial2)

    pipe1 = start_pipeline(serial1)
    pipe2 = start_pipeline(serial2)

    print("双相机已启动：按 's' 截图（两张都存），按 'q' 退出。")

    try:
        while True:
            frames1 = pipe1.wait_for_frames()
            frames2 = pipe2.wait_for_frames()

            c1 = frames1.get_color_frame()
            c2 = frames2.get_color_frame()
            if not c1 or not c2:
                continue

            img1 = np.asanyarray(c1.get_data())
            img2 = np.asanyarray(c2.get_data())

            cv2.imshow(f"RS Color [{serial1}]", img1)
            cv2.imshow(f"RS Color [{serial2}]", img2)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                f1 = f"realsense_{serial1}_{ts}.png"
                f2 = f"realsense_{serial2}_{ts}.png"
                cv2.imwrite(f1, img1)
                cv2.imwrite(f2, img2)
                print("已保存截图：", f1, f2)

            if key == ord('q'):
                break

    finally:
        pipe1.stop()
        pipe2.stop()
        cv2.destroyAllWindows()
        print("已退出。")


if __name__ == "__main__":
    main()
