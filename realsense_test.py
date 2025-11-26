#相机调试
#只连接 1 个相机
#实时显示彩色画面
#按 s 截图保存
#按 q 退出

import pyrealsense2 as rs
import numpy as np
import cv2
import time

def main():
    # 1. 创建 RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. 启用彩色流（可以改分辨率/帧率）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 3. 开始采集
    pipeline.start(config)
    print("RealSense 已启动，按 's' 截图，按 'q' 退出。")

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

            # 7. 键盘监听
            key = cv2.waitKey(1) & 0xFF

            # 按 s 保存截图
            if key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"realsense_frame_{ts}.png"
                cv2.imwrite(filename, color_image)
                print(f"已保存截图: {filename}")

            # 按 q 退出
            if key == ord('q'):
                break

    finally:
        # 8. 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("已退出。")

if __name__ == "__main__":
    main()
