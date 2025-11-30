# 回到初始位姿

# go_home.py
# 让 UR5e 回到固定初始位姿（无夹爪控制）

# -0.4896101190440506,
# -0.471618141865121,
# 0.42969127076079894,
# 1.1734031785209829,
# 2.848240223282975,
# 0.03257676565011889
    
import socket
import time

def main():
    host = "192.168.0.3"   # 控制箱 IP
    port = 30001           # 与你原来的脚本保持一致

    #尽量的低，但手腕相机视野包括了整张白纸
    home_pose = [-0.4896101190440506,
                 -0.471618141865121,
                 0.42969127076079894,
                 1.1734031785209829,
                 2.848240223282975,
                 0.03257676565011889]

    cmd = (
        "movel(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a=1.2, v=0.2)\n"
        .format(x=home_pose[0], y=home_pose[1], z=home_pose[2],
                rx=home_pose[3], ry=home_pose[4], rz=home_pose[5])
    )

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect((host, port))

    time.sleep(0.5)        # 稍等网络稳定
    s.send(cmd.encode("utf-8"))
    time.sleep(5.0)        # 等待机械臂运动完成（按需要调整）

    s.close()

if __name__ == "__main__":
    main()
    



