实机UR5e数据收集   

pc：ur 右边， zw 账户

系统重装为25.04，所有轮子重新配


### （1）机械臂基本控制

ur5e_test_mini_clean.py

最简单的UR5e控制脚本，仅手臂，不含夹爪

pip install keyboard

pip install pyserial

检查和机械臂的网络连接: ping 192.168.0.3

远程控制


### （2）相关组件安装相机（realsense-viewer）

sudo apt update

sudo apt install librealsense2-utils librealsense2-dkms librealsense2-dev librealsense2-dbg librealsense2

sudo apt install librealsense2-gl librealsense2-net librealsense2-udev

sudo apt install realsense-viewer

sudo apt install -y apt-transport-https ca-certificates curl

sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
  | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo noble main" \
  | sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt update

sudo apt install -y \
  librealsense2-dkms \
  librealsense2-utils \
  librealsense2-dev \
  librealsense2-gl \
  librealsense2-udev-rules

realsense-viewer

### (3)相机画面获取

pip install pyrealsense2

realsense_test.py


### TODO

收集，转hdf5

训练; 核桃gpu

ai插件更新

规划实现（from draft&feishu）

考古sy实现api



