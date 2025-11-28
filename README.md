Real World UR5e Data Collection   

PC: right UR, zw account

The system has been reinstalled to Ubuntu 25.04, and all packages need to be set up again.

Conda environment based on RoboTwin 2.0 environment

### (1) UR5e Basic Control 

The simplest UR5e control script, only the robotic arm, without the gripper

```bash
pip install keyboard
pip install pyserial
```

Check the network connection with the robotic arm: ping 192.168.0.3

Switch Polyscope to remote control

```bash
python ur5e_test_mini_clean.py
```

### （2）Camera-related component installation (realsense-viewer)

```bash
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
```

Ubuntu 25.04 is a bit troublesome, but running a few more commands will do.

### (3) Camera feed acquisition

```bash
pip install pyrealsense2
```

```bash
realsense_test.py
```

### (4) RTDE

RTDE Project: 
https://github.com/UniversalRobots/RTDE_Python_Client_Library

Download zip: 
https://github.com/UniversalRobots/RTDE_Python_Client_Library/releases

```bash
pip install wheel
pip install rtde-2.7.12-release.zip
```

```bash
pip install numpy
pip install matplotlib
```

Servoj_RTDE_UR5 Project: Provides smoother control

Download Folder: 
https://github.com/danielstankw/Servoj_RTDE_UR5

Output the current TCP pose every second: 
```bash
python rtde_init_test.py
```

### TODO

收集，转hdf5

训练; 核桃gpu

ai插件更新

规划实现
https://mcneznd3g628.feishu.cn/wiki/ITixw3WjMiX4fakEbEPchoMtnXc?from=from_copylink

考古sy实现api



