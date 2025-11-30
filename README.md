Real World UR5e Data Collection   

This records the entire process of building the system from the ground up, including all environments and fine-grained subfunctions.

PC: right UR, zw account

The system has been reinstalled to Ubuntu 25.04, and all packages need to be set up again.

Conda environment based on RoboTwin 2.0 environment

### (1) UR5e Connection and Basic Control 

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

### （2）Camera-related component installation

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
```

Ubuntu 25.04 is a bit troublesome, but running a few more commands will do.

Use Intel's app to view the camera: 

```bash
realsense-viewer
```

### (3) Camera feed acquisition using python

```bash
pip install pyrealsense2
```
Display color image in real time

Press s to take a screenshot

Press q to quit

```bash
realsense_test.py
```

### (4) RTDE install

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

RTDE basic test: output TCP pose per second:

```bash
python rtde_init_test.py
```

### (5) Collect action and camera data separately

Collect action data using RTDE.

Record the TCP pose of the robotic arm and save it:

```bash
python rtde_collect_2_csv.py
```

Collect camera data.

Basic implementation that saves images to a folder at a certain frequency:

```bash
python realsense_collect_2_folder.py
```

### (6) Simultaneous collect action and camera data 

Encapsulated the functions of rtde_collect_2_csv.py and realsense_collect_2_folder.py into functions

realsense_collect_2_folder_func.py

rtde_collect_2_csv_func.py

Call the encapsulated function to simultaneously collect action data and camera footage at a fixed frequency:

```bash
python collect_data_action_camera.py
```

### (7) Convert to HDF5 format

Get the RoboTwin 2.0 repository as a subfolder using a git submodule:

```bash
git submodule add https://github.com/RoboTwin-Platform/RoboTwin.git RoboTwin
git add .gitmodules RoboTwin
git commit -m "Add RoboTwin 2.0 as submodule"
```

After having submodules, how to clone this repository:

``` bash
git clone --recurse-submodules https://github.com/AnonAnonAnonAnonAnon/UR5e_DataCollection.git
```



确定数据收集和推理的初始位姿


### TODO

数据转hdf5

训练; 核桃gpu

ai插件更新

规划实现
https://mcneznd3g628.feishu.cn/wiki/ITixw3WjMiX4fakEbEPchoMtnXc?from=from_copylink

考古sy实现api



