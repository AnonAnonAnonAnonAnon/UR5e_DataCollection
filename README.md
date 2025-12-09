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

10hz

### (7) Convert to HDF5 format

Include RoboTwin 2.0 as a subfolder of the project, without git: 

UR5e_DataCollection/RoboTwin

In RoboTwin 2.0, each policy has a script that processes data in HDF5 format into the format required for training that policy. Such as: 

UR5e_DataCollection/RoboTwin/policy/ACT/process_data.py

Determine the initial pose for data collection and reasoning, go home script:

```bash
go_home.py
```

Convert all existing data in folder to HDF5:

```bash
python convert_2_hdf5_output_log.py
```

The converted HDF5, for example:

```bash
UR5e_DataCollection/RoboTwin_like_data/run_20251130_194629/torch_cube/simple/data/episode0.hdf5
```

View the contents of an HDF5 file: 

```bash
python preview_hdf5.py /home/zhangw/UR5e_DataCollection/RoboTwin_like_data/run_20251130_194629/torch_cube/simple/data/episode0.hdf5
```

### (8) Convert to ACT format

Modify the data processing script for each strategy, using ACT as an example

```bash
cd /home/zhangw/UR5e_DataCollection/RoboTwin/policy/ACT
python process_data_real.py \
  /home/zhangw/UR5e_DataCollection/RoboTwin_like_data/run_20251201_193912 \
  torch_cube \
  simple \
  3
```

### (9) Train

Install the ACT-related environment according to the RoboTwin documentation:

```bash
cd /home/zhangw/UR5e_DataCollection/RoboTwin/policy/ACT
pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 \
           opencv-python matplotlib einops packaging h5py ipython
cd detr
pip install -e .
cd ..
```

Start Training:

```bash
cd /home/zhangw/UR5e_DataCollection/RoboTwin/policy/ACT
bash train.sh torch_cube simple 3 0 0
#            ^task_name  ^task_config  ^expert_data_num  ^seed  ^gpu_id
```

### (10) Inference

Some preliminary phased test scripts: 

```bash
python /home/zhangw/UR5e_DataCollection/RoboTwin/policy/ACT/real_eval_stage1_load_act.py
python /home/zhangw/UR5e_DataCollection/RoboTwin/policy/ACT/real_eval_stage2_hdf5_forward.py
python /home/zhangw/UR5e_DataCollection/RoboTwin/policy/ACT/real_eval_stage3_online_no_ctrl.py
```

First, return the robotic arm to its initial position:

```bash
python /home/zhangw/UR5e_DataCollection/go_home.py
```

Performing inference on a real UR5e: 

```bash
python /home/zhangw/UR5e_DataCollection/RoboTwin/policy/ACT/real_eval.py
```

### TODO

控制：rtde读写，平滑，推理时自由驱动

数据：数量，夹爪，头部相机，随机化，vr

训练：核桃，240(登录问题)

更多：模型(rt2)，lerobot框架



