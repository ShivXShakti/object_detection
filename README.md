# object_detection
This package used YOLO8 and realsense d435i camera to detect objects and find there 3D coordinates.

## Usage
```bash
ros2 launch realsense2_camera rs_launch.py camera_namespace:=robot1 camera_name:=D435_1   unite_imu_method:=2 enable_gyro:=True enable_accel:=True   gyro_qos:=SENSOR_DATA accel_qos:=SENSOR_DATA   accel_info_qos:=SENSOR_DATA gyro_info_qos:=SENSOR_DATA publish_tf:=true depth_module.profile:=640x480x30 rgb_camera.profile:=1280x720x30 align_depth.enable:=true clip_distance:=2.0 pointcloud.enable:=true enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true
```
```bash
ros2 run object_detection detection_service
```
```bash
ros2 run object_detection detection_client
```
