# Visualization
2D and 3D visualization

## 2D visualization
```
$ roslaunch yolov5_ros yolov5.launch
$ rosbag play data/assemble_depthformer.bag -l
$ rosrun visualization 2D.py
```

## 3D visualization
```
$ roscore
$ rosbag play data/assemble_depthformer.bag -l
$ rosrun visualization 3D.py
$ rviz
```
