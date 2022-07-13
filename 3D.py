#!/usr/bin/env python3

import numpy as np, rospy, open3d, struct, ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge


ROI_x, ROI_y, ROI_z = [-0, 30], [-5, 10], [-1.5, 3]

EPS = 0.4
MIN_POINTS = 20

Instrinsic = open3d.camera.PinholeCameraIntrinsic(1216, 352, 961.298, 928.864, 625.29, 139.819)
K= np.array([[925.927,0,612.29],[0,933.509,116.819],[0,0,1]])
RT = np.array([[0.0219621,-0.999754,-0.00293465,0.236921],[0.0853894,0.0048004,-0.996336,-0.647117],[0.996106,0.0216311,0.0854738,-2.14507],[0,0,0,1]])
EXTRINSIC = np.array([[0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1]])

def mix_cloud():
    global depth_img, CV_IMG, Instrinsic
    rgb = open3d.geometry.Image(CV_IMG)
    d = open3d.geometry.Image(depth_img)
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, d, depth_scale=1000,
                                                                    depth_trunc=1000, convert_rgb_to_intensity=False) 
    rgbd_pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,Instrinsic, extrinsic=EXTRINSIC)
    # open3d.visualization.draw_geometries([rgbd_pcd])
    return rgbd_pcd

def convertCloudFromOpen3dToRos(open3d_cloud, pcd_msg):
    global fields

    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points)
    n_points = len(points[:, 0])
    
    data = np.zeros(n_points, dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.uint32)
        ])
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]

    colors = np.floor(np.asarray(open3d_cloud.colors)*255)
    colors = colors[:, 0] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
    colors = colors.astype(np.uint32)
    data['rgb'] = colors

    # tmp_c = colors[:,0] * 255 * 2**16 + colors[:,1] * 255 * 2**8 + colors[:,2] * 255
    rospc = ros_numpy.msgify(PointCloud2, data)
    rospc.header = pcd_msg.header
    rospc.fields = fields
    
    # pc_p = np.asarray(open3d_cloud.points)
    # pc_c = np.asarray(open3d_cloud.colors)
    # tmp_c = np.c_[np.zeros(pc_c.shape[1])]
    # tmp_c = np.floor(pc_c[:,0] * 255) * 2**16 + np.floor(pc_c[:,1] * 255) * 2**8 + np.floor(pc_c[:,2] * 255) # 16bit shift, 8bit shift, 0bit shift

    # pc_pc = np.c_[pc_p, tmp_c]
    # create ros_cloud
    return rospc


def convertCloudFromRosToOpen3d(ros_cloud):
    cloud_data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(ros_cloud)
    xyz_1 = np.delete(cloud_data, np.where(cloud_data[:,2]<ROI_z[0]), axis=0)
    xyz_2 = np.delete(xyz_1, np.where(xyz_1[:,2]<ROI_z[0]), axis=0)
    xyz_3 = np.delete(xyz_2, np.where(xyz_2[:,0]<ROI_x[0]), axis=0)
    xyz_4 = np.delete(xyz_3, np.where(xyz_3[:,0]>ROI_x[1]), axis=0)
    xyz_5 = np.delete(xyz_4, np.where(xyz_4[:,1]<ROI_y[0]), axis=0)
    xyz_6 = np.delete(xyz_5, np.where(xyz_5[:,1]>ROI_y[1]), axis=0)
    
    open3d_cloud = open3d.geometry.PointCloud()
    open3d_cloud.points = open3d.utility.Vector3dVector(xyz_6)
    return open3d_cloud

def callback_pcd(pcd_msg):
    global a, pub1, pub2, depth_img, CV_IMG
    o3d_cloud = convertCloudFromRosToOpen3d(pcd_msg)
    down_cloud = open3d.geometry.PointCloud.voxel_down_sample(o3d_cloud, 0.1)
    labels = np.array(down_cloud.cluster_dbscan(eps=EPS, min_points=MIN_POINTS, print_progress=False))

    clusters = []
    num_cls = labels.max() + 1
    for _ in range(num_cls):
        temp = []
        clusters.append(temp)

    xyz_array = np.asarray(down_cloud.points)
    for i, xyz in enumerate(xyz_array):
        if labels[i] == -1:
            continue
        clusters[labels[i]].append([xyz[0], xyz[1], xyz[2]])

    points = []
    for i, cluster in enumerate(clusters):
        if np.shape(cluster)[0] < 70: continue
        for x, y, z in cluster:
            r, g, b = np.uint8(100*i), np.uint8(80*i), np.uint8(30*i)
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            point = [x, y, z, rgb]
            points.append(point)

    pc2 = point_cloud2.create_cloud(pcd_msg.header, fields, points)
    pub1.publish(pc2)

    rgbd_pcd = mix_cloud()
    rospc = convertCloudFromOpen3dToRos(rgbd_pcd, pcd_msg)
    # rospc = orh.o3dpc_to_rospc(rgbd_pcd, frame_id=pcd_msg.header.frame_id, stamp=pcd_msg.header.stamp) 

    pub2.publish(rospc)


def callback_img(msg):
    global CV_IMG
    # print("I get IMG")
    CV_IMG = bridge.imgmsg_to_cv2(msg, "bgr8")

def callback_depth(depth_msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(depth_msg, "mono16")

a = 255
fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgb', 12, PointField.UINT32, 1)]

BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
bridge = CvBridge()
rospy.init_node('show_3D', anonymous=True)
rospy.Subscriber("/rslidar_points", PointCloud2, callback_pcd)
pub1 = rospy.Publisher("/new_points", PointCloud2, queue_size=1)
pub2 = rospy.Publisher("/rgbd_points", PointCloud2, queue_size=1)
rospy.Subscriber("/depth_img", Image, callback_depth)
rospy.Subscriber("/pylon_camera_node/image_raw", Image, callback_img)


rospy.spin()
