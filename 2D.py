#!/usr/bin/env python3

import numpy as np, rospy, open3d, struct, ros_numpy, cv2
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from detection_msgs.msg importBoundingBoxes
from cv_bridge import CvBridge

ROI_x, ROI_y, ROI_z = [-0, 50], [-5, 8], [-1.5, 3]
EPS = 0.3
MIN_POINTS = 20
DOWN = 0.1
CLUSTER_COM_DEPTH = [[],[]]
K= np.array([[961.298,0,625.29],[0,928.864,139.819],[0,0,1]])
RT = np.array([[0.0213413,-0.999772,-0.00126187,0.101743],[0.086477,0.00310347,-0.996248,-0.611212],[0.996024,0.0211521,0.0865234,-0.751865],[0,0,0,1]])
FLAG_DB = False

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

def callback_yolo(yolo_msg):
    global CLUSTER_COM_DEPTH, CV_IMG, depth_img
    COM, DEPTH = CLUSTER_COM_DEPTH[0], CLUSTER_COM_DEPTH[1]
    store_box = []
    depth_temp = depth_img.copy()

    for i, pts in enumerate(COM):
        np_pts = np.asarray(pts).reshape(3,1)
        np_pts = np.concatenate([np_pts, np.ones((1,1))],axis=0)
        xyz_c = RT@np_pts
        xyz = K@xyz_c[:3,:3]
        box_xmin, box_ymin, box_xmax, box_ymax = 0, 0, 0, 0
        OD_class = "Unknown"
        depth = xyz[2]
        xy = xyz[0:2]/depth
        for BoundingBox in yolo_msg.bounding_boxes:
            obj_class = BoundingBox.Class
            xmin, xmax = BoundingBox.xmin, BoundingBox.xmax
            ymin, ymax = BoundingBox.ymin, BoundingBox.ymax

            if xmin<=xy[0]<=xmax and ymin<=xy[1]<=ymax:
                box_xmin, box_ymin, box_xmax, box_ymax = xmin, ymin, xmax, ymax
                OD_class = obj_class
                break
        if box_xmin ==0 and box_ymin ==0 and box_xmax ==0 and box_ymax ==0:
            cv2.imshow("show", CV_IMG);cv2.waitKey(1)
            continue
        if [box_xmin, box_ymin, box_xmax, box_ymax] in store_box:
            cv2.imshow("show", CV_IMG);cv2.waitKey(1)
            continue
        store_box.append([box_xmin, box_ymin, box_xmax, box_ymax])

        depth_value=round(np.median(depth_temp[box_ymin:box_ymax,box_xmin:box_xmax])/1000, 2)

        cv2.rectangle(CV_IMG, (box_xmin, box_ymin), (box_xmax, box_ymax), (255, 0, 0), 2)


        cv2.putText(CV_IMG, OD_class, (box_xmin, box_ymin-22), 0, 0.4, (255, 0, 0), 1)
        cv2.putText(CV_IMG, str(DEPTH[i]), (box_xmin, box_ymin-10), 0, 0.4, (0, 0, 255), 1)
        cv2.putText(CV_IMG, str(depth_value), (box_xmin+40, box_ymin-10), 0, 0.4, (0, 255, 0), 1)
        cv2.imshow("show", CV_IMG);cv2.waitKey(1)

def callback_img(msg):
    global CV_IMG
    # print("I get IMG")
    CV_IMG = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.putText(CV_IMG, "Red: LiDAR", (15, 15), 0, 0.6, (0, 0, 255), 1)
    cv2.putText(CV_IMG, "Green: Depthformer", (15, 30), 0, 0.6, (0, 255, 0), 1)

def callback_depth(depth_msg):
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(depth_msg, "mono16")


def callback_pcl(pcl_msg):
    global CLUSTER_COM_DEPTH, pub1, fields
    # be = rospy.Time.now()
    o3d_cloud = convertCloudFromRosToOpen3d(pcl_msg)
    down_cloud = open3d.geometry.PointCloud.voxel_down_sample(o3d_cloud, DOWN)
    print(down_cloud)
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


    CLUSTER_COM_DEPTH = [[],[]]
    for i, clus in enumerate(clusters):
        CLUSTER_COM_DEPTH[0].append(list(np.mean(clus, axis=0)))
        CLUSTER_COM_DEPTH[1].append(round(np.asarray(clus)[:,0].min(), 2))

    if FLAG_DB == True:
        points = []
        for i, cluster in enumerate(clusters):
            if np.shape(cluster)[0] < 70: continue
            for x, y, z in cluster:
                r, g, b = np.uint8(100*i), np.uint8(80*i), np.uint8(30*i)
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                point = [x, y, z, rgb]
                points.append(point)

        pc2 = point_cloud2.create_cloud(pcl_msg.header, fields, points)
        pub1.publish(pc2)



fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgb', 12, PointField.UINT32, 1)]
CV_IMG = None
depth_img = None
bridge = CvBridge()
rospy.init_node('show_2D', anonymous=True)
rospy.Subscriber("/pylon_camera_node/image_raw", Image, callback_img)
rospy.Subscriber("/rslidar_points", PointCloud2, callback_pcl)
rospy.Subscriber("/yolov5/detections", BoundingBoxes, callback_yolo)
rospy.Subscriber("/depth_img", Image, callback_depth)
pub1 = rospy.Publisher("/new_points", PointCloud2, queue_size=1)

rospy.spin()
