import copy
import numpy as np
import open3d as o3d

from loader_kitti import Loader_KITTI
from feature_extractor_loam import Feature_Extractor_Loam
from odometry_loam import Odometry_Loam
from mapping_loam import Mapping_Loam
import utils

import time

def find_transformation(source, target, trans_init):
    threshold = 0.5
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    transformation = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                       o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
    return transformation


def change_axis(pcd, order):

    points = np.asarray(pcd.points)
    
    temp0 = copy.deepcopy(points[:,0])
    temp1 = copy.deepcopy(points[:,1])
    temp2 = copy.deepcopy(points[:,2])

    points[:,order[0]] = temp0
    points[:,order[1]] = temp1
    points[:,order[2]] = temp2

    output = o3d.geometry.PointCloud() 
    output.points = o3d.utility.Vector3dVector(points)   

    return output 


def transform_change_axis(pcd, T, color, order):

    output = change_axis(pcd, order)

    output = copy.deepcopy(output).transform(T)

    output = change_axis(output, order)

    output.paint_uniform_color(color)

    return output



if __name__ == '__main__':
    data_folder = '/home/mlv/research/SLAM/dataset/KITTI_LiDAR/dataset/sequences/'
    gt_folder = '/home/mlv/research/SLAM/dataset/data_odometry_poses/dataset/poses/'

    loader = Loader_KITTI(data_folder, gt_folder, '00')
    feature_extractor = Feature_Extractor_Loam()
    odometry_predictor = Odometry_Loam()
    mapper = Mapping_Loam()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vehicle = o3d.geometry.PointCloud()
    vehicle.points = o3d.utility.Vector3dVector(np.array([[0.0,0.0,3.0]]))
    #vehicle.color = o3d.utility.Vector3dVector(np.array([[1,0,0]]))
    vehicle.paint_uniform_color([0, 0, 1])

    map_points = o3d.geometry.PointCloud()

    vis.add_geometry(vehicle)
    vis.add_geometry(map_points)

    initial_calib = loader.load_calib()
    print(initial_calib)

    '''
    ################################################
    # LOAM
    ################################################
    for i in range(loader.length()):
        if i >= 120:
            print("###########################################################################")
            start = time.time()

            lidar_data = loader.get_item(i)
            loading_time = time.time()
            print("loading_time = ", loading_time - start)

            feature = feature_extractor.extract_features(lidar_data)
            feature_time = time.time()
            print("feature_time = ", feature_time - loading_time)

            T, points = odometry_predictor.predict(lidar_data, feature)
            odometry_time = time.time()
            print("odometry_time = ", odometry_time - feature_time)

            registered_pcd = mapper.append_undistorted(lidar_data[0], T, points)
            register_time = time.time()
            print("register_time = ", register_time - odometry_time)


            vis.add_geometry(registered_pcd)
            vis.poll_events()
            vis.update_renderer()
    '''

    '''
    ################################################
    # ICP
    ################################################
    pcds = []
    global_transform = np.eye(4)
    for i in range(80,350):
        if i >= 50:
            laser_data_1, gt_1 = loader.get_item(i)
            laser_data_2, gt_2 = loader.get_item(i+1)

            pcd_np_1 = utils.get_pcd_from_numpy(laser_data_1[0])
            pcd_np_2 = utils.get_pcd_from_numpy(laser_data_2[0])

            T = find_transformation(pcd_np_2, pcd_np_1, np.eye(4))

            print("########################################")
            print(T)
            global_transform = global_transform @ T

            pcds.append(transform_change_axis(pcd_np_2, global_transform, [0,0,1], [0,1,2]))

            vis.add_geometry(pcds[-1])
            #o3d.visualization.draw_geometries(pcds)
            #vis.update_geometry(pcds[-1])
            
            vis.poll_events()
            vis.update_renderer()
    '''

    
    ################################################
    # GT
    ################################################
    pcds = []
    global_transform = np.eye(4)
    for i in range(80,350):
        if i >= 50:
            laser_data_1, gt_1 = loader.get_item(i)
            laser_data_2, gt_2 = loader.get_item(i+1)

            pcd_np_1 = utils.get_pcd_from_numpy(laser_data_1[0])
            pcd_np_2 = utils.get_pcd_from_numpy(laser_data_2[0])

            gt_T1 = np.array( [[gt_1[0],gt_1[1],gt_1[2], gt_1[3]], [gt_1[4],gt_1[5],gt_1[6], gt_1[7]], [gt_1[8],gt_1[9],gt_1[10], gt_1[11]], [0,0,0,1] ] )
            gt_T2 = np.array( [[gt_2[0],gt_2[1],gt_2[2], gt_2[3]],[gt_2[4],gt_2[5],gt_2[6], gt_2[7]], [gt_2[8],gt_2[9],gt_2[10], gt_2[11]], [0,0,0,1] ] )

            print("########################################")
            #print(gt_T1)
            #print(gt_T2)
            global_transform = np.eye(4)
            temp = np.array([[0,0,1,0],
                            [ 1,0,0,0],
                            [ 0,1,0,0],
                            [ 0,0,0,1]])

            global_transform1 = temp @ gt_T1 @ initial_calib
            global_transform2 = temp @ gt_T2 @ initial_calib

            #pcds.append(transform_change_axis(pcd_np_1, global_transform1, [1,0,0], [0,1,2]))
            #if i%10 == 0:
            #pcds.append(transform_change_axis(pcd_np_2, global_transform2, [0,0,1], [0,1,2]))
            #o3d.visualization.draw_geometries(pcds)

            map_points.points.extend( transform_change_axis(pcd_np_2, global_transform2, [0,0,1], [0,1,2]).points )
            #map_points.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([map_points.voxel_down_sample(0.5)])
            #vis.add_geometry(map_points.voxel_down_sample(0.5))
            #vis.update_geometry(map_points)
        
            vis.poll_events()
            vis.update_renderer()
    

    #vis.destroy_window()
