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
    threshold = 0.2
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    transformation = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                       o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
    return transformation

if __name__ == '__main__':
    folder = '/home/mlv/research/SLAM/dataset/KITTI_LiDAR/dataset/sequences/'

    loader = Loader_KITTI(folder, '00')
    feature_extractor = Feature_Extractor_Loam()
    odometry_predictor = Odometry_Loam()
    mapper = Mapping_Loam()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    '''
    for i in range(loader.length()):
        if i >= 50:
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

    pcds = []
    global_transform = np.eye(4)
    #or i in range(loader.length()):
    for i in range(80,150):
        if i >= 50:
            pcd_np_1 = utils.get_pcd_from_numpy(loader.get_item(i)[0])
            pcd_np_2 = utils.get_pcd_from_numpy(loader.get_item(i + 1)[0])

            T = find_transformation(pcd_np_2, pcd_np_1, np.eye(4))
            print(T)
            T1 = copy.deepcopy(T)
            #T1[0][1] *= -1
            #T1[1][0] *= -1
            #T1[1][3] *= -1
            #T1[2][2] *= -1
            
            global_transform = T1 @ global_transform
            pcds.append(copy.deepcopy(pcd_np_2).transform(global_transform))
            #pcds.append(copy.deepcopy(pcd_np_1))

            vis.add_geometry(pcds[-1])
            vis.poll_events()
            vis.update_renderer()         


    #o3d.visualization.draw_geometries(pcds)

    vis.destroy_window()