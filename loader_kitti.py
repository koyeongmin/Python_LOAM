import numpy as np
import os
import utils

class Loader_KITTI:
    def __init__(self, dataset_path, gt_path, sequence):
        self.N_SCANS = 64

        self.data_folder_path = os.path.join(dataset_path, sequence, 'velodyne')
        self.pcds_list = os.listdir(self.data_folder_path)
        self.pcds_list.sort()

        self.gt = np.loadtxt(gt_path+sequence+".txt")
        #print(self.gt.shape)
        #self.gt_file = open(gt_path+sequence+".txt", 'r')
        #self.gt = self.gt_file.readlines()
        print(len(self.pcds_list))
        print(len(self.gt))

    def length(self):
        return len(self.pcds_list)

    def get_item(self, ind):
        path = os.path.join(self.data_folder_path, self.pcds_list[ind])
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        return self.reorder_pcd(pcd), self.gt[ind]

    def _get_scan_ids(self, pcd):
        depth = np.linalg.norm(pcd[:, :3], 2, axis=1)
        pitch = np.arcsin(pcd[:, 2] / depth)
        fov_down = -24.8 / 180.0 * np.pi
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        scan_ids = (pitch + abs(fov_down)) / fov
        scan_ids *= self.N_SCANS
        scan_ids = np.floor(scan_ids)
        scan_ids = np.minimum(self.N_SCANS - 1, scan_ids)
        scan_ids = np.maximum(0, scan_ids).astype(np.int32)
        return scan_ids

    def reorder_pcd(self, pcd):
        scan_start = np.zeros(self.N_SCANS, dtype=int)
        scan_end = np.zeros(self.N_SCANS, dtype=int)

        scan_ids = self._get_scan_ids(pcd)
        sorted_ind = np.argsort(scan_ids, kind='stable')
        sorted_pcd = pcd[sorted_ind]
        sorted_scan_ids = scan_ids[sorted_ind]

        elements, elem_cnt = np.unique(sorted_scan_ids, return_counts=True)

        start = 0
        for ind, cnt in enumerate(elem_cnt):
            scan_start[ind] = start
            start += cnt
            scan_end[ind] = start
        ordered_point_clouds = np.hstack((sorted_pcd, sorted_scan_ids.reshape((-1, 1))))
        return ordered_point_clouds, scan_start, scan_end

    def load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        # Load the calibration file
        calib_filepath = os.path.join('/home/mlv/research/SLAM/dataset/data_odometry_calib/dataset/sequences/00/calib.txt')
        filedata = utils.read_calib_file(calib_filepath)

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        #self.calib = namedtuple('CalibData', data.keys())(*data.values())

        return data['T_cam0_velo']
