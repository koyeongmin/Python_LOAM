import numpy as np
import os

class Loader_KITTI:
    def __init__(self, dataset_path, sequence):
        self.N_SCANS = 64
        self.folder_path = os.path.join(dataset_path, sequence, 'velodyne')
        self.pcds_list = os.listdir(self.folder_path)
        self.pcds_list.sort()

    def length(self):
        return len(self.pcds_list)

    def get_item(self, ind):
        path = os.path.join(self.folder_path, self.pcds_list[ind])
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
        return self.reorder_pcd(pcd)

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