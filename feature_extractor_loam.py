import math
import numpy as np
import open3d as o3d
import utils

class Feature_Extractor_Loam:

    # Number of segments to split every scan for feature detection
    N_SEGMENTS = 6
    # Number of less sharp points to pick from point cloud
    PICKED_NUM_LESS_SHARP = 20
    # Number of sharpest points to pick from point cloud
    PICKED_NUM_SHARP = 4
    # Number of less flat points to pick from point cloud
    PICKED_NUM_FLAT = 4
    # Threshold to split sharp and flat points
    SURFACE_CURVATURE_THRESHOLD = 0.1
    SHARP_CURVATURE_THRESHOLD = 0.5
    # Radius of points for curvature analysis (S/2 from original paper, 5.1 section)
    FEATURES_REGION = 5


    def extract_features(self, lidar_data):
        ordered_point_clouds, scan_start, scan_end = lidar_data

        #compute curvatures of all points
        cloud_curvatures = self.get_curvatures(ordered_point_clouds)

        keypoints_sharp = []  # some sharpest points
        keypoints_less_sharp = [] # sharp points, but less than above
        keypoints_flat = []
        keypoints_less_flat = []

        cloud_label = np.zeros((ordered_point_clouds.shape[0]))
        cloud_neighbors_picked = np.zeros((ordered_point_clouds.shape[0]))

        #remove unreliable points, fig. 4. But the code looks different from the paper.
        cloud_neighbors_picked = self.remove_unreliable(cloud_neighbors_picked, ordered_point_clouds, scan_start, scan_end)

        # get each scan(=each lidar channel)(velodyne 64 has 64 channel)
        for i in range(scan_end.shape[0]):  
            scan_start_ind = scan_start[i] + self.FEATURES_REGION  #cut some points at boundary(=FEATURES_REGION)
            scan_end_ind = scan_end[i] - self.FEATURES_REGION - 1
            if scan_end_ind - scan_start_ind < self.N_SEGMENTS:
                continue

            # divide each scan to some segments(subregion)
            for j in range(self.N_SEGMENTS): 
                segment_start_point_ind = scan_start_ind + (scan_end_ind - scan_start_ind) * j // self.N_SEGMENTS
                segment_end_point_ind = scan_start_ind + (scan_end_ind - scan_start_ind) * (j + 1) // self.N_SEGMENTS - 1

                segments_curvatures = cloud_curvatures[segment_start_point_ind:segment_end_point_ind + 1]
                sort_indices = np.argsort(segments_curvatures) # to find largest curvature

                #find edge points
                largest_picked_num = 0
                for k in reversed(range(segment_end_point_ind - segment_start_point_ind)):
                    if i < 45:
                        break
                    ind = sort_indices[k] + segment_start_point_ind

                    # check each point can be edge
                    if cloud_neighbors_picked[ind] == 0 and cloud_curvatures[ind] > self.SHARP_CURVATURE_THRESHOLD and self.can_be_edge(ordered_point_clouds, ind):
                        largest_picked_num += 1
                        if largest_picked_num <= self.PICKED_NUM_SHARP:
                            keypoints_sharp.append(ordered_point_clouds[ind])
                            keypoints_less_sharp.append(ordered_point_clouds[ind])
                            cloud_label[ind] = 2
                        elif largest_picked_num <= self.PICKED_NUM_LESS_SHARP:
                            keypoints_less_sharp.append(ordered_point_clouds[ind])
                            cloud_label[ind] = 1
                        else:
                            break

                        # marking points that are already checked
                        cloud_neighbors_picked = self.mark_as_picked(ordered_point_clouds, cloud_neighbors_picked, ind)

                #find plane points
                smallest_picked_num = 0
                for k in range(segment_end_point_ind - segment_start_point_ind):
                    if i < 50:
                        break
                    ind = sort_indices[k] + segment_start_point_ind

                    if cloud_neighbors_picked[ind] == 0 and cloud_curvatures[ind] < self.SURFACE_CURVATURE_THRESHOLD:
                        smallest_picked_num += 1
                        cloud_label[ind] = -1
                        keypoints_flat.append(ordered_point_clouds[ind])

                        if smallest_picked_num >= self.PICKED_NUM_FLAT:
                            break

                        # marking points that are already checked
                        cloud_neighbors_picked = self.mark_as_picked(ordered_point_clouds, cloud_neighbors_picked, ind)

                for k in range(segment_start_point_ind, segment_end_point_ind + 1):
                    if cloud_label[k] <= 0 and cloud_curvatures[k] < self.SURFACE_CURVATURE_THRESHOLD and not self.has_gap(ordered_point_clouds, k):
                        keypoints_less_flat.append(ordered_point_clouds[k])


        return keypoints_sharp, keypoints_less_sharp, keypoints_flat, keypoints_less_flat

    '''
    Compute curvatures in FEATURES_REGION(=5)
    In paper, 5.1
    '''
    def get_curvatures(self, ordered_point_clouds):
        coef = [1, 1, 1, 1, 1, -10, 1, 1, 1, 1, 1] # features_region is 5
        assert len(coef) == 2 * self.FEATURES_REGION + 1
        discr_diff = lambda x: np.convolve(x, coef, 'valid') # In paper 5.1, summation part
        x_diff = discr_diff(ordered_point_clouds[:, 0])
        y_diff = discr_diff(ordered_point_clouds[:, 1])
        z_diff = discr_diff(ordered_point_clouds[:, 2])
        curvatures = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
        curvatures /= np.linalg.norm(ordered_point_clouds[self.FEATURES_REGION:-self.FEATURES_REGION], axis=1) * 10 # normalization
        curvatures = np.pad(curvatures, self.FEATURES_REGION)
        return curvatures


    def remove_unreliable(self, cloud_neighbors_picked, ordered_point_clouds, scan_start, scan_end):
        for i in range(scan_end.shape[0]):
            sp = scan_start[i] + self.FEATURES_REGION
            ep = scan_end[i] - self.FEATURES_REGION

            if ep - sp < self.N_SEGMENTS:
                continue

            for j in range(sp + 1, ep):
                prev_point = ordered_point_clouds[j - 1][:3]
                point = ordered_point_clouds[j][:3]
                next_point = ordered_point_clouds[j + 1][:3]

                diff_next = np.dot(point - next_point, point - next_point)

                if diff_next > 0.1:
                    depth1 = np.linalg.norm(point)
                    depth2 = np.linalg.norm(next_point)

                    if depth1 > depth2:
                        weighted_dist = np.sqrt(np.dot(point - next_point * depth2 / depth1,
                                                       point - next_point * depth2 / depth1)) / depth2
                        if weighted_dist < 0.1:
                            cloud_neighbors_picked[j - self.FEATURES_REGION:j + 1] = 1
                            continue
                    else:
                        weighted_dist = np.sqrt(np.dot(point - next_point * depth1 / depth2,
                                                       point - next_point * depth1 / depth2)) / depth1

                        if weighted_dist < 0.1:
                            cloud_neighbors_picked[j - self.FEATURES_REGION: j + self.FEATURES_REGION + 1] = 1
                            continue
                    diff_prev = np.dot(point - prev_point, point - prev_point)
                    dis = np.dot(point, point)

                    if diff_next > 0.0002 * dis and diff_prev > 0.0002 * dis:
                        cloud_neighbors_picked[j] = 1

        return cloud_neighbors_picked

    def mark_as_picked(self, ordered_point_clouds, cloud_neighbors_picked, ind):
        cloud_neighbors_picked[ind] = 1

        diff_all = ordered_point_clouds[ind - self.FEATURES_REGION + 1:ind + self.FEATURES_REGION + 2] - \
                   ordered_point_clouds[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION + 1]

        sq_dist = utils.matrix_row_wise_dot_product(diff_all[:, :3], diff_all[:, :3])

        for i in range(1, self.FEATURES_REGION + 1):
            if sq_dist[i + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + i] = 1

        for i in range(-self.FEATURES_REGION, 0):
            if sq_dist[i + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + i] = 1

        return cloud_neighbors_picked

    def has_gap(self, ordered_point_clouds, ind):
        diff_S = ordered_point_clouds[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION + 1, :3] - ordered_point_clouds[ind, :3]
        sq_dist = utils.matrix_row_wise_dot_product(diff_S[:, :3], diff_S[:, :3])
        gapped = sq_dist[sq_dist > 0.3]
        if gapped.shape[0] > 0:
            return True
        else:
            return False

    def can_be_edge(self, ordered_point_clouds, ind):
        diff_S = ordered_point_clouds[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION, :3] -\
                 ordered_point_clouds[ind - self.FEATURES_REGION + 1:ind + self.FEATURES_REGION + 1, :3]
        sq_dist = utils.matrix_row_wise_dot_product(diff_S[:, :3], diff_S[:, :3])
        gapped = ordered_point_clouds[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION, :3][sq_dist > 0.2]
        if len(gapped) == 0:
            return True
        else:
            return np.any(np.linalg.norm(gapped, axis=1) > np.linalg.norm(ordered_point_clouds[ind][:3]))