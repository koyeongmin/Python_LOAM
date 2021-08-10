import numpy as np
import open3d as o3d


def get_pcd_from_numpy(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd


def matrix_row_wise_dot_product(A, B):
    """
    Fast way to calculate dot product of every row in two matrices A and B
    Returns array like:
    [ np.dot(A[0], B[0]),
    np.dot(A[1], B[1]),
    np.dot(A[2], B[2]),
    ... ,
    np.dot(A[M - 1], B[M - 1])]
    :param A: MxN numpy array
    :param B: MXN numpy array
    :return: Mx1 numpy array
    """
    assert A.shape == B.shape
    return np.einsum('ij,ij->i', A, B)  # r_i = SUM_j[A_ij * B_ij]

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
