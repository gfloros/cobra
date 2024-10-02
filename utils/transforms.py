import numpy as np


def transform_2D_to_3D(K, exrtinsics, P2D, P3D):

    # Invert the calibration matrix
    K_inv = np.linalg.inv(K)

    # Extract rotation matrix R and translation vector t from the extrinsic matrix
    R = exrtinsics[:, :3]
    t = exrtinsics[:, 3]

    # compute lambda
    l = ((K_inv @ P2D).T @ (R @ P3D + t)) / np.linalg.norm(K_inv @ P2D, ord=2) ** 2
    P3D_est = l * R.T @ K_inv @ P2D - R.T @ t

    return P3D_est


def transform_3D_to_2D(K, extrinsics, P3D):

    p2d_homo = K @ extrinsics[:3, :] @ P3D
    return p2d_homo[:-1] / p2d_homo[-1]