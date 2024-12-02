import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points

    reprojected_points = np.zeros((P.shape[0], 2))

    P_augmented = np.hstack((P, np.ones((P.shape[0], 1)))) # Adding in column of ones to normalize
    projection_homogenous = (K @ M_tilde @ P_augmented.T).T

    reprojected_points = projection_homogenous[:, :2] / projection_homogenous[:, 2, np.newaxis]

    return reprojected_points
