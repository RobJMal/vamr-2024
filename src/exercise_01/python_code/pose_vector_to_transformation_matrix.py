import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    axis_angle_rotation_vec = pose_vec[:3]
    translational_vec = pose_vec[3:]

    # Applying Rodrigues' formula to get the rotation matrix
    theta = np.linalg.norm(axis_angle_rotation_vec)
    k = axis_angle_rotation_vec / theta
    k_cross_product_matrix = np.array([[0, -k[2], k[1]],
                                       [k[2], 0, -k[0]],
                                       [-k[1], k[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(theta) * k_cross_product_matrix + (1 - np.cos(theta)) * k_cross_product_matrix @ k_cross_product_matrix

    transformation_matrix : np.ndarray = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translational_vec

    return transformation_matrix 
