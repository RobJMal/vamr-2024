import numpy as np


def distort_points(x: np.ndarray,
                   D: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points xon the image plane.

    Args:
        x: 2d points (2xN)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)

    Returns:
        distorted_points: distorted 2d points (2xN)
    """
    optical_center = K[0:2, 2]
    focal_length = np.array([K[0, 0], K[1, 1]])
    x = x.T # Transpose to make math easier

    x_normalized = (x - optical_center) / focal_length

    r_squared = x_normalized[0]**2 + x_normalized[1]**2
    radial_distortion = 1 + D[0]*r_squared + D[1]*r_squared**2

    x_distorted = x_normalized * radial_distortion
    distorted_points = (x_distorted * focal_length) + optical_center

    return distorted_points.T   # Transpose to match 
