import numpy as np

from distort_points import distort_points, undistort_points

import cv2

def undistort_image_vectorized(img: np.ndarray,
                               K: np.ndarray,
                               D: np.ndarray) -> np.ndarray:

    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        und_img: undistorted image (HxW)
    """
    # Creating grid for undistorted image 
    img_height, img_width = img.shape
    X, Y = np.meshgrid(np.arange(img_width), np.arange(img_height))
    undistorted_img_grid = np.vstack([X.ravel(), Y.ravel()]).T

    # breakpoint()

    # Applying distortion to grid to get mapping to distorted image
    distorted_img_grid = undistort_points(undistorted_img_grid.T, D, K).T  # Applying transposes for proper input and output shape
    distorted_map_x, distorted_map_y = distorted_img_grid[:, 0], distorted_img_grid[:, 1]
    distorted_map_x = distorted_map_x.reshape(img_height, img_width).astype(np.float32)
    distorted_map_y = distorted_map_y.reshape(img_height, img_width).astype(np.float32)

    # breakpoint()

    # Performing sampling 
    undistorted_img = cv2.remap(img, distorted_map_x, distorted_map_y, interpolation=cv2.INTER_LINEAR)

    return undistorted_img
