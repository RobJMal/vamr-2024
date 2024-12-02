import numpy as np

from distort_points import distort_points

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

    # Applying distortion to grid to get mapping to distorted image
    distorted_pixel_locations = distort_points(undistorted_img_grid.T, D, K).T  # Applying transposes for proper input and output shape
    intensity_vals = img[np.round(distorted_pixel_locations[:, 1].astype(np.int)),
                         np.round(distorted_pixel_locations[:, 0].astype(np.int))]
    undistorted_img = intensity_vals.reshape(img.shape).astype(np.uint8)

    return undistorted_img
