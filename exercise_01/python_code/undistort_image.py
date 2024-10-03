import math
import numpy as np

from distort_points import distort_points


def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """
    # Creating grid for undistorted image 
    img_height, img_width = img.shape

    undistorted_img = np.zeros([img_height, img_width])

    for x in range(img_width):
        for y in range(img_height):
            distorted_pixel_location = distort_points(np.array([[x, y]]).T, D, K).T
            u, v = distorted_pixel_location[0, :]

            u1, v1 = math.floor(u), math.floor(v)

            if bilinear_interpolation:
                a = u - u1
                b = v - v1
                if (u1 >= 0) & (u1+1 < img_width) & (v1 >= 0) & (v1+1 < img_height):
                    undistorted_img[y, x] = (1 - b) * (
                        (1 - a) * img[v1, u1] + a * img[v1, u1+1]
                    ) + b * (
                        (1 - a) * img[v1 + 1, u1] + a * img[v1 + 1, u1 + 1]
                        )
            else:
                if (u1 >= 0) & (u1 < img_width) & (v1 >= 0) & (v1 < img_height):
                    undistorted_img[y, x] = img[v1, u1]

    return undistorted_img
