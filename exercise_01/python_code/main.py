import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized

# Addtional imports 


def main():
    # load camera poses
    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame
    camera_poses = np.loadtxt('../data/poses.txt')

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system
    square_size = 0.04
    x_range = np.arange(0, 9, 1) * square_size
    y_range = np.arange(0, 6, 1) * square_size
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)    # Checkerboard is flat in world frame 
    checkerboard_corners_world = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # load camera intrinsics and distortion coefficients
    K_matrix = np.loadtxt('../data/K.txt')
    D_matrix = np.loadtxt('../data/D.txt')

    # load one image with a given index
    image_index = 1
    image_path = '../data/images_undistorted/img_{:04d}.jpg'.format(image_index)
    image_i = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    transformation_matrix = pose_vector_to_transformation_matrix(camera_poses[image_index - 1])

    # Appending a column of ones to the checkerboard_corners to apply transformation matrix 
    checkerboard_corners_world_homogeneous = np.hstack([checkerboard_corners_world, np.ones((checkerboard_corners_world.shape[0], 1))])
    checkerboard_corners_camera_homogenous = transformation_matrix @ checkerboard_corners_world_homogeneous.T

    # Convert back to 3D (x, y, z) by dividing by the homogeneous coordinate w (the fourth component)
    checkerboard_corners_camera = (checkerboard_corners_camera_homogenous[:3, :] / checkerboard_corners_camera_homogenous[3, :]).T

    # transform 3d points from world to current camera pose
    projected_points = project_points(checkerboard_corners_camera.T, K_matrix, D_matrix)

    # draw the projected points on the image
    plt.figure(figsize=(10, 7))
    plt.imshow(image_i, cmap='gray')

    plt.scatter(projected_points[0, :], projected_points[1, :], c='r', s=10, marker='o', label='Projected points')

    plt.xlim([0, image_i.shape[1]]) # Image width
    plt.ylim([image_i.shape[0], 0]) # Image height

    plt.title("Projected points on undistorated image")
    plt.legend()
    plt.show()


    # undistort image with bilinear interpolation
    """ Remove this comment if you have completed the code until here
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()
    """

    # calculate the cube points to then draw the image
    # TODO: Your code here
    
    # Plot the cube
    """ Remove this comment if you have completed the code until here
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 3

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()
    """


if __name__ == "__main__":
    main()
