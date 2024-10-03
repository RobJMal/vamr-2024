import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized

# Section 2.2
def project_points_to_undistorted_image(camera_poses, K_matrix, D_matrix, plot=True):
    """
    Corresponds to Section 2.2 of the exercise. 
    """
    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system
    square_size = 0.04
    x_range = np.arange(0, 9, 1) * square_size
    y_range = np.arange(0, 6, 1) * square_size
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)    # Checkerboard is flat in world frame 
    checkerboard_corners_world = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # load one image with a given index
    img_index = 1
    img_path = '../data/images_undistorted/img_{:04d}.jpg'.format(img_index)
    img_undistorted = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    transformation_matrix = pose_vector_to_transformation_matrix(camera_poses[img_index - 1])

    # Appending a column of ones to the checkerboard_corners to apply transformation matrix 
    checkerboard_corners_world_homogeneous = np.hstack([checkerboard_corners_world, np.ones((checkerboard_corners_world.shape[0], 1))])
    checkerboard_corners_camera_homogenous = transformation_matrix @ checkerboard_corners_world_homogeneous.T

    # Convert back to 3D (x, y, z) by dividing by the homogeneous coordinate w (the fourth component)
    checkerboard_corners_camera = (checkerboard_corners_camera_homogenous[:3, :] / checkerboard_corners_camera_homogenous[3, :]).T

    # transform 3d points from world to current camera pose
    projected_points = project_points(checkerboard_corners_camera.T, K_matrix, D_matrix, distorted=False)

    # draw the projected points on the image
    if plot:
        plt.figure(figsize=(10, 7))
        plt.imshow(img_undistorted, cmap='gray')

        plt.scatter(projected_points[0, :], projected_points[1, :], c='r', s=10, marker='o', label='Projected points')

        plt.xlim([0, img_undistorted.shape[1]]) # Image width
        plt.ylim([img_undistorted.shape[0], 0]) # Image height

        plt.title("Projected points on undistorated image")
        plt.legend()
        plt.show()

# Section 2.3
def draw_cube_to_unidistorted_image(camera_poses, K_matrix, D_matrix, 
                                    x_unit_coord_start_vertex=0, 
                                    y_unit_coord_start_vectex=0,
                                    cube_unit_size=4,
                                    plot=True):
    """
    Draws a cube on the undistorted image.

    Corresponds to Section 2.3 of the exercise. 
    """

    # Defining the cube points expressed in world coordinate system 
    def generate_cube_points(x_unit_coord_start_vertex, y_unit_coord_start_vertex, cube_unit_size=2, grid_size=0.04):
        """
        Generates the points of the cube given the start vertex and the size of the cube.

        Note that cube points are returned based on plotting convention established in exercise. 

        Args:
            x_unit_coord_start_vertex: x coordinate of the start vertex. Using unit coordinates
            y_unit_coord_start_vertex: y coordinate of the start vertex. Using unit coordinates
            grid_size: size of the grid in meters
            cube_unit_size: size of the cube

        Returns:
            cube_points: 3D points (8x3)
        """
        # Check the input x and y coordinates to make sure it is valid
        assert x_unit_coord_start_vertex >= 0, "x_unit_coord_start_vertex should be greater than or equal to 0"
        assert y_unit_coord_start_vertex >= 0, "y_unit_coord_start_vertex should be greater than or equal to 0"
        assert x_unit_coord_start_vertex < 9 - 1, "x_unit_coord_start_vertex should be less than 8" # Limiting to 9 - 1 to avoid going out of checkerboard
        assert y_unit_coord_start_vertex < 6 - 1, "y_unit_coord_start_vertex should be less than 5" # Limiting to 6 - 1 to avoid going out of checkerboard

        x_vertex = x_unit_coord_start_vertex * grid_size
        y_vertex = y_unit_coord_start_vertex * grid_size
        cube_size = cube_unit_size * grid_size

        # z coordinate is negative to make sure the cube above the checkerboard
        # (based on world coordinate system)
        cube_points = np.array([
            [x_vertex, y_vertex, -cube_size],                       # Vertex 0, top layer of cube
            [x_vertex, y_vertex, 0],                                # Vertex 1, bottom layer of cube
            [x_vertex+cube_size, y_vertex, -cube_size],             # Vertex 2, top layer of cube
            [x_vertex+cube_size, y_vertex, 0],                      # Vertex 3, bottom layer of cube
            [x_vertex, y_vertex+cube_size, -cube_size],             # Vertex 4, top layer of cube
            [x_vertex, y_vertex+cube_size, 0],                      # Vertex 5, bottom layer of cube
            [x_vertex+cube_size, y_vertex+cube_size, -cube_size],   # Vertex 6, top layer of cube
            [x_vertex+cube_size, y_vertex+cube_size, 0],            # Vertex 7, bottom layer of cube
            ])

        return cube_points
    
    cube_points_world = generate_cube_points(x_unit_coord_start_vertex=x_unit_coord_start_vertex, 
                                             y_unit_coord_start_vertex=y_unit_coord_start_vectex, 
                                             cube_unit_size=cube_unit_size, grid_size=0.04)
    
    # load one image with a given index
    img_index = 1
    img_path = '../data/images_undistorted/img_{:04d}.jpg'.format(img_index)
    img_undistorted = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    transformation_matrix = pose_vector_to_transformation_matrix(camera_poses[img_index - 1])

    # Appending a column of ones to the cube_points to apply transformation matrix 
    cube_points_world_homogeneous = np.hstack([cube_points_world, np.ones((cube_points_world.shape[0], 1))])
    cube_points_camera_homogenous = transformation_matrix @ cube_points_world_homogeneous.T

    # Convert back to 3D (x, y, z) by dividing by the homogeneous coordinate w (the fourth component)
    cube_points_camera = (cube_points_camera_homogenous[:3, :] / cube_points_camera_homogenous[3, :]).T

    # transform 3d points from world to current camera pose
    cube_pts = project_points(cube_points_camera.T, K_matrix, D_matrix, distorted=False).T

    # Plot the cube
    if plot: 
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

# Section 3.2
def project_points_to_distorted_image(camera_poses, K_matrix, D_matrix, plot=True):
    """
    Corresponds to Section 3.2 of the exercise. 
    """
    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system
    square_size = 0.04
    x_range = np.arange(0, 9, 1) * square_size
    y_range = np.arange(0, 6, 1) * square_size
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)    # Checkerboard is flat in world frame 
    checkerboard_corners_world = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # load one image with a given index
    img_index = 1
    img_path = '../data/images/img_{:04d}.jpg'.format(img_index)
    img_distorted = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame
    transformation_matrix = pose_vector_to_transformation_matrix(camera_poses[img_index - 1])

    # Appending a column of ones to the checkerboard_corners to apply transformation matrix 
    checkerboard_corners_world_homogeneous = np.hstack([checkerboard_corners_world, np.ones((checkerboard_corners_world.shape[0], 1))])
    checkerboard_corners_camera_homogenous = transformation_matrix @ checkerboard_corners_world_homogeneous.T

    # Convert back to 3D (x, y, z) by dividing by the homogeneous coordinate w (the fourth component)
    checkerboard_corners_camera = (checkerboard_corners_camera_homogenous[:3, :] / checkerboard_corners_camera_homogenous[3, :]).T

    # transform 3d points from world to current camera pose
    projected_points = project_points(checkerboard_corners_camera.T, K_matrix, D_matrix, distorted=True)

    # draw the projected points on the image
    if plot:
        plt.figure(figsize=(10, 7))
        plt.imshow(img_distorted, cmap='gray')

        plt.scatter(projected_points[0, :], projected_points[1, :], c='r', s=10, marker='o', label='Projected points')

        plt.xlim([0, img_distorted.shape[1]]) # Image width
        plt.ylim([img_distorted.shape[0], 0]) # Image height

        plt.title("Projected points on undistorated image")
        plt.legend()
        plt.show()

# Section 3.3
def undistort_image_exercise(img, K, D, bilinear_interpolation=True, plot=True): 
    """
    Corresponds to Section 3.3 of the exercise. 
    """
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=False)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    if plot:
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

def main():

    # Load camera poses
    camera_poses = np.loadtxt('../data/poses.txt')

    # Load camera intrinsics and distortion coefficients
    K_matrix = np.loadtxt('../data/K.txt')
    D_matrix = np.loadtxt('../data/D.txt')

    # # Project points to undistorted image
    # project_points_to_undistorted_image(camera_poses, K_matrix, D_matrix)

    # # Draw cube on undistorted image
    # draw_cube_to_unidistorted_image(camera_poses, K_matrix, D_matrix)

    # Project points to distorted image
    # project_points_to_distorted_image(camera_poses, K_matrix, D_matrix)

    # undistort image with bilinear interpolation
    img_index = 1
    img_path = '../data/images/img_{:04d}.jpg'.format(img_index)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    undistort_image_exercise(img, K_matrix, D_matrix)
    


if __name__ == "__main__":
    main()
