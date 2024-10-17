import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    # Convert 2D to normalized coordinates
    p_augmented = np.hstack((p, np.ones((p.shape[0], 1)))) # Adding in column of ones to normalize
    p_normalized = (np.linalg.inv(K) @ p_augmented.T).T

    # Build measurement matrix Q
    num_points = p.shape[0] // 2
    Q_n = np.zeros((2, 12))   # Submatrix for each point 
    Q = np.zeros((2 * num_points, 12))
    for p_i in range(num_points):
        X_w_n, Y_w_n, Z_w_n = P[p_i, 0], P[p_i, 1], P[p_i, 2]
        x_n, y_n =  p_normalized[p_i, 0], p_normalized[p_i, 1] 

        # Q_n = np.array([
        #     [X_w_n, Y_w_n, Z_w_n, 1, 0, 0, 0, 0, -x_n*X_w_n, -x_n*Y_w_n, -x_n*Z_w_n, -x_n],
        #     [0, 0, 0, 0, X_w_n, Y_w_n, Z_w_n, 1, -y_n*X_w_n, -y_n*Y_w_n, -y_n*Z_w_n, -y_n],
        # ])

        Q[2*p_i:2*p_i+2, :] = np.array([
            [X_w_n, Y_w_n, Z_w_n, 1, 0, 0, 0, 0, -x_n*X_w_n, -x_n*Y_w_n, -x_n*Z_w_n, -x_n],
            [0, 0, 0, 0, X_w_n, Y_w_n, Z_w_n, 1, -y_n*X_w_n, -y_n*Y_w_n, -y_n*Z_w_n, -y_n],
        ])

        # breakpoint()
    # breakpoint()
    
    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    # TODO: Your code here
    
    # Extract [R | t] with the correct scale
    # TODO: Your code here

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    # TODO: Your code here

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    # TODO: Your code here

    # Build M_tilde with the corrected rotation and scale
    # TODO: Your code here
