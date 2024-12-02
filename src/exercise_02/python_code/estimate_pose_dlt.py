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
    num_corners = p_normalized.shape[0]
    Q = np.zeros((2 * num_corners, 12))

    for p_i in range(num_corners):
        X_w_n, Y_w_n, Z_w_n = P[p_i, 0], P[p_i, 1], P[p_i, 2]
        x_n, y_n =  p_normalized[p_i, 0], p_normalized[p_i, 1] 

        Q[2*p_i:2*p_i+2, :] = np.array([
            [X_w_n, Y_w_n, Z_w_n, 1, 0, 0, 0, 0, -x_n*X_w_n, -x_n*Y_w_n, -x_n*Z_w_n, -x_n],
            [0, 0, 0, 0, X_w_n, Y_w_n, Z_w_n, 1, -y_n*X_w_n, -y_n*Y_w_n, -y_n*Z_w_n, -y_n],
        ])

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    U, S, V_t = np.linalg.svd(Q)
    M_tilde = V_t[-1,:]
    M_tilde /= np.linalg.norm(M_tilde)  # Enforces constraint ||M_tilde||=1
    
    # Extract [R | t] with the correct scale
    M_tilde = M_tilde.reshape(3, 4)
    R_scaled = M_tilde[:, 0:3]
    t_scaled = M_tilde[:, -1]

    # Ensuring that determinant is +1 
    if t_scaled[-1] < 0:
        R_scaled *= -1
        t_scaled *= -1

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    U_R_scaled, S_R_scaled, V_t_R_scaled = np.linalg.svd(R_scaled)
    R_tilde = U_R_scaled @ V_t_R_scaled

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    scale_factor = np.linalg.norm(R_tilde, 'fro')/np.linalg.norm(R_scaled, 'fro')
    # scale_factor = np.linalg.norm(R_tilde)/np.linalg.norm(R_scaled)

    # Checking R_tilde orthogonality
    assert np.isclose(np.linalg.det(R_tilde), 1.0, atol=1e-10)
    assert np.allclose(R_tilde.T @ R_tilde - np.eye(3), np.zeros((3, 3)), atol=1e-10)   

    # Build M_tilde with the corrected rotation and scale
    M_tilde[:, 0:3] = R_tilde
    M_tilde[:, -1] = scale_factor * t_scaled

    return M_tilde
