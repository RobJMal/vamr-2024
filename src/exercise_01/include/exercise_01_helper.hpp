#ifndef EXERCISE_01_HELPER_H
#define EXERCISE_01_HELPER_H

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

/**
 * @brief Applies lens distortion to 2D points.
 * 
 * This function distorts points based on the distortion coefficients and camera matrix.
 * 
 * @param x Input 2D points (Nx2 matrix).
 * @param D Distortion coefficients (4x1 vector).
 * @param K Camera matrix (3x3 matrix).
 * @return Distorted 2D points (Nx2 matrix).
 */
Eigen::MatrixXd distort_points(const Eigen::MatrixXd &x, 
                                const Eigen::MatrixXd &D, 
                                const Eigen::MatrixXd &K);

#endif // EXERCISE_01_HELPER_H