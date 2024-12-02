#ifndef VAMR_UTILS_H
#define VAMR_UTILS_H

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>

/**
 * @brief Loads a matrix from a txt file.
 * 
 * This function distorts points based on the distortion coefficients and camera matrix.
 * 
 * @param file_path File path of the matrix.
 * @return Matrix from .txt file (n x m matrix).
 */
Eigen::MatrixXd load_matrix_from_file(const std::string& file_path);

#endif  // VAMR_UTILS_H