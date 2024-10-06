#include <iostream>
#include "exercise_01_helper.hpp"
#include "vamr_utils.hpp"

int main() 
{    
    std::string camera_poses_file_path = "src/exercise_01/data/poses.txt";
    std::string K_intrinsic_matrix_file_path = "src/exercise_01/data/K.txt";
    std::string D_distortion_matrix_file_path = "src/exercise_01/data/D.txt";

    Eigen::MatrixXd camera_poses = load_matrix_from_file(camera_poses_file_path);
    Eigen::MatrixXd K_intrinsic_matrix = load_matrix_from_file(K_intrinsic_matrix_file_path);
    Eigen::MatrixXd D_distortion_matrix = load_matrix_from_file(D_distortion_matrix_file_path);

    std::cout << "Camera poses: " << std::endl << camera_poses << std::endl;

    return 0; 
}