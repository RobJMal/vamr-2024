#include <iostream>
#include "exercise_01_helper.hpp"
#include "vamr_utils.hpp"

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>


// Section 2.2
void project_points_to_undistorted_image()
{
    // Define 3D corner positions
    float square_size = 0.04;   // 4 cm
    int num_corners_x = 9;
    int num_corners_y = 6;

    xt::xarray<float> x_range = xt::linspace<float>(0, num_corners_x - 1, num_corners_x) * square_size;
    xt::xarray<float> y_range = xt::linspace<float>(0, num_corners_y - 1, num_corners_y) * square_size;

    auto [X, Y] = xt::meshgrid(x_range, y_range);
    auto Z = xt::zeros<float>(X.shape());

    checkerboard_corners_w = xt::stack(xt::xtuple(X, Y, Z), 2);

    // Eigen::VectorXd x_range = Eigen::VectorXd::LinSpaced(num_corners_x, 0, num_corners_x - 1) * square_size;
    // Eigen::VectorXd y_range = Eigen::VectorXd::LinSpaced(num_corners_y, 0, num_corners_y - 1) * square_size;

    // Eigen::MatrixXd X, Y;

    // xt::xarray<double> Z = create_meshgrid(x_range, y_range, X, Y)

    // create_meshgrid(x_range, y_range, X, Y);
    // Z
}

// Function to convert Eigen::MatrixXd to xt::xarray
xt::xarray<double> eigen_to_xtensor(const Eigen::MatrixXd& eigen_matrix) {
    std::vector<size_t> shape = {static_cast<size_t>(eigen_matrix.rows()), static_cast<size_t>(eigen_matrix.cols())};

    // Use xt::adapt to create xtensor from Eigen matrix data
    xt::xarray<double> xtensor_data = xt::adapt(eigen_matrix.data(), shape);

    return xtensor_data;
}

void create_meshgrid(const Eigen::VectorXd& x, const Eigen::VectorXd& y, Eigen::MatrixXd& X, Eigen::MatrixXd& Y) {
    // X will have each row identical to x
    X = Eigen::MatrixXd(y.size(), x.size());
    for (int i = 0; i < y.size(); ++i) {
        X.row(i) = x.transpose();  // Fill each row of X with the values of x
    }

    // Y will have each column identical to y
    Y = Eigen::MatrixXd(y.size(), x.size());
    for (int i = 0; i < x.size(); ++i) {
        Y.col(i) = y;  // Fill each column of Y with the values of y
    }
}

int main() 
{    
    std::string camera_poses_file_path = "src/exercise_01/data/poses.txt";
    std::string K_intrinsic_matrix_file_path = "src/exercise_01/data/K.txt";
    std::string D_distortion_matrix_file_path = "src/exercise_01/data/D.txt";

    Eigen::MatrixXd camera_poses = load_matrix_from_file(camera_poses_file_path);
    Eigen::MatrixXd K_intrinsic_matrix = load_matrix_from_file(K_intrinsic_matrix_file_path);
    Eigen::MatrixXd D_distortion_matrix = load_matrix_from_file(D_distortion_matrix_file_path);

    // std::cout << "Camera poses: " << std::endl << camera_poses << std::endl;
    // xt::xarray<double> camera_poses_xtensor = eigen_to_xtensor(camera_poses);
    // std::cout << "Camera poses xtensor: " << std::endl << camera_poses_xtensor << std::endl;

    return 0; 
}