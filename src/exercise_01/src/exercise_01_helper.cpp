#include "exercise_01_helper.hpp"

Eigen::MatrixXd distort_points(const Eigen::MatrixXd &x, 
                                const Eigen::MatrixXd &D, 
                                const Eigen::MatrixXd &K) 
{
    Eigen::Vector2d optical_center = K.block<2,1>(0, 2);

    Eigen::VectorXd x_normalized = (x - optical_center);

    Eigen::MatrixXd distorted_points; 

    return distorted_points;
}
