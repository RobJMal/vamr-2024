#include "vamr_utils.hpp"

Eigen::MatrixXd load_matrix_from_file(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<double> values;
    std::string line;

    int rows = 0;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double val;
        while (iss >> val) {
            values.push_back(val);
        }
        ++rows;
    }

    if (rows == 0 || values.size() % rows != 0) {
        throw std::runtime_error("Error in determining matrix dimensions. Check the input file format.");
    }

    int cols = values.size() / rows;  // Calculate number of columns

    // Create Eigen matrix from the values
    Eigen::MatrixXd matrix = Eigen::Map<Eigen::MatrixXd>(values.data(), rows, cols);
    
    return matrix;
}
