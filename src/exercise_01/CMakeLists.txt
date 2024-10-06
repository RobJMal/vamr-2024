cmake_minimum_required(VERSION 3.5)
project(exercise_01)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/exercise_01/include)

# Find packages
find_package(OpenCV REQUIRED)

# Specify the executable
add_executable(${PROJECT_NAME}_executable src/main.cpp src/vamr_utils.cpp src/exercise_01_helper.cpp)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME}_executable ${OpenCV_LIBS})

# Specify include directories
target_include_directories(${PROJECT_NAME}_executable PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Install target
install(TARGETS ${PROJECT_NAME}_executable
        RUNTIME DESTINATION bin)
