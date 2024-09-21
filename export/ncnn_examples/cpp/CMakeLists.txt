# 设置 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(classification)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(classify cpp_ncnn_inference.cpp)

# 查找 OpenCV 包并包含它的头文件路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 手动指定 ncnn 安装路径
# set(NCNN_DIR ../ncnn-20240410-ubuntu-2204)

# 使用 find_package 找到 ncnn 库
# find_package(ncnn REQUIRED PATHS ${NCNN_DIR})

set(ncnn_DIR /home/wenjtop/myProject/image_classification/export/ncnn_examples/ncnn-20240410-ubuntu-2204/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# 包含 ncnn 的头文件
# include_directories(${ncnn_INCLUDE_DIRS})

# 链接 OpenCV 和 ncnn 库
target_link_libraries(classify ${OpenCV_LIBS} ncnn)
