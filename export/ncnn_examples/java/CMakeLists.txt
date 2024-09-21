# 设置 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(classification)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV 包并包含它的头文件路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加 JVM 头文件路径
include_directories("/usr/lib/jvm/java-17-openjdk-amd64/include")
include_directories("/usr/lib/jvm/java-17-openjdk-amd64/include/linux")

# 查找 ncnn 包
set(ncnn_DIR /home/wenjtop/myProject/image_classification/export/ncnn_examples/ncnn-20240410-ubuntu-2204/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# 添加源文件
set(SOURCES ncnn_inference.cpp)

# 设置生成共享库
add_library(inference SHARED ${SOURCES})

# 为共享库设置编译选项
set_target_properties(inference PROPERTIES
    COMPILE_FLAGS "-fPIC"            # 添加 -fPIC 编译选项
    OUTPUT_NAME "inference"          # 设置输出库的名称
)

# 添加可执行文件
# add_executable(inference ncnn_inference.cpp)

# 链接 OpenCV 和 ncnn 库到共享库和可执行文件
target_link_libraries(inference ${OpenCV_LIBS} ncnn)
