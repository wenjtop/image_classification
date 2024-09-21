cmake_minimum_required(VERSION 3.10)

project(classification)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_DIR ../onnxruntime-linux-x64-1.19.0/lib/cmake/onnxruntime)  # 将此路径替换为您的 ONNX Runtime 安装路径
include_directories(../onnxruntime-linux-x64-1.19.0/include)
link_directories(../onnxruntime-linux-x64-1.19.0/lib)

# 添加 JVM 头文件路径
include_directories("/usr/lib/jvm/java-17-openjdk-amd64/include")
include_directories("/usr/lib/jvm/java-17-openjdk-amd64/include/linux")

# 添加源文件
set(SOURCES onnx_inference.cpp)

# 设置生成共享库
add_library(inference SHARED ${SOURCES})

# 为共享库设置编译选项
set_target_properties(inference PROPERTIES
    COMPILE_FLAGS "-fPIC"            # 添加 -fPIC 编译选项
    OUTPUT_NAME "inference"          # 设置输出库的名称
)

target_link_libraries(inference ${OpenCV_LIBS})

target_link_libraries(inference onnxruntime)