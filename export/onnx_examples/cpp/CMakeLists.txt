cmake_minimum_required(VERSION 3.10)

project(classification)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_DIR onnxruntime-linux-x64-1.19.0)  # 将此路径替换为您的 ONNX Runtime 安装路径
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(classification  cpp_onnx_inference.cpp)

target_link_libraries(classification ${OpenCV_LIBS})

target_link_libraries(classification onnxruntime)