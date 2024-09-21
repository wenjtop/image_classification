#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 帮助函数：加载图片并为模型进行预处理
cv::Mat preprocessImage(const std::string& image_path, const cv::Size& input_size) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("无法加载图片: " + image_path);
    }

    // 转换颜色通道顺序，从 BGR 到 RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // 调整图片大小以匹配模型输入
    cv::resize(image, image, input_size);
    // 转换图片数据类型为浮点型并归一化到 [0, 1] 范围
    image.convertTo(image, CV_32F);
    image /= 255.0f;

    return image;
}

std::vector<float> matToFloatVector(const cv::Mat& mat) {
    std::vector<float> data;
    // 将 HWC 格式转换为 CHW 格式
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);
    for (const auto& channel : channels) {
        data.insert(data.end(), channel.begin<float>(), channel.end<float>());
    }
    return data;
}

int main(int argc, char* argv[])
{
    // --- define model path
    const char* model_path = "../../mobilenet_v3_s.onnx";
    const std::string image_path = "../../../dog.jpg";  // 替换为你的图片路径

    // --- init onnxruntime env
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // set options
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(5); // extend the number to do parallel
    session_option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // --- prepare data
    const char* input_names[] = { "input" }; // must keep the same as model export
    const char* output_names[] = { "output" };

    // use statc array to preallocate data buffer
//    std::array<float, 1* 3 * 256 * 256> input_matrix;
    std::array<float, 1 * 2> output_matrix;

    // must use int64_t type to match args
    std::array<int64_t, 4> input_shape{ 1, 3, 256, 256};
    std::array<int64_t, 2> output_shape{ 1, 2 };

    int64_t height = 256;
    int64_t width = 256;

    cv::Size input_size(width, height);
    cv::Mat image = preprocessImage(image_path, input_size);

    // 1* 3 * 256 * 256 = 196608
    std::vector<float> input_matrix = matToFloatVector(image);

    int sample_y = 4;

    // [1, 3, 256, 256]
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_matrix.data(), input_matrix.size(), input_shape.data(), input_shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_matrix.data(), output_matrix.size(), output_shape.data(), output_shape.size());

    // --- predict
    Ort::Session session(env, model_path, session_option); // FIXME: must check if model file exist or valid, otherwise this will cause crash
    session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, &output_tensor, 1); // here only use one input output channel

    // --- result
    std::cout << "--- predict result ---" << std::endl;

    // matrix output
    std::cout << "ouput matrix: ";
    for (int i = 0; i < 2; i++)
        std::cout << output_matrix[i] << " ";
    std::cout << std::endl;
    // argmax value
    int argmax_value = std::distance(output_matrix.begin(), std::max_element(output_matrix.begin(), output_matrix.end()));
    std::cout << "output argmax value: " << argmax_value << std::endl;

    getchar();

    return 0;
}
