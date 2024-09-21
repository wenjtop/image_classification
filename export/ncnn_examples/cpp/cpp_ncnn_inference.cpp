#include <iostream>
#include <net.h>
#include <opencv2/opencv.hpp>

// 替换为你自己的类名
static const char* class_names[] = {"class0", "class1"};


static const char* model_param = "../../ncnn-20240410-ubuntu-2204/mobilenet_v3_s_ubuntu_2204.param";
static const char* model_bin = "../../ncnn-20240410-ubuntu-2204/mobilenet_v3_s_ubuntu_2204.bin";
static const char* image_path = "../../../cat.jpg";

int main(int argc, char** argv) {

    // 加载模型
    ncnn::Net net;
    if (net.load_param(model_param) != 0) {
        std::cerr << "Failed to load model param." << std::endl;
        return -1;
    }
    if (net.load_model(model_bin) != 0) {
        std::cerr << "Failed to load model bin." << std::endl;
        return -1;
    }

    // 读取图片
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to read image." << std::endl;
        return -1;
    }

    // 转换为 ncnn 的 Mat
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, 256, 256);

    // 归一化（根据训练时的预处理调整均值和标准差）
    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals[3] = { 1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f };
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 创建 ncnn extractor
    ncnn::Extractor ex = net.create_extractor();

    // 输入模型
    ex.input("input", in);

    // 输出模型
    ncnn::Mat out;
    ex.extract("output", out);

    // 解析输出
    float class0_score = out[0];
    float class1_score = out[1];

    // 打印结果
    std::cout << class_names[0] << ": " << class0_score << std::endl;
    std::cout << class_names[1] << ": " << class1_score << std::endl;

    // 判断类别
    int predicted_class = class0_score > class1_score ? 0 : 1;
    std::cout << "Predicted class: " << class_names[predicted_class] << std::endl;

    return 0;
}
