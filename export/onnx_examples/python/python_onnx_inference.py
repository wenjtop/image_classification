import cv2
import onnxruntime
import numpy as np

images_path = '../../cat.jpg'
input_img = cv2.imread(images_path)

# 将图像从 BGR 转换为 RGB
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

# 将图像转换为浮点数，并进行归一化
input_img = input_img.astype(np.float32) / 255.0

input_img = cv2.resize(input_img, (256, 256))
# 将图像数据的维度调整为 (1, C, H, W)，即添加一个 batch 维度
input_img = np.transpose(input_img, (2, 0, 1))
input_img = np.expand_dims(input_img, axis=0)

ort_session = onnxruntime.InferenceSession("../mobilenet_v3_s.onnx")
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0][0]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print(softmax(ort_output))
