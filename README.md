# image_classification

作者：wenjtop

邮箱：1007131354@qq.com

### 仓库主要任务是图片分类，对训练策略做了优化，添加了数据增强，基本能还原论文精度，支持onnx的python、C++、java、javascript和ncnn的C++和java部署。支持以下模型：

```
# ['lng_n', 'lng_t', 'lng_s', 'lng_b',
#  'resnet34', 'resnet_50', 'resnet_101', 'resnext50_32x4d', 'resnext101_32x8d'
#  'vit_b', 'vit_l', input 224x224 7的倍数
#  'swin_t', 'swin_s', 'swin_b', input 224x224 7的倍数
#  'conv_next_t', 'conv_next_s', 'conv_next_b', 'conv_next_l',
#  'effNet_b1', 'effNet_b2', 'effNet_b3', 'effNet_b4', 'effNet_b5', 'effNet_b6', 'effNet_b7',
#  'effnet_v2_s', 'effNet_v2_m', 'effnet_v2_l',
#  'mobilenet_v2', 'mobilenet_v3_s', 'mobilenet_v3_l']
```

## 1、环境

缺什么安装什么，环境都比较常规

### 2、数据集格式

catdog数据文件格式：

```
├── catdog
      ├── classname.yaml
      ├── v1
      │   ├── train
      │   │   ├── cat
      │   │   └── dog
      │   └── val
      │       ├── cat
      │       └── dog
      └── v2
          ├── train
          │   ├── cat
          │   └── dog
          └── val
              ├── cat
              └── dog
```

优先读取存在的classname.yaml 文件，classname.yaml 文件不存在会自动生成，classname.yaml 文件内容：

```
cat: 0
dog: 1
```

### 3、训练前修改配置文件config/default.yaml

包含权重，数据集路径，学习率，批次大小，图片尺寸，训练次数等。

### 4、下载权重

打开models文件，找到想使用的模型，最下面有提供每个模型的权重路径。

### 5、训练

```
修改 config/default.yaml
scene: train

使用CPU：
python main.py

使用2张GPU卡并行训练：
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py
```

### 6、继续训练

设置想要继续训练的权重，代码会自动加载之前的epoch，学习率、梯队优化算法等。

```
修改 config/default.yaml
scene: train
resume:/home/wenjtop/myProject/imageClassification/output/train/resnet34_exp_0/resnet34_last.pth


使用CPU：
python main.py

使用2张GPU卡并行训练：
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py
```

### 7、验证

```
修改 config/default.yaml
scene: eval
resume:/home/wenjtop/myProject/imageClassification/output/train/resnet34_exp_0/epoch_0_resnet34_67.pth

使用CPU：
python main.py

使用1张GPU卡并行训练：
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py
```

### 8、单图推理

```
修改 inference.py

model_path = 'output/train/mobilenet_v3_s_exp_5/epoch_24_mobilenet_v3_s_90.pth'
imgs_path = '../dataset/catdog/val/dog'

推理：
python inference.py
```

### 9、多图并行推理

```
修改：batch_inference.py

model_path = 'output/train/mobilenet_v3_s_exp_4/epoch_16_mobilenet_v3_s_90.pth'
args.batch_size = 32
# args.batch_size = [224, 224]
args.data_path = '../dataset/catdog/val/dog'

推理：
python batch_inference.py
```