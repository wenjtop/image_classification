import os
import time
import torch
import models.model as create_model

from PIL import Image
from data.datasets import build_transform
from tools.utils import timing_decorator

def get_model_args(model_path, device):
    weights_args_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    weights_dict = weights_args_dict['model']
    args = weights_args_dict['args']
    model = getattr(create_model, args.model)(num_classes=args.nb_classes)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}


    model.load_state_dict(load_weights_dict)
    model = model.to(device)
    return model, args

@timing_decorator
def inference(model, imgs_path, device, args):
    transform = build_transform(augmentation=False, input_size=args.input_size)
    imgs = os.listdir(imgs_path)
    with torch.no_grad():
        for img_name in imgs:
            start_time = time.time()
            img_path = os.path.join(imgs_path, img_name)
            img = Image.open(img_path)
            img = transform(img)
            img = img.unsqueeze(dim=0).to(device)
            output = model(img)[0]
            result = torch.softmax(output, dim=0).cpu().numpy()
            cls = result.argmax(axis=0)
            print(result)
            print(cls)
            end_time = time.time()
            print(print(f"耗时: {end_time-start_time:.6f} 秒"))

# ['lng_n', 'lng_t', 'lng_s', 'lng_b',
#  'resnet34', 'resnet_50', 'resnet_101', 'resnext50_32x4d', 'resnext101_32x8d'
#  'vit_b', 'vit_l', input 224x224 7的倍数
#  'swin_t', 'swin_s', 'swin_b', input 224x224 7的倍数
#  'conv_next_t', 'conv_next_s', 'conv_next_b', 'conv_next_l',
#  'effNet_b1', 'effNet_b2', 'effNet_b3', 'effNet_b4', 'effNet_b5', 'effNet_b6', 'effNet_b7',
#  'effnet_v2_s', 'effNet_v2_m', 'effnet_v2_l',
#  'mobilenet_v2', 'mobilenet_v3_s', 'mobilenet_v3_l']

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_path = '../output/train/catdog_mobilenet_v3_s_exp_5/epoch_5_mobilenet_v3_s_97.pth'
    imgs_path = '../datasets/catdog/v1/val/dog'
    model, args = get_model_args(model_path=model_path, device=device)
    inference(model, imgs_path, device, args)





