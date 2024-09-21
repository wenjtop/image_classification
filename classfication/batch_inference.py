import os
import time
import torch
import models.model as create_model

from PIL import Image
from torchvision import transforms
from tools.utils import timing_decorator
from data.datasets import build_transform, build_dataset
from data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_transform(args=None):
    t = []
    t.append(transforms.Resize(args.input_size, interpolation=3))  # to maintain same ratio w.r.t. 224 images
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class build_dataset(torch.utils.data.DataLoader):
    def __init__(self, args):
        self.transform = build_transform(args)
        self.images_path = [os.path.join(args.data_path, img) for img in os.listdir(args.data_path)]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img

def get_model_args(model_path, device):
    weights_args_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    weights_dict = weights_args_dict['model']
    args = weights_args_dict['args']
    model = getattr(create_model, args.model)(num_classes=args.nb_classes)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}

    model.load_state_dict(load_weights_dict)
    model = model.to(device)
    args.device = device
    return model, args

@timing_decorator
def inference(model, args):
    dataset_inference = build_dataset(args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_inference,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    with torch.no_grad():
        for images in data_loader_val:
            start_time = time.time()
            images = images.to(args.device, non_blocking=True)
            outputs = model(images)
            result = torch.softmax(outputs, dim=1).cpu().numpy()
            cls = result.argmax(axis=1)
            print(result)
            print(cls)
            end_time = time.time()
            print(print(f"耗时: {end_time-start_time:.6f} 秒"))

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_path = '../output/train/catdog_mobilenet_v3_s_exp_3/epoch_5_mobilenet_v3_s_97.pth'
    model, args = get_model_args(model_path=model_path, device=device)

    args.batch_size = 32
    # args.batch_size = [224, 224]
    args.data_path = '../datasets/catdog/v1/val/dog'
    inference(model, args)





