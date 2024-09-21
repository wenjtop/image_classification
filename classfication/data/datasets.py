# modify from https://github.com/facebookresearch/deit/blob/main/datasets.py
import os
import yaml
import torch.utils.data
from torchvision import transforms
from PIL import ImageFile, Image

from data.transforms_factory import create_transform
from data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 只加载正确的图片

def build_transform(augmentation, input_size, args=None):
    resize_im = True
    if augmentation:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = [int((256 / 224) * input_size[0]), int((256 / 224) * input_size[1])]
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class build_dataset(torch.utils.data.DataLoader):
    def __init__(self, is_train, args):
        if is_train == 'train':
            augmentation = True
        else:
            augmentation = False
        self.transform = build_transform(augmentation, args.input_size, args)
        directories = []
        for version in os.listdir(args.data_path):
            current_data_path = os.path.join(args.data_path, version, is_train)
            if os.path.isdir(current_data_path):

                directories += [d for d in os.listdir(current_data_path) if os.path.isdir(os.path.join(current_data_path, d))]
        directories = set(directories)

        self.category_dict = self.get_category_dict(is_train, directories, args)
        self.images_labels = []
        for version in os.listdir(args.data_path):
            current_data_path = os.path.join(args.data_path, version, is_train)
            if os.path.isdir(current_data_path):
                for dir in directories:
                    current_category_path = os.path.join(current_data_path, dir)
                    if os.listdir(current_category_path):
                        self.images_labels += [(os.path.join(current_category_path, img), self.category_dict[dir]) for img in os.listdir(current_category_path)]
    def set_transform(self, augmentation, args):
        self.transform = build_transform(augmentation, args.input_size, args)
    def __len__(self):
        return len(self.images_labels)
    def get_category_dict(self, is_train, directories, args):
        classname_path = os.path.join(args.data_path, 'class_name.yaml')
        if is_train == 'train' and not os.path.exists(classname_path):
            category_dict = {category: index for index, category in enumerate(directories)}
            with open(classname_path, 'w') as file:
                yaml.dump(category_dict, file)

        if os.path.exists(classname_path):
            with open(classname_path, 'r') as file:
                category_dict = yaml.safe_load(file)
        args.category_dict = category_dict
        return category_dict

    def __getitem__(self, item):
        img_path, label = self.images_labels[item]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, label









