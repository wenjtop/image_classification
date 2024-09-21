from .lng_transformer_allimg import LNGTransformer
from .swin_transformer import SwinTransformer
from .resnet import ResNet, BasicBlock, Bottleneck
from .vision_transformer import VisionTransformer
from .conv_next import ConvNeXt
from .efficient_net import EfficientNet
from .efficient_net_v2 import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small

def lng_n(num_classes=100, config=[1, 1, 2, 1], dim=64, **kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=[8, 4], stages=config, num_heads=[2, 4, 8, 16], num_classes=num_classes)
# LSG_Transformer
def lng_t(num_classes=100, stages=[1, 1, 2, 1], dim=96, num_heads=[3, 6, 12, 24],**kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=7, stages = stages, num_heads = num_heads, num_classes=num_classes)

def lng_s(num_classes=100, stages=[1, 1, 2, 1], dim=128, num_heads=[4, 8, 16, 32],**kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=7, stages = stages, num_heads = num_heads, num_classes=num_classes)

def lng_b(num_classes=100, stages=[1, 1, 6, 1], dim=128, num_heads=[4, 8, 16, 32],**kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=7, stages = stages, num_heads = num_heads, num_classes=num_classes)

# resnet
def resnet34(num_classes=1000, include_top=True, **kwargs):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet_50(num_classes=1000, include_top=True, **kwargs):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet_101(num_classes=1000, include_top=True, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnext50_32x4d(num_classes=1000, include_top=True, **kwargs):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True, **kwargs):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

# vision transformer
def vit_b(num_classes: int = 1000, **kwargs):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None, num_classes=num_classes)
    return model

def vit_l(num_classes: int = 1000, **kwargs):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=None, num_classes=num_classes)
    return model

def swin_t(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), num_classes=num_classes, **kwargs)
    return model

def swin_s(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), num_classes=num_classes, **kwargs)
    return model

def swin_b(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=num_classes, **kwargs)
    return model
# ConvNeXt
def conv_next_t(num_classes: int, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes)
    return model


def conv_next_s(num_classes: int, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=num_classes)
    return model


def conv_next_b(num_classes: int, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes)
    return model


def conv_next_l(num_classes: int, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], num_classes=num_classes)
    return model

# EfficientNet
def effNet_b1(num_classes=1000, **kwargs):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2, num_classes=num_classes)


def effNet_b2(num_classes=1000, **kwargs):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.3, num_classes=num_classes)


def effNet_b3(num_classes=1000, **kwargs):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2, depth_coefficient=1.4, dropout_rate=0.3, num_classes=num_classes)


def effNet_b4(num_classes=1000, **kwargs):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4, depth_coefficient=1.8, dropout_rate=0.4, num_classes=num_classes)


def effNet_b5(num_classes=1000, **kwargs):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6, depth_coefficient=2.2, dropout_rate=0.4, num_classes=num_classes)


def effNet_b6(num_classes=1000, **kwargs):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8, depth_coefficient=2.6, dropout_rate=0.5, num_classes=num_classes)


def effNet_b7(num_classes=1000, **kwargs):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.5, num_classes=num_classes)


# EfficientNet V2
def effnet_v2_s(num_classes=1000, **kwargs):
    return efficientnetv2_s(num_classes=num_classes)

def effNet_v2_m(num_classes=1000, **kwargs):
    return efficientnetv2_m(num_classes=num_classes)

def effnet_v2_l(num_classes=1000, **kwargs):
    return efficientnetv2_l(num_classes=num_classes)

def mobilenet_v2(num_classes=1000, **kwargs):
    return MobileNetV2(num_classes=num_classes)

def mobilenet_v3_s(num_classes=1000, **kwargs):
    return mobilenet_v3_small(num_classes=num_classes)

def mobilenet_v3_l(num_classes=1000, **kwargs):
    return mobilenet_v3_large(num_classes=num_classes)

