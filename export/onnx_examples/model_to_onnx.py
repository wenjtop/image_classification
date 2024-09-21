import sys
import torch

sys.path.append('../../classfication')
import models.model as create_model

model_path = '../output/train/catdog_resnet34_exp_1/epoch_1_resnet34_100.pth'

weights_args_dict = torch.load(model_path, map_location='cpu', weights_only=False)
weights_dict = weights_args_dict['model']
args = weights_args_dict['args']
model = getattr(create_model, args.model)(num_classes=args.nb_classes)
load_weights_dict = {k: v for k, v in weights_dict.items()
                     if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}


model.load_state_dict(load_weights_dict)

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "catdog_resnet34.onnx",
        opset_version=16,
        input_names=['input'],
        output_names=['output'])