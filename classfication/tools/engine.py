"""
Train and eval functions used in main.py
"""
import math
import sys
import torch
# from torch.amp import autocast
from torch.cuda.amp import autocast
from tools.metrics import accuracy
from tools.utils import Metriclog_message, SmoothedValue, timing_decorator

@timing_decorator
def train_one_epoch(model, criterion, data_loader, optimizer, epoch, loss_scaler, max_norm = 0,
                    mixup_fn = None, set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_log_message = Metriclog_message(delimiter="  ")
    metric_log_message.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_log_message.log_every(data_loader, print_freq, header, args):
        samples = samples.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        if mixup_fn is not None and epoch < args.mixup_close_epochs:
            samples, targets = mixup_fn(samples, targets)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        # 报错就修改为：
        # with autocast('cuda'):
        with autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            args.log_message("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        scaled_loss = loss_scaler.scale(loss)
        scaled_loss.backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        if args.device == 'cuda':
            torch.cuda.synchronize()

        metric_log_message.update(loss=loss_value)
        metric_log_message.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_log_message.synchronize_between_processes()
    args.log_message("Averaged stats:"+ str(metric_log_message))
    return {k: meter.global_avg for k, meter in metric_log_message.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_log_message = Metriclog_message(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    # switch to evaluation mode
    model.eval()

    for images, target in metric_log_message.log_every(data_loader, print_freq, header, args):
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        # compute output
        # 报错就修改为：
        # with autocast('cuda'):
        with autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_log_message.update(loss=loss.item())
        metric_log_message.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_log_message.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_log_message.synchronize_between_processes()
    args.log_message('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_log_message.acc1, top5=metric_log_message.acc5, losses=metric_log_message.loss))

    return {k: meter.global_avg for k, meter in metric_log_message.meters.items()}