# 版权所有 (c) 2024 年由 wenjtop，保留所有权利。
#
# 联系方式: wenjtop@gmail.com

import os
import time
import yaml
import torch
import argparse
import datetime
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn

# 报错就修改为：
from torch.cuda.amp import GradScaler
# from torch.amp import GradScaler
from thop import clever_format, profile
from data.data import Mixup
from data.datasets import build_dataset
from tools import utils
from tools.utils import timing_decorator, init_logger, LogMessage
from tools.engine import train_one_epoch, evaluate
from tools.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from optimizer.scheduler_factory import create_scheduler
from optimizer.optim_factory import param_groups_weight_decay

import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

import models.model as create_model

def get_data(args):
    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    args.log_message('dataset_train:'+str(len(dataset_train)))
    args.log_message('dataset_val:' + str(len(dataset_val)))
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                args.log_message('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)        # 样本按顺序排序，不改变。
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.0 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

def get_flops_params(args):
    model = getattr(create_model, args.model)(num_classes=args.nb_classes)
    dummy_input = torch.rand(1, 3, args.input_size[0], args.input_size[1])
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], '%.3f')

    args.log_message('params:'+ params)
    args.log_message('flops:'+ flops)
    del model

def get_model(args):
    args.log_message(f"Creating model: {args.model}")
    model = getattr(create_model, args.model)(num_classes=args.nb_classes)
    get_flops_params(args)

    args.lr = args.lr * utils.get_world_size()
    args.log_message('lr:' + str(args.lr))

    args.min_lr = args.min_lr * utils.get_world_size()
    args.log_message('min_lr:' + str(args.min_lr))

    args.warmup_lr = args.warmup_lr * utils.get_world_size()
    args.log_message('warmup_lr:' + str(args.warmup_lr))

    model = model.to(args.device)
    model_without_ddp = model
    if args.weight_decay:
        parameters = param_groups_weight_decay(model_without_ddp, args.weight_decay)
        weight_decay = 0.
    else:
        parameters = model_without_ddp.parameters()
        weight_decay = 0.
    optimizer = optim.AdamW(parameters, weight_decay=weight_decay, lr=args.lr)

    loss_scaler = GradScaler()
    lr_scheduler = create_scheduler(args, optimizer)

    if args.resume and os.path.exists(args.resume):
        args.log_message(args.resume+': exist!')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        model_dict = model.state_dict()

        # 只保留可以匹配的键
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        # 更新
        model_dict.update(pretrained_dict)
        # 加载
        model.load_state_dict(model_dict, strict=False)

        # 打印未加载的键以检查哪些层没有被加载
        unmatched_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
        args.log_message("Unmatched keys:"+str(unmatched_keys))

        if args.scene=='train' and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    else:
        args.log_message('resume:' +str(args.resume) + ': not exist!')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.log_message('number of params:'+ str(n_parameters))

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return model, model_without_ddp, lr_scheduler, criterion, optimizer, loss_scaler

@timing_decorator
def main(args):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = get_data(args)
    model, model_without_ddp, lr_scheduler, criterion, optimizer, loss_scaler = get_model(args)

    if args.scene == 'eval':
        test_stats = evaluate(data_loader_val, model, args.device, args)
        args.log_message(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    args.log_message(f"Start training for {args.train_epochs} epochs")
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.train_epochs):
        if epoch < args.augmentation_close_epochs:
            dataset_train.set_transform(augmentation=False, args=args)

        epoch_start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        lr_scheduler.step(epoch+1)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            set_training_mode=True,  # keep in eval mode during finetuning
            args=args
        )
        epoch_end_time = time.time()
        args.log_message(f'Epoch time:{epoch_end_time-epoch_start_time}s')

        test_stats = evaluate(data_loader_val, model, args)
        args.log_message(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
        if utils.is_main_process():
            log_message = args.log_message
            args.log_message = ''
            if max_accuracy < test_stats["acc1"]:
                checkpoint_path = os.path.join(args.current_exp_output, 'epoch_'+str(epoch)+'_'+args.model+'_'+str(int(test_stats["acc1"]))+'.pth')
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'args': args,
                }, checkpoint_path)
            checkpoint_path = os.path.join(args.current_exp_output, args.model+'_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
            args.log_message = log_message
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        args.log_message(f'Max accuracy: {max_accuracy:.2f}%')

        total_time = time.time() - epoch_start_time

        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        remaining_time = (args.train_epochs - epoch) * total_time
        remaining_time_str = str(datetime.timedelta(seconds=int(remaining_time)))
        args.log_message('Training time {}'.format(total_time_str))
        args.log_message('Remaining time {}'.format(remaining_time_str))

def add_config_to_args(args):
    try:
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)
            for sub_key, sub_value in config.items():
                setattr(args, sub_key, sub_value)
        args.lr = float(args.lr)
        args.warmup_lr = float(args.warmup_lr)
        args.min_lr = float(args.min_lr)
        return args
    except FileNotFoundError:
        logging.info(f"配置文件 {file_path} 未找到。")
    except yaml.YAMLError:
        logging.info(f"配置文件 {file_path} 格式错误。")
    except Exception as e:
        logging.info(f"读取配置文件时发生错误：{e}")

def make_file(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.current_exp_output = os.path.join(args.output_dir, args.scene)
    if not os.path.exists(args.current_exp_output):
        os.mkdir(args.current_exp_output)

    file_num = 0
    args.current_exp_output = os.path.join(args.current_exp_output, args.data_path.strip('/').split('/')[-1]+'_'+args.model+'_exp_')
    current_exp_output = args.current_exp_output+str(file_num)
    while os.path.exists(current_exp_output):
        file_num += 1
        current_exp_output = args.current_exp_output + str(file_num)

    args.current_exp_output = current_exp_output
    os.mkdir(args.current_exp_output)

    args.log_path = os.path.join(args.current_exp_output, args.model+'.log')
    args.model + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def exp_init(args):
    utils.init_distributed_mode(args)   # 多显卡并行运算
    if args.distributed:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True   # 消耗额外的时间，找到最适合的卷积算法。

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Images Classification')
    parser.add_argument('--config_path', type=str, default='config/default.yaml', help='config file path')
    args = parser.parse_args()
    add_config_to_args(args)
    exp_init(args)
    if utils.is_main_process():
        make_file(args)
        logger = init_logger(args.model, args.log_path)
    else:
        logger = ''
    args.log_message = LogMessage(logger)
    args.log_message('Args in experiment:')
    args.log_message(args)
    args.log_message('device: ' + str(args.device))
    main(args)
    if args.distributed:
        utils.cleanup()
