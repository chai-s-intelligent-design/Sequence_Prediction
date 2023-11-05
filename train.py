import argparse
import os
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from dataset import SequenceDataset
from dataset import original_dir
from LSTMModel_1 import LSTMModel
from utils import calculate_accuracy, MetricLogger, calculate_metrics
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Time Series Prediction')
parser.add_argument('-d',
                    '--data',
                    metavar='DIR',
                    default="./datasets",
                    help='path to dataset')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=5e-4,
                    type=float,
                    metavar='WD',
                    help='weight decay rate',
                    dest='wd')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    metavar='N',
                    help='which gpu device to use')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of training epochs')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-c',
                    '--ckpt_dir',
                    default="./ckpt_dir",
                    type=str,
                    metavar='PATH',
                    help='path to store checkpoint')
# 模型控制参数
parser.add_argument('--num-classes',
                    default=51,
                    type=int,
                    help='Number of classes')
parser.add_argument('--class-num', default=0, type=int, help='Class number')
parser.add_argument('--max-time', default=60, type=int, help='Maximum time')
parser.add_argument('--time-embedding',
                    default=128,
                    type=int,
                    help='Time embedding')
parser.add_argument('--look-back', default=3, type=int, help='Look back')
parser.add_argument('--include-time', action='store_true', help='Include time')
parser.add_argument('--hidden-dim',
                    default=128,
                    type=int,
                    help='Hidden dimension')
parser.add_argument('--num-layers',
                    default=1,
                    type=int,
                    help='Number of layers')
parser.add_argument('--alpha',
                    default=1 - 1 / 51,
                    type=float,
                    help='Balance factor')
parser.add_argument('--gamma', default=2, type=float, help='Focus factor')
parser.add_argument('--max-selected',
                    default=4,
                    type=int,
                    help='Maximum selected')
time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    args = parser.parse_args()
    assert args.ckpt_dir is not None
    args.ckpt_dir = args.ckpt_dir + "/" + str(args.class_num) + "/" + time_str
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print("\n".join("%s: %s" % (k, str(v))
                    for k, v in sorted(dict(vars(args)).items())))
    with open(os.path.join(args.ckpt_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    cudnn.benchmark = True
    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    # 定义超参数
    # 1. 模型参数
    num_classes = args.num_classes
    class_num = args.class_num
    max_time = args.max_time
    time_embedding = args.time_embedding
    look_back = args.look_back
    include_time = args.include_time
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    # 2. 损失函数
    global alpha  # 平衡因子，用于调整正负样本的权重
    global gamma  # 聚焦因子，用于调整难分类样本的权重
    global max_selected  # 最大选取数目，用于控制预测时同时发生的消费行为的最大数目
    alpha = args.alpha
    gamma = args.gamma
    max_selected = args.max_selected

    # 创建数据集
    dataset = SequenceDataset(original_dir, num_classes, class_num, max_time,
                              time_embedding, look_back, include_time)
    # 数据集分割
    train_dataset = dataset.get_subset(0, int(len(dataset) * 0.8))
    val_dataset = dataset.get_subset(int(len(dataset) * 0.8), len(dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 定义模型
    input_size = look_back * num_classes + time_embedding if include_time else look_back * num_classes
    model = LSTMModel(input_size, hidden_dim, num_layers, num_classes)
    model = model.to(device)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), eta_min=1e-6)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    log_dir = "log/" + str(class_num) + "/" + time_str
    global writer
    writer = SummaryWriter(log_dir)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, lr_scheduler, epoch + 1, device,
              args)

        # evaluate on validation set
        validate(val_loader, model, device, epoch + 1, args)

        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, os.path.join(args.ckpt_dir, "checkpoint.pt"))


def train(train_loader, model, optimizer, lr_scheduler, epoch, device, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for i, data in enumerate(
            metric_logger.log_every(train_loader,
                                    args.print_freq,
                                    header=header)):
        x, y = data
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # compute output
        results = model(x)
        loss = model.loss_function(results[0], y, alpha, gamma)
        accuracy, recall, precision, f1 = calculate_metrics(
            results[0], y, 0.5, max_selected)
        metric_logger.update(loss=loss,
                             accuracy=accuracy,
                             recall=recall,
                             precision=precision,
                             f1=f1,
                             n=x.shape[1] * x.shape[2])

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

    # Record metrics in TensorBoard
    writer.add_scalar('Train/Loss',
                      metric_logger.loss.global_avg,
                      global_step=epoch)
    writer.add_scalar('Train/Accuracy',
                      metric_logger.accuracy.global_avg,
                      global_step=epoch)
    writer.add_scalar('Train/Recall',
                      metric_logger.recall.global_avg,
                      global_step=epoch)
    writer.add_scalar('Train/Precision',
                      metric_logger.precision.global_avg,
                      global_step=epoch)
    writer.add_scalar('Train/F1',
                      metric_logger.f1.global_avg,
                      global_step=epoch)

    print(f"Acc: {metric_logger.accuracy.global_avg}")


def validate(val_loader, model, device, epoch, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = "validate"

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(
                metric_logger.log_every(val_loader,
                                        args.print_freq,
                                        header=header)):
            x, y = data
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # compute output
            results = model(x)
            loss = model.loss_function(results[0], y, alpha, gamma)
            accuracy, recall, precision, f1 = calculate_metrics(
                results[0], y, 0.5, max_selected)
            metric_logger.update(loss=loss,
                                 accuracy=accuracy,
                                 recall=recall,
                                 precision=precision,
                                 f1=f1,
                                 n=x.shape[1] * x.shape[2])
    # Record metrics in TensorBoard
    writer.add_scalar('Validation/Loss',
                      metric_logger.loss.global_avg,
                      global_step=epoch)
    writer.add_scalar('Validation/Accuracy',
                      metric_logger.accuracy.global_avg,
                      global_step=epoch)
    writer.add_scalar('Validation/Recall',
                      metric_logger.recall.global_avg,
                      global_step=epoch)
    writer.add_scalar('Validation/Precision',
                      metric_logger.precision.global_avg,
                      global_step=epoch)
    writer.add_scalar('Validation/F1',
                      metric_logger.f1.global_avg,
                      global_step=epoch)

    print(f"Acc: {metric_logger.accuracy.global_avg}")


if __name__ == '__main__':
    main()
