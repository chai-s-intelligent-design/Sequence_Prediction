import argparse
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from dataset import SequenceDataset
from dataset import original_dir
from LSTMModel import LSTMModel
from utils import calculate_accuracy, MetricLogger

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
                    default=0.005,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=0.,
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


def main():
    args = parser.parse_args()
    print("\n".join("%s: %s" % (k, str(v))
                    for k, v in sorted(dict(vars(args)).items())))
    assert args.ckpt_dir is not None
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    cudnn.benchmark = True
    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    # 定义数据集
    num_classes = 51
    class_num = 0
    max_time = 60
    time_embedding = 10
    look_back = 1
    include_time = True

    dataset = SequenceDataset(original_dir, num_classes, class_num, max_time,
                              time_embedding, look_back, include_time)
    # 数据集分割
    train_dataset = dataset.get_subset(0, int(len(dataset) * 0.8))
    val_dataset = dataset.get_subset(int(len(dataset) * 0.8), len(dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # 定义模型
    input_size = look_back * num_classes + time_embedding if include_time else look_back * num_classes
    model = LSTMModel(input_size, 2, 1, num_classes)
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

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, lr_scheduler, epoch + 1, device,
              args)

        # evaluate on validation set
        validate(val_loader, model, device, args)

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
        loss = model.loss_function(results[0], y)
        accuracy = calculate_accuracy(results[0], y)
        metric_logger.update(loss=loss, accuracy=accuracy, n=x.shape[2])

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()


def validate(val_loader, model, device, args):
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
            loss = model.loss_function(results[0], y)
            accuracy = calculate_accuracy(results[0], y)
            metric_logger.update(loss=loss, accuracy=accuracy, n=x.shape[2])

    print(f"Acc: {metric_logger.accuracy.global_avg}")


if __name__ == '__main__':
    main()
