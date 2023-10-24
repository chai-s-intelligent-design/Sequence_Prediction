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

parser = argparse.ArgumentParser(description='Time Series Generation')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu',
                    default=0,
                    type=int,
                    metavar='N',
                    help='which gpu device to use')


def main():
    args = parser.parse_args()
    print("\n".join("%s: %s" % (k, str(v))
                    for k, v in sorted(dict(vars(args)).items())))

    cudnn.benchmark = True
    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    num_classes = 51
    max_time = 60
    time_embedding = 10
    look_back = 1
    include_time = True
    input_size = look_back * num_classes + time_embedding if include_time else look_back * num_classes
    model = LSTMModel(input_size, 128, 1, num_classes)
    print("=> loading checkpoint '{}'".format(args.resume))
    if args.gpu is None:
        checkpoint = torch.load(args.resume)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    generate(model, device, 15, num_classes, look_back, include_time,
             time_embedding, max_time)


def generate(model, device, sequence_length, num_classes, look_back,
             inclue_time, time_embedding, max_time):

    # switch to evaluate mode
    model.eval()
    model.to(device)

    initial_feature = torch.zeros(
        look_back * num_classes,
        device=device)  # 初始输入特征的张量，形状为(1, sequence_length, input_size)

    if inclue_time:
        torch.manual_seed(42)
        time_embedding = nn.Embedding(max_time, time_embedding)  # 创建嵌入层
        time_features = time_embedding(
            torch.arange(sequence_length)).to(device)
        initial_feature = torch.cat([initial_feature, time_features[0]], dim=0)

    # 初始化预测序列的列表
    predicted_X = initial_feature.unsqueeze(0).to(device)
    predicted_Y = torch.tensor([]).reshape(0, num_classes).to(device)
    # 逐步生成序列
    with torch.no_grad():
        for step in range(sequence_length - 1):
            # 使用模型生成下一个时刻的预测
            output, _ = model(predicted_X)
            print(f"step {step}:, {(output[0,-1] > 0.5).float()}")
            predicted_Y = torch.cat(
                [predicted_Y, ((output[0, -1] > 0.5).float()).unsqueeze(0)], 0)
            if step > look_back:
                next_feature = predicted_Y[step - look_back:step].reshape(
                    1, -1)
            else:
                delta = look_back - step
                dummy_y = torch.zeros(delta * num_classes).to(device)
                true_y = predicted_Y[step - look_back + delta:step].flatten()
                next_feature = torch.cat([dummy_y, true_y], 0).reshape(1, -1)
            if inclue_time:
                next_feature = torch.cat(
                    [next_feature, time_features[step].unsqueeze(0)], 1)
            # 将下一个时刻的特征添加到预测序列中
            predicted_X = torch.cat([predicted_X, next_feature], 0)
    # predicted_Y中包含了逐步生成的序列
    print(predicted_Y)


if __name__ == '__main__':
    main()
