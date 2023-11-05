from collections import defaultdict, deque
from functools import partial
import time
import datetime
import os
import random

from sklearn.metrics import confusion_matrix
import torch


def indicate_vectors_to_codes(indicate_vectors):
    codemap_file_path = os.path.join('./datasets', "codemap_file.txt")
    # 从 codemap_file.txt 读取编码映射
    codemap = {}  # 用于存储编码映射的字典
    with open(codemap_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                code, label = parts[0], parts[1]
                codemap[code] = label
    # 使用编码映射将 0/1 向量转换为对应的编码
    codes = []
    for vector in indicate_vectors:
        code_list = [
            codemap.get(str(i), "") for i, indicator in enumerate(vector)
            if indicator == 1
        ]
        codes.append(code_list)

    return codes


def adjust_predictions(predictions,
                       max_selected=4,
                       threshold=0.5,
                       add_random=False):
    # 创建一个张量以保存处理后的标签
    adjusted_labels = torch.zeros_like(predictions)

    # 计算每个样本的>0.5的个数
    count_gt_threshold = torch.sum(predictions > threshold, dim=2)

    # 对每个样本进行处理
    for i in range(predictions.size(1)):
        if count_gt_threshold[0, i] >= max_selected:
            if not add_random:
                # 找到前4个最大概率的位置
                top_indices = torch.topk(predictions[0, i, :],
                                         k=max_selected,
                                         largest=True).indices
                # 设置前4个最大概率对应位置为1
                adjusted_labels[0, i, top_indices] = 1
            else:
                top_indices = torch.topk(predictions[0, i, :],
                                         k=max_selected - 1,
                                         largest=True).indices
                adjusted_labels[0, i, top_indices] = 1
                # 获取所有预测值的位置
                all_indices = torch.arange(predictions.shape[2]).to(
                    predictions.device)
                probs = (predictions[0, i, :] / predictions[0, i, :].sum())
                torch.seed()
                # 依据概率从所有位置中随机选择一个
                random_index = all_indices[torch.multinomial(probs, 1)]
                # 确保随机选择的值不在前面已经选好的值中
                while random_index in top_indices:
                    random_index = all_indices[torch.multinomial(probs, 1)]
                adjusted_labels[0, i, random_index] = 1
                torch.manual_seed(42)  # 42 之前设置的种子值，用于重新启用特定的种子
        elif count_gt_threshold[0, i] == 0:
            # 找到最大概率的位置
            max_index = torch.argmax(predictions[0, i, :])
            # 设置最大概率对应位置为1
            adjusted_labels[0, i, max_index] = 1
        else:
            adjusted_labels = predictions > threshold

    return adjusted_labels.float()


def calculate_accuracy(predictions, targets, threshold=0.5, max_selected=2):
    # 调用 adjust_predictions 处理预测标签
    adjusted_labels = adjust_predictions(predictions, max_selected)

    # 计算准确率
    accuracy = (adjusted_labels == targets).float().mean().item()

    return accuracy


def calculate_metrics(predictions, targets, threshold=0.5, max_selected=4):
    # 调用 adjust_predictions 处理预测标签
    adjusted_labels = adjust_predictions(predictions, max_selected)

    # 计算混淆矩阵
    cm = confusion_matrix(
        targets.view(-1).cpu().numpy(),
        adjusted_labels.view(-1).cpu().numpy())

    # 计算准确率
    accuracy = (adjusted_labels == targets).float().mean().item()

    # 计算查全率（召回率）, 表示模型成功检测到正例样本的能力，即真正例占所有正例样本的比例。
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])

    # 计算查准率（精确率）, 表示模型在预测正例时的准确性，即真正例占所有被模型预测为正例的样本的比例。
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1


def calculate_metrics_without_adjust(adjusted_labels,
                                     targets,
                                     threshold=0.5,
                                     max_selected=2):

    # 计算混淆矩阵
    cm = confusion_matrix(
        targets.view(-1).cpu().numpy(),
        adjusted_labels.view(-1).cpu().numpy())

    # 计算准确率
    accuracy = (adjusted_labels == targets).float().mean().item()

    # 计算查全率（召回率）, 表示模型成功检测到正例样本的能力，即真正例占所有正例样本的比例。
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])

    # 计算查准率（精确率）, 表示模型在预测正例时的准确性，即真正例占所有被模型预测为正例的样本的比例。
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=None, fmt=None):
        if fmt is None:
            fmt = "{value:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    @property
    def all(self):
        return list(self.deque)

    def __str__(self):
        return self.fmt.format(median=self.median,
                               avg=self.avg,
                               global_avg=self.global_avg,
                               max=self.max,
                               value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            # 定义一个生成器函数
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(
                        log_msg.format(i,
                                       len(iterable),
                                       eta=eta_string,
                                       meters=str(self)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
