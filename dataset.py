import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np

root_dir = "./datasets"
original_dir = root_dir + "/original"


# 创建自定义数据集类
class SequenceDataset(Dataset):

    def __init__(self,
                 root,
                 num_classes,
                 class_num,
                 max_time,
                 time_embedding,
                 look_back=1,
                 include_time=True):
        self.root = root
        self.root_dir = os.path.join(root, str(class_num))
        self.num_classes = num_classes
        self.class_num = class_num
        self.max_time = max_time
        self.time_embedding = time_embedding
        self.look_back = look_back
        self.include_time = include_time
        torch.manual_seed(42)
        self.embedding = nn.Embedding(max_time, time_embedding)  # 创建嵌入层
        self.file_list = self.get_csv_files()

    def __len__(self):
        return len(self.file_list)

    def get_csv_files(self):
        file_list = []
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith(".csv"):
                file_list.append(file_name)
        return file_list

    def __getitem__(self, idx):
        # 获取原始时间序列
        original_sequence = pd.read_csv(
            os.path.join(self.root_dir,
                         str(idx) + ".csv"))

        # 提取时指示向量列
        indicator_vectors = original_sequence.drop(columns=["index"]).values

        # 构建 x 和 y
        y = indicator_vectors  # (seq_len, num_classes)
        x = np.array([]).reshape(0, self.look_back * self.num_classes)
        for i in range(len(y)):
            if i < self.look_back:
                delta = self.look_back - i
                dummy_y = np.zeros(delta * self.num_classes)
                true_y = indicator_vectors[i - self.look_back +
                                           delta:i].flatten()
                x = np.concatenate(
                    (x, np.concatenate([dummy_y, true_y], 0).reshape(1, -1)),
                    0)
            else:
                x = np.concatenate([
                    x,
                    indicator_vectors[i - self.look_back:i].flatten().reshape(
                        1, -1)
                ], 0)  # x 是之前 look_back 个时刻的指示向量组成的序列
        # 将 x,y 转换为指示向量的数组
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # 是否增加时间向量
        if self.include_time:
            time_features = self.embedding(torch.arange(y.shape[0]))
            x = torch.cat([x, time_features], dim=1)

        # 返回数据对象
        return x.to(torch.float), y

    def get_subset(self, start, end):
        subset_files = self.file_list[start:end]
        subset = SequenceDataset(self.root, self.num_classes, self.class_num,
                                 self.max_time, self.time_embedding)
        subset.file_list = subset_files
        return subset


if __name__ == "__main__":
    dataset = SequenceDataset(original_dir, 51, 0, 60, 10, 2, True)

    # 每个人的序列长度不一样，只能设置batch_size为1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 迭代获取每个批次的数据对象
    for batch in dataloader:
        x, y = batch
        print(x)
