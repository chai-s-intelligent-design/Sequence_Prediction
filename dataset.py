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

    def __init__(self, root, class_num, max_time, time_embedding):
        self.root_dir = os.path.join(root, str(class_num))
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
        y = indicator_vectors
        x = np.arange(len(y))

        # 将 y 转换为指示向量的数组
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        # 返回数据对象
        return self.embedding(x), y


if __name__ == "__main__":
    dataset = SequenceDataset(original_dir, 0, 60, 10)

    # 每个人的序列长度不一样，只能设置batch_size为1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 迭代获取每个批次的数据对象
    for batch in dataloader:
        x, y = batch
        print(x)