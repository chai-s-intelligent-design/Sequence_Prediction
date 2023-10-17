import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 创建自定义数据集类
class SequenceDataset(Dataset):

    def __init__(self, X, y, time_embedding):
        self.X = X
        self.y = y
        self.embedding = nn.Embedding(X.shape[1], time_embedding)  # 创建嵌入层

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取原始时间序列
        original_sequence = self.X[idx]

        # 执行时间序列嵌入（embedding）
        embedded_sequence = self.embedding(original_sequence)

        # 返回嵌入后的时间序列和对应的标签
        return embedded_sequence, self.y[idx]