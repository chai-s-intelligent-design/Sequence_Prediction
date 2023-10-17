import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import SequenceDataset
from LSTMModel import LSTMModel
from utils import calculate_accuracy
# 数据准备
num_samples = 1000
sequence_length = 10
num_classes = 51
batch_size = 32
time_embedding = 8
max_ones = 10
# 生成随机的张量，每个样本中最多有四个1
y = torch.zeros(num_samples, sequence_length, num_classes, dtype=torch.float32)

for i in range(num_samples):
    for j in range(sequence_length):
        # 随机选择四个位置，设置为1
        ones_indices = torch.randperm(num_classes)[:max_ones]
        y[i, j, ones_indices] = 1

X = torch.arange(sequence_length).repeat(num_samples, 1)

dataset = SequenceDataset(X, y, time_embedding)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
input_size = time_embedding
hidden_size = 128
num_layers = 4
num_epochs = 10000

model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(num_epochs):
    total_accuracy = 0.0  # 用于累积每个 epoch 的总准确率
    total_loss = 0.0  # 用于累积每个 epoch 的总损失
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)

        # 在这里，您需要将模型的输出与目标比较，并计算损失
        # 对于每个时间步，您可以使用交叉熵损失等
        # 这个示例中，假设每个时间步的预测结果是分类问题
        # 您需要根据实际问题调整损失计算
        loss = criterion(outputs.view(-1, num_classes),
                         targets.view(-1, num_classes))
        loss.backward()
        optimizer.step()

        # 计算每个 batch 的准确率并累积
        batch_accuracy = calculate_accuracy(outputs.view(-1, num_classes),
                                            targets.view(-1, num_classes))
        total_accuracy += batch_accuracy
        total_loss += loss.item()

    # 计算每个 epoch 的平均准确率和损失
    average_accuracy = total_accuracy / len(dataloader)
    average_loss = total_loss / len(dataloader)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}'
    )

print('Training finished!')
