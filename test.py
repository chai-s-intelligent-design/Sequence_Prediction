import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 生成示例数据，包含不同长度的序列
data = [
    [1, 2, 3, 4],
    [5, 6],
    [7, 8, 9],
    [10, 11, 12, 13, 14]
]

# 将数据转换为PyTorch张量
sequences = [torch.tensor(seq, dtype=torch.float32) for seq in data]

# 使用torch.nn.utils.rnn.pad_sequence填充序列
padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)

# 构建数据集和数据加载器
labels = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

dataset = TensorDataset(padded_sequences, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 创建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out

# 初始化模型
input_size = 1  # 输入序列的特征维度
hidden_size = 32  # LSTM隐藏层的大小
num_layers = 1  # LSTM层的数量
output_size = 1  # 输出的大小

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_sequence = torch.tensor([[15, 16, 17, 18, 19]], dtype=torch.float32)  # 变长输入序列
    predicted_value = model(test_sequence)
    print(f'Predicted Value: {predicted_value.item()}')
