import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_classes)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() != 3:
            x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        out, (hidden_state, cell_state) = self.lstm(x, (h0, c0))

        outputs = []
        for i in range(out.size(1)):
            output = []
            for j in range(len(self.fc)):
                output.append(self.fc[j](out[:, i, :]))
            outputs.append(torch.cat(output, dim=1))
        outputs = torch.stack(outputs, dim=1)
        outputs = self.sigmoid(outputs)
        return outputs, (hidden_state, cell_state)

    def focal_loss(self, output, input, alpha, gamma):
        ce_loss = F.binary_cross_entropy(output,
                                         input.float(),
                                         reduction='none')  # 计算二元交叉熵损失，不进行求和
        at = torch.where(input == 1, alpha, 1 - alpha)
        pt = torch.exp(-ce_loss)  # 计算易分类样本的权重（概率）
        focal_loss = at * (1 - pt)**gamma * ce_loss  # 计算Focal Loss
        return focal_loss.mean()  # 求Focal Loss的平均值作为最终损失

    def loss_function(self, *args):
        output = args[0]
        input = args[1]
        alpha = args[2]
        gamma = args[3]
        loss = self.focal_loss(output, input, alpha, gamma)
        return loss
