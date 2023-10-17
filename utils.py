# 计算多标签预测准确率
def calculate_accuracy(predictions, targets, threshold=0.5):
    # 将模型的输出值应用阈值，得到二进制标签
    predicted_labels = (predictions > threshold).float()

    # 计算每个样本的准确率
    accuracy = (predicted_labels == targets).float().mean().item()

    return accuracy