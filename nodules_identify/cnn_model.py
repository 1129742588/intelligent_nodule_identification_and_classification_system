import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    """
    一个简单的3D卷积神经网络，用于处理32×32×32的3D图像数据。
    - 参数:
        - in_channels: 输入通道数，默认为1（例如灰度图像）
        - num_classes: 分类类别数，默认为2
        - patch_size: 输入图像的尺寸，默认为32（默认输入为32×32×32的3D图像）
    """

    def __init__(self, in_channels=1, num_classes=2, patch_size=32):
        """
        初始化3D卷积神经网络的各层结构。
        - 参数：
            - in_channels: 输入通道数，默认值为1，适用于单通道CT区块。
            - num_classes: 输出类别数，当前任务中通常为2，表示结节和非结节。
            - patch_size: 输入区块边长，默认值为32，对应32×32×32的立方体区块。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 定义三组卷积、归一化与池化层，逐步提取3D空间特征。
            - 2. 根据输入区块尺寸确定全连接层输入维度。
            - 3. 定义全连接层和Dropout层，用于最终分类。
        """
        super(Simple3DCNN, self).__init__()
        
        # 第一组：卷积 + 池化
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)  # 输出尺寸: 16×16×16
        
        # 第二组
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)  # 输出尺寸: 8×8×8
        
        # 第三组
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)  # 输出尺寸: 4×4×4
        
        # 第四组（可选，如果patch_size更大可以再加）
        # self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm3d(128)
        # self.pool4 = nn.MaxPool3d(2)  # 输出尺寸: 2×2×2
        
        # 计算全连接层输入特征数
        # 经过三次池化后，尺寸为 patch_size/8 = 4，所以特征图大小 4×4×4，通道64
        self.fc_input_size = 64 * 4 * 4 * 4
        
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        定义模型的前向传播过程，输出每个类别的预测分数。
        - 参数：
            - x: 输入张量，形状通常为(batch, 1, 32, 32, 32)。
        - 返回：
            - x: 输出张量，形状为(batch, num_classes)，表示每个类别的logits。
        - 流程:
            - 1. 依次通过三组卷积、归一化、激活和池化层提取特征。
            - 2. 将3D特征图展平为一维向量。
            - 3. 经过全连接层和Dropout后输出分类结果。
        """
        # x shape: (batch, 1, 32, 32, 32)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # x shape: (batch, 64, 4, 4, 4)
        
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x