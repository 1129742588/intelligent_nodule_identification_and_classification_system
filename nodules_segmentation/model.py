"""
预测工具模块 - 独立于训练代码的模型预测功能
包含：模型定义、加载、预测等函数

注意：此文件中的模型代码与训练代码完全一致，确保权重可以正常加载
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ 模型定义（与训练代码 _3model.py 完全一致） ============

class Swish(nn.Module):
    """带可学习参数β的Swish激活函数"""
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class ChannelAttentionBlock(nn.Module):
    """通道注意力模块(CAB)，增强有用特征通道"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class LargeKernelGroupAttentionGate(nn.Module):
    """大核分组注意力门控(LGAG)"""
    def __init__(self, in_channels, g_channels, groups=4, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        # 1. 门控信号通道对齐：1×1×1 3D卷积，仅调整通道数
        self.gate_channel_adjust = nn.Conv3d(g_channels, in_channels, 1, bias=False)

        # 2. 大核分组卷积：生成注意力权重
        self.group_conv = nn.Conv3d(in_channels, in_channels, kernel_size,
                                    padding=kernel_size//2, groups=groups, bias=False)
        self.bn_attn = nn.BatchNorm3d(in_channels)

        # 3. 输出卷积块(Conv Block)：通道变换
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """
        g: 解码器深层特征（门控信号）
        x: 编码器浅层特征
        """
        # 步骤1: 门控信号通道对齐
        g = self.gate_channel_adjust(g)

        # 步骤2: 动态上采样，使g的尺寸与x一致
        x_size = x.shape[2:]  # x的空间尺寸 (z, y, x)
        g = F.interpolate(g, size=x_size, mode='trilinear', align_corners=True)

        # 步骤3: 逐元素相加
        fused = g + x

        # 步骤4: 大核分组卷积生成注意力权重
        attn = self.bn_attn(self.group_conv(fused))
        attn = self.sigmoid(attn)

        # 步骤5: 注意力加权
        x_attended = x * attn

        # 步骤6: Conv Block通道变换
        out = self.conv_block(x_attended)

        return out


class DenseBlock(nn.Module):
    """密集特征堆栈模块，实现特征复用"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv3d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False),
                nn.BatchNorm3d(growth_rate),
                Swish()
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class EnhancedDenseVNet(nn.Module):
    """增强型DenseVNet分割模型（尺寸自适应）"""
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        growth_rates = [4, 8, 16, 16]
        num_layers = [4, 4, 4, 4]

        # 固定各DenseBlock输出通道数
        self.d1_channels = 24 + growth_rates[0] * num_layers[0]    # 24+4×4=40
        self.d2_channels = 48 + growth_rates[1] * num_layers[1]    # 48+8×4=80
        self.d3_channels = 96 + growth_rates[2] * num_layers[2]    # 96+16×4=160
        self.d4_channels = 192 + growth_rates[3] * num_layers[3]   # 192+16×4=256

        # 编码器
        self.init_conv = nn.Conv3d(in_channels, 24, 3, padding=1, bias=False)
        self.dense1 = DenseBlock(24, growth_rates[0], num_layers[0])
        self.cab1 = ChannelAttentionBlock(self.d1_channels)
        self.down1 = nn.Conv3d(self.d1_channels, 48, 2, stride=2, bias=False)

        self.dense2 = DenseBlock(48, growth_rates[1], num_layers[1])
        self.cab2 = ChannelAttentionBlock(self.d2_channels)
        self.down2 = nn.Conv3d(self.d2_channels, 96, 2, stride=2, bias=False)

        self.dense3 = DenseBlock(96, growth_rates[2], num_layers[2])
        self.cab3 = ChannelAttentionBlock(self.d3_channels)
        self.down3 = nn.Conv3d(self.d3_channels, 192, 2, stride=2, bias=False)

        self.dense4 = DenseBlock(192, growth_rates[3], num_layers[3])
        self.cab4 = ChannelAttentionBlock(self.d4_channels)

        # 注意力门控
        self.lgag1 = LargeKernelGroupAttentionGate(self.d1_channels, self.d4_channels)
        self.lgag2 = LargeKernelGroupAttentionGate(self.d2_channels, self.d4_channels)
        self.lgag3 = LargeKernelGroupAttentionGate(self.d3_channels, self.d4_channels)

        # 解码器
        self.up1 = nn.ConvTranspose3d(self.d4_channels, 96, 2, stride=2, bias=False)
        self.dec_conv1 = nn.Sequential(
            nn.Conv3d(96 + self.d3_channels, 96, 3, padding=1, bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose3d(96, 48, 2, stride=2, bias=False)
        self.dec_conv2 = nn.Sequential(
            nn.Conv3d(48 + self.d2_channels, 48, 3, padding=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose3d(48, 24, 2, stride=2, bias=False)
        self.dec_conv3 = nn.Sequential(
            nn.Conv3d(24 + self.d1_channels, 24, 3, padding=1, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.final_conv = nn.Conv3d(24, num_classes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码路径
        x0 = self.init_conv(x)
        d1 = self.dense1(x0)
        d1 = self.cab1(d1)
        x1 = self.down1(d1)

        d2 = self.dense2(x1)
        d2 = self.cab2(d2)
        x2 = self.down2(d2)

        d3 = self.dense3(x2)
        d3 = self.cab3(d3)
        x3 = self.down3(d3)

        d4 = self.dense4(x3)
        d4 = self.cab4(d4)

        # 解码路径
        x_up1 = self.up1(d4)
        d3_gated = self.lgag3(d4, d3)
        x_up1 = torch.cat([x_up1, d3_gated], dim=1)
        x_up1 = self.dec_conv1(x_up1)

        x_up2 = self.up2(x_up1)
        d2_gated = self.lgag2(d4, d2)
        x_up2 = torch.cat([x_up2, d2_gated], dim=1)
        x_up2 = self.dec_conv2(x_up2)

        x_up3 = self.up3(x_up2)
        d1_gated = self.lgag1(d4, d1)
        x_up3 = torch.cat([x_up3, d1_gated], dim=1)
        x_up3 = self.dec_conv3(x_up3)

        # 输出层
        out = self.final_conv(x_up3)
        out = self.sigmoid(out)
        return out






