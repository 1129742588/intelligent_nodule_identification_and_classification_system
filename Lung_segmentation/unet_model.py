import glob
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk

class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features_num=32):
        super().__init__()
        self.features_num = features_num
        #编码器（卷积）
        self.enc1 = self.conv_relu_x2(in_channels        , self.features_num*1)
        self.enc2 = self.conv_relu_x2(self.features_num*1, self.features_num*2)
        self.enc3 = self.conv_relu_x2(self.features_num*2, self.features_num*4)
        self.enc4 = self.conv_relu_x2(self.features_num*4,self.features_num*8)
        #底层
        self.bottom = self.conv_relu_x2(self.features_num*8,self.features_num*16)
        #池化工具
        self.pool = nn.MaxPool2d(2)
        #解码器（上采样）
        self.up4 = nn.ConvTranspose2d(self.features_num*16, self.features_num*8, 2, 2)
        self.dec4 = self.conv_relu_x2(self.features_num*16, self.features_num*8)
        self.up3 = nn.ConvTranspose2d(self.features_num*8, self.features_num*4, 2, 2)
        self.dec3 = self.conv_relu_x2(self.features_num*8, self.features_num*4)
        self.up2 = nn.ConvTranspose2d(self.features_num*4, self.features_num*2, 2, 2)
        self.dec2 = self.conv_relu_x2(self.features_num*4, self.features_num*2)
        self.up1 = nn.ConvTranspose2d(self.features_num*2, self.features_num*1, 2, 2)
        self.dec1 = self.conv_relu_x2(self.features_num*2, self.features_num*1)
        #输出
        self.out = nn.Conv2d(self.features_num*1, out_channels, 3, padding=1)

    #自定义卷积+relu
    def conv_relu_x2(self, in_c ,out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    #数据处理流程
    def forward(self,x):
        #编码（下采样）
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        #底部
        b = self.bottom(self.pool(e4))
        #解码器（上采样+跳跃链接）
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4],dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3],dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2],dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1],dim=1)
        d1 = self.dec1(d1)

        #每阶段输出查看
        # print("1",x.shape)
        # print("2",e1.shape)
        # print("3",e2.shape)
        # print("4",e3.shape)
        # print("5",e4.shape)
        # print("6",b.shape)
        # print("7",d4.shape)
        # print("8",d3.shape)
        # print("9",d2.shape)
        # print("10",d1.shape)

        return self.out(d1)
    
    
if __name__ =="__main__":
    batch_size = 2
    h,w = 64,64
    rand_input = torch.randn(batch_size, 1, h, w)

    model=Unet()
    # 前向传播
    with torch.no_grad():  # 不计算梯度，节省内存
        output = model(rand_input)
    
    # 检查输出形状
    print(f"输入形状: {rand_input.shape}")
    print(f"输出形状: {output.shape}")



