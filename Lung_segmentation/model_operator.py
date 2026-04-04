import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from Lung_segmentation.unet_model import Unet
from Lung_segmentation.data_processor import *
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader,random_split


class ModelOperator:
    #------------------------------------------------------------------------------------
    def __init__(self, dataset_dir):
        """
        初始化模型操作器
        - 参数:
            - dataset_dir: LUNA16数据集根目录，应包含 subset*/ 和 seg-lungs-LUNA16/ 文件夹
        - 成员变量:
            - epochs: 训练轮数
            - batch_size: 每批次样本数量
            - learning_rate: 学习率
            - train_size_ratio: 训练集占总数据的比例
            - iscontinue: 是否启用断点续训
            - num_workers: DataLoader的子进程数量，建议0,2,4,8依次尝试，选取速度最快的
            - best_val_loss: 验证集上的最佳损失值，用于模型保存
            - device: 计算设备（GPU或CPU）
            - model: UNet模型实例
            - criterion: 损失函数
            - optimizer: 优化器
            - scaler: 混合精度训练的梯度缩放器
            - dataset_dir: 数据集根目录
            - model_data_dir: 模型数据保存目录
            - model_data_path: 最佳模型权重文件路径
        """
        # 参数配置
        self.epochs = 10
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.train_size_ratio = 0.8
        self.iscontinue = True
        self.num_workers = 4
        self.best_val_loss = float('inf')

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Unet(in_channels=1, out_channels=1, features_num=32).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = GradScaler()  # 用于混合精度训练

        # 路径配置
        self.dataset_dir = dataset_dir
        self.model_data_dir = os.path.join(os.path.dirname(__file__), 'model_data')
        self.model_data_path = os.path.join(self.model_data_dir, 'best_unet_model.pth')

        # 必要目录创建
        os.makedirs(self.model_data_dir, exist_ok=True)

    #------------------------------------------------------------------------------------
    def train(self):
        """
        训练模型主函数，无需参数，无返回，若需要调整参数，则在实例化对象后修改属性即可
        """
        print(f"当前计算设备：{self.device}")
        print("数据加载...")

        #读取数据
        dataset = LUNA16_dataload(self.dataset_dir, cache_npy=True)
        #划分训练与验证集
        dataset_size = len(dataset)
        train_size = int(dataset_size*self.train_size_ratio)
        val_size = dataset_size - train_size
        # 随机划分
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        """小数据测试能否跑通"""
        # val_size = int(0.01*dataset_size)
        # x_size = dataset_size - val_size
        # x,val_dataset = random_split(dataset, [x_size, val_size])

        # 数据加载
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        # 断点续训
        if self.iscontinue and os.path.exists(self.model_data_path):
            self.model.load_state_dict(torch.load(self.model_data_path, map_location=self.device))
            # 验证当前模型性能
            self.best_val_loss = self.verify(val_dataloader)
            print(f"继续训练，当前验证: loss={self.best_val_loss:.4f}")

        # 开始训练
        for epoch in range(self.epochs):
            train_loss = 0.0
            for images, masks in tqdm(train_dataloader, desc=f"第{epoch+1}/{self.epochs}次训练", ncols=100):
                #移动到显存
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 如果 masks 没有通道维，添加
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)   # (batch, 1, H, W)
                # 前向传播 (使用 autocast 混合精度训练)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks.float())
                train_loss += loss.item()

                # 反向传播和优化
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # 每轮结束后验证
            train_loss /= len(train_dataloader)
            val_loss = self.verify(val_dataloader)
            print(f"第{epoch+1}轮训练: loss={train_loss:.4f}")
            print(f"第{epoch+1}轮验证: loss={val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_data_path)
                print(f"！！！保存最佳模型，第{epoch+1}个epoch！！！")
            print("-"*100)

    #------------------------------------------------------------------------------------
    def verify(self, val_dataloader):
        """
        验证模型性能
        - 参数:
            - val_dataloader: 验证数据加载器
        - 返回:
            - average_loss: 验证集上的平均损失值
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader, desc="验证中", ncols=100):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)   # (batch, 1, H, W)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks.float())
                # 累加损失
                total_loss += loss.item()

        average_loss = total_loss / len(val_dataloader)
        return average_loss

    #------------------------------------------------------------------------------------
    def predict(self, ct_data):
        """
        预测函数，输入CT图像数组，文件路径或dicom文件目录，输出预测的肺部掩膜数组
        - 参数:
            - ct_data: (N, H, W)或(H, W)的numpy数组或CT图像路径字符串，表示待预测的CT图像数据
        - 返回:
            - predicted_mask: (N, H, W)的numpy，二值化的肺部掩膜
        - 注: 输入CT图像会被归一化并转换为4D张量 (batch, channels, height, width) 以适应模型输入要求
        """
        # 判断输入的是路径还是数组
        if isinstance(ct_data, str):
            data_processor = DataProcessor(self.dataset_dir)
            ct_array = data_processor.load_data(ct_data)
        elif isinstance(ct_data, np.ndarray):
            ct_array = ct_data
        else:
            raise ValueError("输入数据必须是CT图像路径或CT图像数组")


        print(f"使用设备: {self.device}")
        # 检测模型数据是否存在
        if not os.path.exists(self.model_data_path):
            raise FileNotFoundError(f"模型数据文件 {self.model_data_path} 不存在，请先训练模型")
        # 单张归一化
        ct_array_norm = np.zeros_like(ct_array, dtype=np.float32)
        for i in range(ct_array.shape[0]):
            ct_array_norm[i] = (ct_array[i] - np.min(ct_array[i])) / (np.max(ct_array[i]) - np.min(ct_array[i]) + 1e-8)

        # 确保输入是4D张量 (batch, channels, height, width)
        if ct_array_norm.ndim == 3:
            # 添加通道维度
            ct_array_norm = np.expand_dims(ct_array_norm, axis=1)
        if ct_array_norm.ndim != 4:
            raise ValueError("输入的CT图像数组维度不正确")
        ct_tensor = torch.from_numpy(ct_array_norm)

        # 加载为data_loader的形式进行预测
        dataset = TensorDataset(ct_tensor)
        data_loader = DataLoader(dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        # 预测
        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_data_path, map_location=self.device))
        predicted_mask_list = []
        for batch in tqdm(data_loader, desc="预测中", ncols=100):
            images = batch[0].to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                predicted_mask = torch.sigmoid(outputs) > 0.5  # 二值化
                predicted_mask = predicted_mask.float()
                predicted_mask = predicted_mask.cpu().numpy()
                predicted_mask_list.append(predicted_mask)

        return np.concatenate(predicted_mask_list, axis=0).squeeze(1)  # (N, H, W)


if __name__ =="__main__":
    """模型操作使用示例"""
    # 全局参数配置
    dataset_dir = r'D://Learnfile//Dataset//LUNA16'
    ct_path = r'D:\Learnfile\Dataset\LIDC-IDRI\LIDC-IDRI-0003\1.3.6.1.4.1.14519.5.2.1.6279.6001.101370605276577556143013894866\1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264'
    data_processor = DataProcessor(dataset_dir)
    model_operator = ModelOperator(dataset_dir)

    """训练模型"""
    # model_operator.train()

    """预测示例"""
    # 读取CT图像
    ct_array = data_processor.load_data(ct_path)
    # 预测肺部掩膜
    predicted_mask = model_operator.predict(ct_array)
    print("预测完成，CT  形状:", ct_array.shape)
    print("预测完成，掩膜形状:", predicted_mask.shape)
    # 保存结果为npy文件
    np.save('predicted_lung_mask.npy', predicted_mask)

    # 结果显示
    slice_idx = ct_array.shape[0] // 2 # 显示中间切片
    plt.subplot(1, 2, 1)
    plt.title("CT Image")
    plt.imshow(ct_array[slice_idx], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Predicted Lung Mask")
    plt.imshow(predicted_mask[slice_idx], cmap='gray')
    plt.show()






