"""
模型训练和评估的主函数，一些对模型相关的操作都在此类里面进行。
"""
import napari
import pandas as pd
import torch.nn as nn
from torch import optim
import SimpleITK as sitk
from nodules_identify.data_process import *
from nodules_identify.cnn_model import Simple3DCNN
from torch.cuda.amp import autocast, GradScaler

class ModelOperate:

    #------------------------------------------------------------------------------------
    def __init__(self, dataset_dir, output_dir):
        """
        初始化模型操作类，配置训练、验证和预测所需的参数、路径与模型对象。
        - 参数：
            - dataset_dir: 数据集目录路径，包含原始CT和掩膜数据。
            - output_dir: 输出目录路径，用于保存处理后的数据和模型权重，应与DataProcessor的输出目录一致。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 初始化训练超参数和最佳验证损失等状态变量。
            - 2. 创建DataProcessor对象，复用数据预处理与样本加载能力。
            - 3. 根据当前设备初始化模型、损失函数、优化器和混合精度缩放器。
            - 4. 构造样本CSV和模型权重文件的相关路径。
        """
        # 配置清单，这些参数通常在对象的整个生命周期中保持不变。
        self.patch_size = 32                                                                # 模型输入的区块大小，必须与数据预处理时的剪切区块大小一致，否则会报错
        self.threshold_up = 0.6                                                             # 预测时判断存在结节的概率阈值，超过该值则认为存在结节
        self.threshold_down = 0.3                                                           # 预测时判断不存在结节的概率阈值，低于该值则认为不存在结节，介于两个阈值之间则认为不确定
        self.epoch_nums = 50                                                                # 训练的总轮数，根据实际情况调整
        self.learning_rate = 1e-4                                                           # 优化器的学习率，根据实际情况调整
        self.is_continue = True                                                             # 是否继续训练，如果之前已经训练过并保存了模型权重，可以设置为True继续训练，否则设置为False从头开始训练
        self.best_val_loss = float('inf')                                                   # 初始化最佳验证损失为正无穷，训练过程中如果验证损失下降则更新该值并保存模型权重
        self.dataprocessor = DataProcessor(dataset_dir, output_dir)                         # 创建DataProcessor对象，复用数据预处理与样本加载能力

        # 模型初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")          # 根据当前环境选择计算设备，优先使用GPU加速，如果没有GPU则使用CPU
        self.model = Simple3DCNN(1, 2, self.patch_size).to(self.device)                     # 初始化模型对象，并将其移动到选定的计算设备上，输入通道数为1（CT图像），输出类别数为2（结节和非结节），区块大小由patch_size参数决定
        self.criterion = nn.CrossEntropyLoss()                                              # 定义交叉熵损失函数，适用于多分类问题，后续会根据训练集类别比例设置类别权重
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)         # 定义Adam优化器，传入模型参数和学习率，Adam优化器在训练深度神经网络时通常表现良好
        self.scaler = GradScaler()                                                          # 定义混合精度训练的梯度缩放器，用于在使用autocast进行前向传播时动态调整损失缩放因子，以防止数值下溢和提高训练效率  


        self.dataset_dir = dataset_dir
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = r"output"
        # 输出目录路径，用于保存处理后的训练数据，应与DataProcessor的输出目录一致，后续可能用于构造样本CSV路径和模型权重路径
        self.sample_csv_dir = os.path.join(self.output_dir, 'sample_csv')                        # 样本CSV文件目录，存储正负样本的CSV文件，后续用于加载训练数据和构造模型权重路径
        self.all_samples_csv_path = os.path.join(self.sample_csv_dir, 'all_samples.csv')    # 总样本CSV文件路径，包含所有正负样本的记录，后续用于加载训练数据和构造模型权重路径
        self.model_data_dir = os.path.join(os.path.dirname(__file__), 'model_data')                       # 模型权重文件目录，存储训练过程中保存的最佳模型权重，后续用于构造模型权重路径
        self.model_data_path = os.path.join(self.model_data_dir, 'best_nodule_model.pth')   # 最佳模型权重文件路径，存储验证损失最低时的模型权重，后续用于加载模型权重进行继续训练或预测

    #------------------------------------------------------------------------------------
    def train(self):
        """
        模型训练主函数，负责模型加载、训练循环、验证、保存最佳模型。
        - 参数：
            - 无显式参数，训练所需数据与配置均来自当前对象属性。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 输出当前设备信息，并通过DataProcessor加载训练集和验证集。
            - 2. 根据训练集类别比例设置交叉熵损失的类别权重。
            - 3. 如果允许继续训练且已有模型权重，则先加载历史最佳模型并计算当前验证损失。
            - 4. 按epoch循环执行前向传播、反向传播、梯度缩放与参数更新。
            - 5. 每轮训练结束后执行验证，并在验证损失下降时保存最佳模型权重。
        """
        # 检查设备
        print(f"当前计算设备：{self.device}")

        # 加载数据集
        train_dataloader, val_dataloader, wight_tensor = self.dataprocessor.divide_samples_dataset()

        # 加载模型优化器
        self.criterion.weight = wight_tensor.to(self.device)

        # 如果继续训练，加载之前的模型权重，并验证下损失
        if self.is_continue and os.path.exists(self.model_data_path):
            self.model.load_state_dict(torch.load(self.model_data_path, map_location=self.device))
            self.best_val_loss, _ = self.varify(val_dataloader)
            print("继续训练，当前验证，loss=", self.best_val_loss)


        # 开始训练
        for epoch in range(self.epoch_nums):
            self.model.train()
            total_loss = 0
            for images, labels in tqdm(train_dataloader, desc="训练进行中", ncols=100):
                # 移动至显存
                images = images.to(self.device)
                labels = labels.to(self.device)
                if not (images.shape[4]==self.patch_size):
                    print("剪切区块和训练参数大小不一致")
                    return


                # 前向传播 (使用 autocast)
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # 反向传播和优化 (使用 scaler)
                self.scaler.scale(loss).backward()           # 缩放损失并反向传播
                self.scaler.step(self.optimizer)                  # 缩放梯度并更新参数
                self.scaler.update()                          # 更新缩放因子
                # 清零梯度
                self.optimizer.zero_grad()
                #损失值记录
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_dataloader)

            # 如果当前验证 loss 比之前都好，保存模型
            val_loss, val_acc = self.varify(val_dataloader)
            print(f"第{epoch+1}次epoch---训练，Loss= {avg_train_loss:.4f}")
            print(f"第{epoch+1}次epoch---验证，loss= {val_loss:.4f},验证acc={val_acc*100:.4f}%")
            if val_loss<self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_data_path)
                print(f"{'-'*25}保存最佳模型，第{epoch+1}个epoch {'-'*25}")

    #------------------------------------------------------------------------------------
    def optimize_train(self, optimizer_times=3):
        """
        优化训练流程，会自动添加优化负样本，再进行训练
        - 参数：
            - optimizer_times: 优化训练的迭代次数，每次迭代会先进行一次训练，然后根据当前模型权重添加新的优化负样本，默认为2。
        """
        for i in range(optimizer_times):
            # 1. 先进行一次训练，得到当前模型权重
            self.train()
            self.dataprocessor.batch_optimizer_false_samples(model_operator=self)

    #------------------------------------------------------------------------------------
    def varify(self, dataloader):
        """
        验证模型性能，计算平均损失和准确率。
        - 参数：
            - dataloader: 验证集DataLoader。
        - 返回：
            - average_loss: 验证集平均损失。
            - accuracy: 验证集准确率。
        - 流程:
            - 1. 在不计算梯度的模式下遍历验证集。
            - 2. 将数据移动到当前设备，并执行混合精度前向传播。
            - 3. 累计损失、预测正确数和样本总数。
            - 4. 计算并返回平均损失与准确率。
        """
        total_loss = 0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="验证进行中", ncols=100):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                # 计算预测类别
                _, preds = torch.max(outputs, dim=1) # preds 形状: (batch_size,)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        average_loss = total_loss/len(dataloader)
        accuracy = correct / total_samples
        return average_loss,accuracy

    #------------------------------------------------------------------------------------
    def predict(self, ct, mask=None):
        """
        使用训练好的模型进行批量预测，输出预测结果和相关信息。
        - 参数：
            - ct: 输入的CT数据，可以是文件路径或重采样后的numpy数组。
            - mask: 输入的掩膜数据，可以是文件路径或重采样后的numpy数组，可以为空，但会增加无关的patch，速度减慢。
            - 注: (可正确处理dicom目录和mhd文件)
        - 返回：
            - identify_results: 包含各个结节字典的列表，identify_results[0]:
                -  key : value
                - "nodule_index": 结节索引，int类型，表示该样本在所有预测patch中的索引位置。
                - "nodule_prob": 结节概率，float类型，表示该样本被预测为结节的概率。
                - "patch_center": 区块中心坐标，(z, y, x)。
                - "patch_array": 区块数据，numpy数组，该样本所在区块的重采样CT图像数据。

                
        - 流程:
            - 1. 判断输入数据类型，支持路径或数组。
            - 2. 滑动获取区块，并转换为DataLoader。
            - 3. 加载模型权重并切换到预测模式。
            - 4. 进行预测，计算结节概率。
            - 5. 根据概率阈值判断是否存在结节，并统计数量。
            - 6. 输出预测结果和相关信息。
        """
        # 全局变量
        nodule_nums = 0
        all_nodule_probs = []
        identify_results = []
        
        # 判断输入数据类型，支持路径或数组-ct
        if isinstance(ct, str):
            ct_array, ct_information = self.dataprocessor.load_data(ct)
            resampled_ct = self.dataprocessor.resample_array(ct_array, ct_information)

        elif isinstance(ct, np.ndarray):
            resampled_ct = ct
        else:
            raise ValueError("输入数据类型不支持，必须为文件路径或numpy数组。")
        
        # 判断输入数据类型，支持路径或数组-mask
        if isinstance(mask, str):
            mask_array, mask_information = self.dataprocessor.load_data(mask)
            resampled_mask = self.dataprocessor.resample_array(mask_array, mask_information)
        elif isinstance(mask, np.ndarray):
            resampled_mask = mask
        else:
            # 如果没有掩膜数据，设置为None，后续处理时需要考虑这种情况
            resampled_mask = None
            mask_array = None
        

        # 检测模型数据是否存在
        if not os.path.exists(self.model_data_path):
            raise FileNotFoundError(f"模型数据文件 {self.model_data_path} 不存在，请先训练模型")
        
        # 滑动获取区块
        patches_array, patch_centers = self.dataprocessor.slide_getpatch(resampled_ct, resampled_mask)

        # 将patches_array转换为DataLoader
        dataloader = self.dataprocessor.get_dataloader(patches_array)
        
        # 加载模型权重并切换到预测模式
        self.model.load_state_dict(torch.load(self.model_data_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 进行预测
        with torch.no_grad():
            for images in tqdm(dataloader, desc="结节识别：", ncols=100):
                images = images[0].to(self.device).float()
                with autocast():
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)
                # 取正类（结节）的概率
                all_nodule_probs.append(probs[:, 1].cpu().numpy())  # 形状 (N,)
            nodule_probs = np.concatenate(all_nodule_probs)


        # # 返回结果处理
        # for i in range(nodule_probs.shape[0]):
        #     if nodule_probs[i] >= self.threshold_up:
        #         identify_results['nodule_indexs'].append(i)
        #         nodule_nums += 1
        # identify_results['nodule_probs'] = nodule_probs
        # identify_results['nodule_nums'] = nodule_nums
        # identify_results['patch_centers'] = patch_centers
        # identify_results['patches_array'] = patches_array


        # 简化返回处理
        identify_results = []
        for i in range(nodule_probs.shape[0]):
            if nodule_probs[i] >= self.threshold_up:
                identify_results.append({
                    'nodule_index': i,
                    'nodule_prob': nodule_probs[i],
                    'patch_center': patch_centers[i],
                    'patch_array': patches_array[i]
                })

        # 输出显示
        # if np.any(nodule_probs >= self.threshold_up):
        #     print(f"预测结果：存在结节，结节概率最高为{nodule_probs.max():.4f}，结节数量为{nodule_nums}")
        # elif np.max(nodule_probs) < self.threshold_down:
        #     print(f"预测结果：不存在结节，结节概率最高为{nodule_probs.max():.4f}")
        # else:
        #     print(f"预测结果：不确定，结节概率最高为{nodule_probs.max():.4f}")
        
        return identify_results
    

if __name__ == "__main__":
    """使用示例"""
    # 全局变量
    dataset_dir = r"D:\learnfile\dataset\LUNA16"
    output_dir = r"D:\Learnfile\Dataset\LUNA16\sample_data"
    data_processor = DataProcessor(dataset_dir, output_dir)
    model_operator = ModelOperate(dataset_dir, output_dir)
    model_operator.epoch_nums = 30
    # 数据集划分和加载

    """模型训练和评估"""
    print(f"当前设备: {model_operator.device}")
    # model_operator.optimize_train(optimizer_times=5)

    """模型预测"""
    # 单个ct路径构造
    file_name = r"1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
    predict_ct_dir = r"D:\Learnfile\Dataset\LUNA16\subset0"
    predict_mask_dir = r"D:\Learnfile\Dataset\LUNA16\seg-lungs-LUNA16"
    predict_ct_path = os.path.join(predict_ct_dir, file_name)
    predict_mask_path = os.path.join(predict_mask_dir, file_name)
    predict_mask_path = None
    predict_ct_path = r"D:\Learnfile\Dataset\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
    
    # 进行预测,ct数组，mhd文件，dicom文件的目录都可以。mask可以不提供，但会增加无关的patch，预测速度减慢。
    output_information = model_operator.predict(predict_ct_path, predict_mask_path)
    print(f"patch形状:{output_information['patches_array'].shape}")
    print(len(output_information['patch_centers']))
    print(f"结节数量: {output_information['nodule_nums']} 个")
    for idx in output_information['nodule_indexs']:
        print(f"结节索引: {idx}, 结节概率: {output_information['nodule_probs'][idx]:.4f}, 有结节的区块中心坐标: {output_information['patch_centers'][idx]}")
    
    """多维数组可视化测试"""
    viewer = napari.view_image(output_information['patches_array'], name='CT', colormap='gray')
    napari.run()
    # 保存



