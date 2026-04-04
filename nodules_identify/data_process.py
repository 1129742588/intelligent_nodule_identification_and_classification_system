"""
此文件应包含该项目理论上的所有数据处理相关的操作，例如读取，加载，保存，预处理等等
"""
from hmac import new
import os
import glob
from sympy import false
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from torch.utils.data import ConcatDataset, Dataset, DataLoader, TensorDataset, random_split

class PatchDataset(Dataset):

    """---------------裁剪区块读取---------------"""

    #------------------------------------------------------------------------------------
    def __init__(self, csv_file, transform=None):
        """
        加载csv文件为Dataset对象，便于批量训练。
        - 参数：
            - csv_file: 包含区块路径和标签的CSV文件。
            - transform: 可选预处理或数据增强函数，接收单个patch并返回处理后的patch。
        - 返回：
            - 无返回值。
        - 注: 自动读取CSV索引信息，样本实际加载发生在__getitem__中，当前实现要求区块文件为npy格式。
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
    
    #------------------------------------------------------------------------------------
    def __len__(self):
        """
        返回数据集中样本的总数量。
        - 参数：
            - 无。
        - 返回：
            - 样本总数，类型为int。
        """
        return len(self.df)
    
    #------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        获取指定索引的样本，返回patch和label。
        - 参数：
            - idx: 索引。
        - 返回：
            - patch: 标准化后的张量。
            - label: 标签张量。
        - 流程:
            - 1. 从CSV中读取指定索引对应的样本路径和标签。
            - 2. 加载npy格式的3D区块，并补充通道维度。
            - 3. 对区块执行带低方差保护的标准化处理。
            - 4. 若传入transform，则执行预处理或数据增强。
            - 5. 将区块和标签转换为PyTorch张量后返回。
        """
        row = self.df.iloc[idx]
        patch = np.load(row['path'])  # shape (patch_size, patch_size, patch_size)
        patch = patch[np.newaxis, :, :, :].astype(np.float32)
        label = int(row['label'])
        mean = patch.mean()
        std = patch.std()
        if std < 1e-6:
            std = 1.0
        patch = (patch - mean) / std

        if self.transform is not None:
            patch = self.transform(patch)

        if isinstance(patch, np.ndarray):
            patch = np.ascontiguousarray(patch)
            patch_tensor = torch.from_numpy(patch.astype(np.float32, copy=True))
        elif isinstance(patch, torch.Tensor):
            patch_tensor = patch.to(dtype=torch.float32)
        else:
            raise TypeError("transform 必须返回 numpy.ndarray 或 torch.Tensor")

        return patch_tensor, torch.tensor(label, dtype=torch.long)




"""-------------------------------------数据处理类-------------------------------------"""
class DataProcessor:
    
    #-----------------------------------初始化-------------------------------------------------
    def __init__(self, dataset_dir, output_dir):
        """
        初始化数据处理类，负责所有数据相关操作。
        - 参数：
            - dataset_dir: 原始数据集目录。
            - output_dir: 输出基础路径。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 初始化区块大小、采样数量、填充值和重采样分辨率等处理参数。
            - 2. 构造重采样结果、样本缓存和样本CSV的各类路径。
            - 3. 创建后续处理所需的输出目录，确保数据可直接写入。
        """
        # 初始化参数
        self.patch_size = 32                                                    # 模型输入的区块大小，必须与模型定义时的输入尺寸一致，否则会报错
        self.centers_num = 5                                                    # 每个mhd文件提取的区块中心数量，过多可能导致训练过拟合，过少可能导致训练不足，根据实际情况调整
        self.batch_size = 12                                                    # 训练时的批大小，根据显存大小和模型复杂度调整，过大可能导致显存不足，过小可能导致训练不稳定
        self.fill_value = -1024                                                 # 边界外填充的数值，通常选择CT图像中的空气值，确保模型学习到边界信息
        self.safety_margin = 10                                                 # 在提取区块时添加安全边距，避免过于靠近结节导致的无效样本，单位为像素，根据实际情况调整
        self.is_regenerate = False                                              # 是否强制重新生成数据，如果之前已经生成过数据并保存了结果，可以设置为False跳过重生成，否则设置为True强制重新生成   
        self.use_mask = True                                                    # 是否使用mask进行负样本筛选，如果为True则在提取负样本时会检查对应位置的mask值，只有当mask值为0（非结节区域）时才会被选为负样本，否则设置为False则不考虑mask信息直接提取负样本
        self.new_spacing = (1.0, 1.0, 1.0)                                      # 重采样后的空间分辨率，单位为mm/体素，通常设置为(1.0, 1.0, 1.0)以简化后续处理，确保重采样后每个体素对应实际空间中的1mm³
        self.new_forbid_radius = int(self.patch_size/4)                         # 新增禁止区域半径，默认为patch_size的1/4
        self.one_ct_optimizer_false_sample_limit = 5                            # 每个CT新增的优化负样本数量上限，防止过多假阳性导致训练不稳定
        self.batch_ct_optimizer_false_sample_limit = 100                        # 每批次CT新增的优化负样本数量上限，防止过多假阳性导致训练不稳定

        # 初始化路径
        self.dataset_dir = dataset_dir                                          # 数据集目录路径，包含原始CT和掩膜数据，后续可能用于加载数据或构造样本CSV路径
        self.output_dir = output_dir                                            # 输出目录路径，用于保存处理后的训练数据，应与模型操作类中的输出目录一致，后续可能用于构造样本CSV路径和模型权重路径
        if self.output_dir is not None:
            # 构造处理结果的各类路径
            self.annotations_csv_path = os.path.join(dataset_dir, "CSVFILES", "annotations.csv")
            self.resampled_ct_dir = os.path.join(output_dir, 'ct(resampled)')
            self.resampled_mask_dir = os.path.join(output_dir, 'mask(resampled)')
            self.transformed_csv_path = os.path.join(output_dir, 'transformed_annotations.csv')
            self.true_sample_patches_dir = os.path.join(output_dir, 'true_sample_npy_cache')
            self.false_sample_patches_dir = os.path.join(output_dir, 'false_sample_npy_cache')
            self.optimizer_false_sample_patches_dir = os.path.join(output_dir, 'optimizer_false_sample_npy_cache')
            self.sample_csv_dir = os.path.join(output_dir, 'sample_csv')
            self.true_sample_csv_path = os.path.join(self.sample_csv_dir, 'true_samples.csv')
            self.false_sample_csv_path = os.path.join(self.sample_csv_dir, 'false_samples.csv')
            self.optimizer_false_sample_csv_path = os.path.join(self.sample_csv_dir, 'optimizer_false_samples.csv')
            self.all_sample_csv_path = os.path.join(self.sample_csv_dir, 'all_samples.csv')
            # 创建必要的目录
            os.makedirs(self.resampled_ct_dir, exist_ok=True)
            os.makedirs(self.resampled_mask_dir, exist_ok=True)
            os.makedirs(self.true_sample_patches_dir, exist_ok=True)
            os.makedirs(self.false_sample_patches_dir, exist_ok=True)
            os.makedirs(self.optimizer_false_sample_patches_dir, exist_ok=True)
            os.makedirs(self.sample_csv_dir, exist_ok=True)


    #-----------------------------------加载数据-----------------------------------------------
    def load_data(self, data_path):
        """
        读取单个医学影像文件或DICOM序列文件夹，并返回图像数组及空间信息。
        - 参数：
            - data_path: 单个(mhd/dicom)文件路径或包含dicom文件的文件夹路径
        - 返回：
            - image_array: (N, H, W)的numpy数组，N为切片数，H和W为图像尺寸
            - sitk_information: 包含ct空间信息的字典，用于后续重采样，包含以下键值:
                - spacing: 图像空间分辨率
                - size: 图像尺寸
                - direction: 图像方向
                - origin: 图像原点
        - 流程:
            - 1. 判断输入路径是单文件还是目录。
            - 2. 若为单文件则直接读取，若为目录则按DICOM序列方式读取。
            - 3. 将图像统一转换为三维numpy数组，必要时为2D图像补充切片维度。
            - 4. 提取并返回spacing、size、direction和origin等空间信息。
        """
        sitk_information = {}
        if os.path.isfile(data_path):
            image = sitk.ReadImage(data_path)
            image_array = sitk.GetArrayFromImage(image)
            if image_array.ndim == 2:
                # 如果是2D图像，添加一个维度
                image_array = image_array[np.newaxis, ...]

        elif os.path.isdir(data_path):
            # 读取该文件夹下的DICOM序列（自动按空间位置排序）
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(data_path)
            if not series_ids:
                raise ValueError("文件夹中未找到DICOM序列")

            # 通常一个病人只有一个序列，取第一个
            file_names = reader.GetGDCMSeriesFileNames(data_path, series_ids[0])
            reader.SetFileNames(file_names)

            # 转换为numpy数组，维度顺序为 (z, y, x)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image)
            if image_array.ndim == 2:
                # 如果是2D图像，添加一个维度
                image_array = image_array[np.newaxis, ...]
        else:
            raise ValueError("输入路径既不是文件也不是文件夹，可能不存在该路径")
        
        # 保存图像的空间信息
        sitk_information['spacing'] = image.GetSpacing()
        sitk_information['size'] = image.GetSize()
        sitk_information['direction'] = image.GetDirection()
        sitk_information['origin'] = image.GetOrigin()
        return image_array, sitk_information


    #-----------------------------------单张CT重采样-------------------------------------------
    def resample_array(self, image_array, sitk_information, save_path=None):
        """
        重采样3D数组为统一spacing。
        - 参数：
            - image_array: 输入的numpy数组。
            - sitk_information: 空间信息字典。
            - save_path: 保存路径（可选）。
        - 返回：
            - resampled_array: 重采样后的数组。
        - 流程:
            - 1. 若目标文件已存在且当前不强制重生成，则直接读取缓存结果。
            - 2. 将numpy数组转为SimpleITK图像并恢复原始空间信息。
            - 3. 根据原始spacing和目标spacing计算新的图像尺寸。
            - 4. 执行重采样并在需要时写入磁盘。
        """
        # 如果已存在重采样后的文件且不需重新生成，则直接读取并返回
        if self.is_regenerate == False:
            if (save_path is not None):
                if os.path.exists(save_path):
                    resampled_array, _ = self.load_data(save_path)
                    return resampled_array

        # 创建sitk图像对象
        Image = sitk.GetImageFromArray(image_array)
        Image.SetSpacing(sitk_information['spacing'])
        Image.SetOrigin(sitk_information['origin'])
        Image.SetDirection(sitk_information['direction'])
        # 创建新的spacing信息
        new_size = [int(round(s * sp / new_sp)) for s, sp, new_sp in zip(sitk_information['size'], sitk_information['spacing'], self.new_spacing)]
        # 重采样图像
        resampled_image = sitk.Resample(Image, new_size, sitk.Transform(), sitk.sitkLinear,
                                        Image.GetOrigin(), self.new_spacing, Image.GetDirection())
        # 转换回numpy数组
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        if save_path is not None:
            sitk.WriteImage(resampled_image, save_path)
        return resampled_array


    #-----------------------------------批量重采样--------------------------------------------
    def batch_resample(self):
        """
        批量重采样数据集目录下所有mhd文件。
        - 参数：
            - 无显式参数，使用对象内部配置的输入输出路径和处理开关。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 遍历数据集目录，收集待处理的CT和mask文件路径。
            - 2. 逐个对CT执行重采样，并将结果保存到重采样目录。
            - 3. 若启用mask处理，则同步对mask执行同样的重采样操作。
            - 4. 输出批处理完成后的保存位置说明。
        """
        # 临时参数
        ct_paths = []
        mask_paths = []
        # 输入文件路径构造
        for subset in glob.glob(os.path.join(self.dataset_dir, "subset*")):#这里需修改为*，以读取全部数据集
            for mhd_path in glob.glob(os.path.join(subset, "*.mhd")): 
                ct_paths.append(mhd_path)
                if self.use_mask:
                        mask_name = os.path.basename(mhd_path)
                        mask_path = os.path.join(self.dataset_dir, 'seg-lungs-LUNA16', mask_name)
                        mask_paths.append(mask_path)

        # CT批量重采样
        for ct_path in tqdm(ct_paths, desc="ct重采样  \t", ncols=100):
            # 构造保存路径并重采样保存
            ct_save_path = os.path.join(self.resampled_ct_dir, os.path.basename(ct_path))
            # 如果文件已存在且不需要重新生成，则跳过重采样
            if self.output_dir!=None and self.is_regenerate==False and os.path.isfile(ct_save_path):
                continue
            # 读取原始ct图像
            ct_array, ct_information = self.load_data(ct_path)
            # 单张重采样
            self.resample_array(ct_array, ct_information, save_path=ct_save_path)
        
        # mask批量重采样
        if self.use_mask:
            for mask_path in tqdm(mask_paths, desc="mask重采样\t", ncols=100):
                # 构造保存路径并重采样保存
                mask_save_path = os.path.join(self.resampled_mask_dir, os.path.basename(mask_path))
                # 如果文件已存在且不需要重新生成，则跳过重采样
                if self.output_dir!=None and self.is_regenerate==False and os.path.isfile(mask_save_path):
                    continue
                # 读取原始ct图像
                mask_array, mask_information = self.load_data(mask_path)
                # 单张重采样
                self.resample_array(mask_array, mask_information, save_path=mask_save_path)
        # 输出完成提示
        print(f"批量重采样完成:\n---ct文件保存在 {self.resampled_ct_dir}，\n---mask文件保存在 {self.resampled_mask_dir}")


    #-----------------------------------坐标转换----------------------------------------------
    def transform_coord(self):
        """
        将原始标注坐标转换到重采样空间，生成新的csv。
        - 参数：
            - 无显式参数，直接使用对象中保存的标注CSV路径与重采样CT目录。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 若转换结果已存在且无需重新生成，则直接跳过。
            - 2. 读取原始标注CSV，并按seriesuid整理每个CT的结节信息。
            - 3. 遍历重采样CT文件，将世界坐标转换为体素坐标。
            - 4. 将转换结果整理为新的DataFrame并写入transformed_annotations.csv。
        """
        # 是否需要重新转换，如果不需要且文件已存在，则直接返回
        if self.is_regenerate == False and os.path.exists(self.transformed_csv_path):
            print(f"转换文件已存在且不需要重新生成，跳过坐标转换，直接使用 {self.transformed_csv_path}")
            return

        # 读取csv结节标注文件
        annotations_df = pd.read_csv(self.annotations_csv_path)
        # 存储结节信息key:文件名,value:世界坐标列表
        converted_annotations_list = []
        nodules_dict = {}
        for _, row in annotations_df.iterrows():
            uid = row['seriesuid']
            if uid not in nodules_dict:
                nodules_dict[uid] = []
            nodules_dict[uid].append({
            'coordX': row['coordX'],
            'coordY': row['coordY'],
            'coordZ': row['coordZ'],
            'diameter_mm': row['diameter_mm']
        })
        # 坐标转换
        for filename in tqdm(os.listdir(self.resampled_ct_dir), desc="坐标转换", ncols=100):
            # 判断是否是mhd文件
            if not filename.endswith('.mhd'):
                continue

            # 去除扩展名，并检查是否有结节
            filename_text = os.path.splitext(filename)[0]
            if filename_text not in nodules_dict:
                continue

            # 读取重采样后的mhd文件
            resampled_image = sitk.ReadImage(os.path.join(self.resampled_ct_dir, filename))

            # 对该CT的每个结节进行转换
            for nodule in nodules_dict[filename_text]:
                world_point = (nodule['coordX'], nodule['coordY'], nodule['coordZ'])
                try:
                    voxel_index = resampled_image.TransformPhysicalPointToIndex(world_point)
                except Exception as e:
                    print(f"转换失败: {filename_text}, 点 {world_point}, 错误: {e}")
                    continue
                # 将转换后的坐标保存到临时列表中
                converted = {
                                'seriesuid': filename_text,
                                'world_x': nodule['coordX'],
                                'world_y': nodule['coordY'],
                                'world_z': nodule['coordZ'],
                                'voxel_x': voxel_index[0],  # 列索引
                                'voxel_y': voxel_index[1],  # 行索引
                                'voxel_z': voxel_index[2],  # 层索引
                                'diameter_mm': nodule['diameter_mm'],
                                'diameter_vox': nodule['diameter_mm']  # 因为重采样后1mm/体素，直径体素数≈毫米数
                            }
                converted_annotations_list.append(converted)
        # 将转换后的坐标保存到新的CSV文件
        converted_df = pd.DataFrame(converted_annotations_list)
        converted_df.to_csv(self.transformed_csv_path, index=False)

        print(f"转换完成，共转换 {len(converted_annotations_list)} 个结节坐标，保存到 {self.transformed_csv_path}")


    #-----------------------------------提取3D块----------------------------------------------
    def extract_3d_patch(self, image_array, center, patch_size=None):
        """
        从3D数组中裁剪立方体patch，边界外填充指定值。
        - 参数：
            - image_array: 原始3D数组。
            - center: patch中心坐标（z,y,x）。
            - patch_size: patch边长，若为None则使用对象默认的patch_size。
        - 返回：
            - patch: 裁剪后的patch数组。
        - 流程:
            - 1. 根据中心点和边长计算目标区块在原图中的起止位置。
            - 2. 计算与原图相交的有效区域范围。
            - 3. 创建填充值初始化的目标立方体区块。
            - 4. 将原图中的有效区域复制到目标区块的对应位置。
        """
        if patch_size==None:
            patch_size = self.patch_size


        cz, cy, cx = center
        half = patch_size // 2
        
        # 计算目标块在原始数组中的边界（可能超出）
        z_start = cz - half
        z_end = cz + half
        y_start = cy - half
        y_end = cy + half
        x_start = cx - half
        x_end = cx + half
        
        # 计算实际需要从原始数组切片的区域（有效区域）
        z_src_start = max(0, z_start)
        z_src_end = min(image_array.shape[0], z_end)
        y_src_start = max(0, y_start)
        y_src_end = min(image_array.shape[1], y_end)
        x_src_start = max(0, x_start)
        x_src_end = min(image_array.shape[2], x_end)
        
        # 计算目标块中对应区域的偏移
        z_tgt_start = z_src_start - z_start
        z_tgt_end = z_tgt_start + (z_src_end - z_src_start)
        y_tgt_start = y_src_start - y_start
        y_tgt_end = y_tgt_start + (y_src_end - y_src_start)
        x_tgt_start = x_src_start - x_start
        x_tgt_end = x_tgt_start + (x_src_end - x_src_start)
        
        # 创建目标块，初始化为填充值
        patch = np.full((patch_size, patch_size, patch_size), self.fill_value, dtype=image_array.dtype)
        
        # 复制有效区域
        patch[z_tgt_start:z_tgt_end, y_tgt_start:y_tgt_end, x_tgt_start:x_tgt_end] = \
            image_array[z_src_start:z_src_end, y_src_start:y_src_end, x_src_start:x_src_end]
        
        return patch


    #-----------------------------------生成禁止区域-------------------------------------------
    def generate_forbid_region(self, ct_shape, nodule_centers, diameters):
        """
        生成禁止采样区域mask，防止负样本采样到结节附近。
        - 参数：
            - ct_shape: CT图像形状。
            - nodule_centers: 结节中心坐标列表。
            - diameters: 结节直径列表。
        - 返回：
            - forbid_region: 布尔数组，True为禁止采样。
        - 流程:
            - 1. 初始化与CT同形状的布尔型禁止区域数组。
            - 2. 根据每个结节的中心和直径计算带安全边距的球形区域。
            - 3. 将球形区域映射到数组中并标记为禁止采样。
        """
        forbid_region = np.zeros(ct_shape, dtype=bool)
        for center, diam in zip(nodule_centers, diameters):
            radius = diam / 2.0 + self.safety_margin
            cz, cy, cx = center
            # 计算球体覆盖的立方体范围
            z0 = max(0, int(cz - radius))
            z1 = min(ct_shape[0], int(cz + radius) + 1)
            y0 = max(0, int(cy - radius))
            y1 = min(ct_shape[1], int(cy + radius) + 1)
            x0 = max(0, int(cx - radius))
            x1 = min(ct_shape[2], int(cx + radius) + 1)
            
            # 创建网格
            Z, Y, X = np.ogrid[z0:z1, y0:y1, x0:x1]
            dist = np.sqrt((Z - cz)**2 + (Y - cy)**2 + (X - cx)**2)
            forbid_region[z0:z1, y0:y1, x0:x1] |= (dist <= radius)
        return forbid_region


    #-----------------------------------随机生成中心点-----------------------------------------
    def rand_centers(self, forbid_array, mask_array):
        """
        在允许区域内随机生成负样本中心点，并动态更新禁止区域。
        - 参数：
            - forbid_array: 初始禁止区域mask。
            - mask_array: 肺部mask。
        - 返回：
            - centers: 负样本中心点列表。
        - 流程:
            - 1. 从初始禁止区域复制出可动态更新的采样约束。
            - 2. 在CT范围内随机生成候选中心点。
            - 3. 过滤掉落在禁止区域或肺外区域的点。
            - 4. 每接受一个中心点后，在其周围扩展新的禁止区域，直到数量满足要求。
        """
        num = 0
        centers=[]
        forbid_region = forbid_array.copy()

        while(1):
            if num==self.centers_num:
                break
            
            #随机中心点
            randz = random.randint(0,forbid_region.shape[0]-1)
            randy = random.randint(0,forbid_region.shape[1]-1)
            randx = random.randint(0,forbid_region.shape[2]-1)
            if (forbid_region[randz, randy, randx]==True):
                continue
            if (mask_array[randz, randy, randx]==0):
                continue
            # 添加中心点
            center = (randz, randy, randx)
            centers.append(center)
            # 新增禁止区域
            z_start = max(0, randz-self.new_forbid_radius)
            z_end = min(forbid_region.shape[0], randz+self.new_forbid_radius)
            y_start = max(0, randy-self.new_forbid_radius)
            y_end = min(forbid_region.shape[1], randy+self.new_forbid_radius)
            x_start = max(0, randx-self.new_forbid_radius)
            x_end = min(forbid_region.shape[2], randx+self.new_forbid_radius)
            forbid_region[z_start:z_end, y_start:y_end, x_start:x_end] = True

            num += 1
        return centers


    #-----------------------------------生成正样本---------------------------------------------
    def create_true_sample_patches(self):
        """
        批量生成正样本patch，保存为npy并记录csv。
        - 参数：
            - 无显式参数，使用重采样目录和转换后的标注CSV生成样本。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 读取转换后的结节标注，并按seriesuid对结节分组。
            - 2. 逐个读取对应CT和mask，从结节中心裁剪固定大小的正样本区块。
            - 3. 将每个正样本保存为npy文件，并记录样本路径、标签和位置信息。
            - 4. 最终将全部正样本记录写入true_samples.csv。
        """
        # 正样本信息记录列表
        positive_sample_records = []
        # 构造路径
        transformed_annotations_path = os.path.join(self.output_dir, 'transformed_annotations.csv')
        # 读取csv文件名字及对应体素中心
        transformed_annotations = pd.read_csv(transformed_annotations_path)
        # 按seriesuid分组
        groups = transformed_annotations.groupby('seriesuid')

        for uid, group in tqdm(groups, desc="正在生成正样本：", ncols=100):
            # 构建CT和mask路径
            ct_resample_path = os.path.join(self.resampled_ct_dir, f"{uid}.mhd")
            mask_resample_path = os.path.join(self.resampled_mask_dir, f"{uid}.mhd")
            # 判断文件是否存在
            if (not os.path.exists(ct_resample_path)) or (not os.path.exists(mask_resample_path)) :
                print(f"CT或mask文件不存在,文件名:{uid}.mhd")
                continue
                
            # 读取CT和mask(3D)
            ct_array = self.load_data(ct_resample_path)[0]  # shape (D, H, W)
            mask_array = self.load_data(mask_resample_path)[0]  # shape (D, H, W)

            # 对同一个CT的多个结节生成正样本
            for index, (_,row) in enumerate(group.iterrows()):
                # 中心体素坐标 (voxel_z, voxel_y, voxel_x)
                z = int(row['voxel_z'])
                y = int(row['voxel_y'])
                x = int(row['voxel_x'])
                center = (z,y,x)

                # 检查是否在肺内，若不是则跳过

                # 裁剪区块
                patch = self.extract_3d_patch(ct_array, center, patch_size=self.patch_size)

                # 另起文件名保存为npy缓存
                file_name = f"{uid}_nodule_{index}"
                true_sample_patches_path = os.path.join(self.true_sample_patches_dir, f"{file_name}.npy")
                os.makedirs(os.path.dirname(true_sample_patches_path), exist_ok=True)
                np.save(true_sample_patches_path, patch)

                # 记录信息
                positive_sample_records.append({
                    'sample_id': file_name,
                    'path': true_sample_patches_path,
                    'label': 1,
                    'seriesuid': uid,
                    'center_z': z,
                    'center_y': y,
                    'center_x': x,
                    'diameter_mm': row['diameter_mm']
                })
        # 保存样本记录CSV
        samples_df = pd.DataFrame(positive_sample_records)
        samples_df.to_csv(self.true_sample_csv_path, index=False)
        print(f"正样本生成完成，共 {len(samples_df)} 个，patch_size={self.patch_size}")
        return


    #-----------------------------------生成负样本---------------------------------------------
    def create_false_sample_patches(self):
        """
        批量生成负样本patch，保存为npy并记录csv。
        - 参数：
            - 无显式参数，使用当前对象维护的重采样结果和标注信息。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 遍历全部重采样后的CT及对应mask文件。
            - 2. 根据结节标注生成禁止采样区域，避免负样本落在结节附近。
            - 3. 在肺区内随机采样负样本中心并裁剪区块，保存为npy文件。
            - 4. 将负样本写入false_samples.csv，并与正样本CSV合并成all_samples.csv。
        """
        # 临时参数
        ct_paths = []
        mask_paths = []
        false_sample_records = []      # 负样本信息记录列表

        # 1.构造重采样后的所有ct以及mask文件路径
        # 输入文件路径构造
        for ct_path in glob.glob(os.path.join(self.resampled_ct_dir, "*.mhd")):
            ct_paths.append(ct_path)
            mask_name = os.path.basename(ct_path)
            mask_path = os.path.join(self.resampled_mask_dir, mask_name)
            mask_paths.append(mask_path)

        # 2.读取转换坐标后的csv文件
        transformed_annotations = pd.read_csv(self.transformed_csv_path)
        # 3.循环进行单个CT的负样本生成
        for ct_path, mask_path in tqdm(zip(ct_paths, mask_paths), total=len(ct_paths), desc="正在生成负样本：", ncols=100):
            # 结节中心列表
            nodules_centers = []
            diameters = []
            # 读取CT和mask, shape(D, H, W)
            ct_array = self.load_data(ct_path)[0]
            mask_array = self.load_data(mask_path)[0]
            # 获取当前CT的uid
            uid = os.path.splitext(os.path.basename(ct_path))[0]
            # 判断当前ct的uid是否在csv文件中
            if uid in transformed_annotations['seriesuid'].values:
                # 有结节，读取多个结节数据
                ct_nodules = transformed_annotations[transformed_annotations['seriesuid'] == uid]
                for _, row in ct_nodules.iterrows():
                    nodules_centers.append((row['voxel_z'], row['voxel_y'], row['voxel_x']))
                    diameters.append(row['diameter_vox'])

                # 生成禁止区域
                forbid_array = self.generate_forbid_region(mask_array.shape, nodules_centers, diameters)
            else:
                # 无结节，则直接生成全false的禁止区域
                forbid_array = np.zeros(mask_array.shape, dtype=bool)  # 全False表示无禁止区域

            # 生成负样本中心点
            false_sample_centers = self.rand_centers(forbid_array, mask_array)
            # 循环裁剪所有负样本中心对应的区块，保存npy缓存，并记录csv信息
            for index, negative_center in enumerate(false_sample_centers):
                patch = self.extract_3d_patch(ct_array, negative_center, patch_size=self.patch_size)
                # 另起文件名保存为npy缓存
                file_name = f"{uid}_no_noudle_{index}"
                save_path = os.path.join(self.false_sample_patches_dir, f"{file_name}.npy")
                np.save(save_path, patch)

                # 记录信息
                false_sample_records.append({
                    'sample_id': file_name,
                    'path': save_path,
                    'label': 0,
                    'seriesuid': uid,
                    'center_z': negative_center[0],
                    'center_y': negative_center[1],
                    'center_x': negative_center[2],
                    'diameter_mm': -1
                })
        # 保存负样本记录CSV
        samples_df = pd.DataFrame(false_sample_records)
        samples_df.to_csv(self.false_sample_csv_path, index=False)
        print(f"负样本生成完成，共 {len(samples_df)} 个，patch_size={self.patch_size}")

        if not (os.path.exists(self.true_sample_csv_path) and os.path.exists(self.false_sample_csv_path)):
            print("警告：正负样本CSV文件不存在，无法合并")
            return
        #合并正负样本
        df1 = pd.read_csv(self.true_sample_csv_path)
        df2 = pd.read_csv(self.false_sample_csv_path)
        # 纵向合并（ignore_index 重置索引）
        merged = pd.concat([df1, df2], ignore_index=True)
        # 保存为新文件
        merged.to_csv(self.all_sample_csv_path, index=False)


    #-----------------------------------优化负样本生成------------------------------------------
    def create_optimized_false_sample_patches(self, ct_path, model_operator):
        """
        根据模型预测结果，将假阳性patch加入负样本池。
        - 参数：
            - ct_path: 原始的mhd文件路径。
            - model_operator: 已初始化的模型操作对象，用于复用当前模型进行预测。
        - 返回：
            - new_sample_count: 本次新增的优化负样本数量。
        - 过程：
            1. 对无结节CT滑动裁剪patch并预测。
            2. 预测为结节的patch视为假阳性，保存到optimizer_false_sample_npy_cache。
            3. 更新optimizer_false_samples.csv和all_samples.csv。
        """
        # 临时参数
        new_sample_count = 0
        optimizer_false_sample_records = []      # 负样本信息记录列表

        # 读取csv文件名字及对应体素中心
        transformed_annotations = pd.read_csv(self.transformed_csv_path)

        # 加载模型并预测
        if model_operator is None:
            raise ValueError("create_optimized_false_sample_patches 需要传入已初始化的 ModelOperate 对象")
        # 此处不使用mask过滤肺部外区域，部分纯白色会被识别为结节，需要优化--------------------------------------------------
        output_information = model_operator.predict(ct_path)

        # 判断传入的ct文件是否应该有结节,如果有结节则不生成优化负样本
        uid = os.path.splitext(os.path.basename(ct_path))[0]
        if uid in transformed_annotations['seriesuid'].values:
            print(f"\rCT {uid} 已有结节标注，跳过优化负样本生成", end='', flush=True)
            return 0
        
        # 根据预测结果和阈值筛选假阳性结节
        if output_information["nodule_nums"] ==0:
            # 无假阳性结节，跳过优化负样本生成
            return 0
        
        # 随机保存假阳性负样本，数量为one_ct_optimizer_false_sample_limit，避免过多假阳性样本导致训练集失衡
        indexs = random.sample(output_information['nodule_indexs'], min(len(output_information['nodule_indexs']), self.one_ct_optimizer_false_sample_limit))

        # 生成并保存假阳性负样本，并记录信息
        for index in indexs:
            # 取出index对应的区块patch
            patch = output_information['patches_array'][index]
            # 另起文件名保存为npy缓存,名字后续是假阳性区块的序号
            file_name = f"{uid}_no_nodule_optimizer_{index}"
            # 读取原本的csv文件，判断是否有同名文件，如果有则跳过
            if os.path.exists(self.optimizer_false_sample_csv_path):
                existing_csv = pd.read_csv(self.optimizer_false_sample_csv_path)
                if file_name in existing_csv['sample_id'].values:
                    continue

            # 保存假阳性负样本为npy文件
            optimizer_false_sample_patches_path = os.path.join(self.optimizer_false_sample_patches_dir, f"{file_name}.npy")
            np.save(optimizer_false_sample_patches_path, patch)

            # 记录信息
            optimizer_false_sample_records.append({
                'sample_id': file_name,
                'path': optimizer_false_sample_patches_path,
                'label': 0,
                'seriesuid': uid,
                'center_z':output_information['patch_centers'][index][0],
                'center_y': output_information['patch_centers'][index][1],
                'center_x': output_information['patch_centers'][index][2],
                'diameter_mm': -1
            })
            new_sample_count += 1

        # 保存增添负样本记录CSV
        if os.path.exists(self.optimizer_false_sample_csv_path):
            existing_csv = pd.read_csv(self.optimizer_false_sample_csv_path)
            current_csv = pd.DataFrame(optimizer_false_sample_records)
            # 合并
            df_combined = pd.concat([existing_csv, current_csv], ignore_index=True)
            df_combined.to_csv(self.optimizer_false_sample_csv_path, index=False)
        else:
            samples_df = pd.DataFrame(optimizer_false_sample_records)
            samples_df.to_csv(self.optimizer_false_sample_csv_path, index=False)

        # 合并新的假阳性负样本到总样本CSV
        if not os.path.exists(self.all_sample_csv_path):
            print("警告：总样本CSV文件不存在，无法合并优化负样本")
            return 0
        df_all = pd.read_csv(self.all_sample_csv_path)
        df_optimizer = pd.DataFrame(optimizer_false_sample_records)
        merged = pd.concat([df_all, df_optimizer], ignore_index=True)
        merged.to_csv(self.all_sample_csv_path, index=False)
        print(f"\n该CT预测共有{output_information['nodule_nums']}个结节，新增 {len(optimizer_false_sample_records)} 个假阳性负样本，已保存并合并到总样本CSV")

        return new_sample_count


    #-----------------------------------批量优化假阳性样本生成-----------------------------------
    def batch_optimizer_false_samples(self, model_operator):
        """
        批量生成假阳性负样本，提升模型性能。
        - 参数：
            - model_operator: 已初始化的模型操作对象，用于复用当前模型进行预测。
        - 返回：
            - 无返回值。
        - 流程:
            - 1. 收集全部重采样后的CT和mask路径。
            - 2. 读取转换后的标注信息，筛选无结节CT。
            - 3. 对每个无结节CT调用优化负样本提取逻辑。
            - 4. 累计并输出新增的假阳性负样本数量。
        """
        # 1.构造重采样后的所有ct以及mask文件路径
        new_sample_count = 0
        ct_paths = []
        mask_paths = []
        # 输入文件路径构造
        for ct_path in glob.glob(os.path.join(self.resampled_ct_dir, "*.mhd")):
            ct_paths.append(ct_path)
            mask_name = os.path.basename(ct_path)
            mask_path = os.path.join(self.resampled_mask_dir, mask_name)
            mask_paths.append(mask_path)

        # 2.读取转换坐标后的csv文件
        transformed_annotations = pd.read_csv(self.transformed_csv_path)
        # 3.循环读取每个ct文件和对应的mask文件，提取假阳性样本并加入训练集
        for ct_path, mask_path in tqdm(zip(ct_paths, mask_paths), total=len(ct_paths), desc="优化假阳性样本训练生成中", ncols=100):
            # 单次数量限制，避免一次性生成过多假阳性样本导致训练集失衡
            if new_sample_count >= self.batch_ct_optimizer_false_sample_limit:
                print(f"\r已达到假阳性样本数量限制 {self.batch_ct_optimizer_false_sample_limit}，停止生成", end='', flush=True)
                break

            # 判断当前ct文件是否应该有结节
            uid = os.path.basename(ct_path).replace('.mhd', '')
            if uid in transformed_annotations['seriesuid'].values:
                continue
            # 读取ct和mask
            new_sample_count += self.create_optimized_false_sample_patches(ct_path, model_operator)

        print(f"优化假阳性样本生成完成！新增 {new_sample_count} 个假阳性负样本")


    #-----------------------------------划分样本数据集------------------------------------------
    def divide_samples_dataset(self, train_size=0.8):
        """
        划分样本csv为训练集和验证集，返回DataLoader。
        - 参数：
            - train_size: 训练集比例。
        - 返回：
            - train_dataloader: 训练集DataLoader。
            - val_dataloader: 验证集DataLoader。
            - weight_tensor: 正负样本权重张量，用于损失函数。
        - 过程：
            1. 随机划分数据集。
            2. 自动计算类别权重。
        """

        true_sample_dataset = PatchDataset(self.true_sample_csv_path, transform=self.transform)
        false_sample_dataset = PatchDataset(self.false_sample_csv_path, transform=self.transform)
        if os.path.exists(self.optimizer_false_sample_csv_path):
            optimizer_false_dataset = PatchDataset(self.optimizer_false_sample_csv_path, transform=self.transform)

        # 合并正负样本并划分出一个验证集
        if os.path.exists(self.optimizer_false_sample_csv_path):
            true_and_false_sample_dataset = ConcatDataset([true_sample_dataset, false_sample_dataset, optimizer_false_dataset])
        else:
            true_and_false_sample_dataset = ConcatDataset([true_sample_dataset, false_sample_dataset])
        # 比例设置
        dataset_size = len(true_and_false_sample_dataset)
        train_size = int(train_size*dataset_size)
        val_size = dataset_size - train_size
        # 划分
        train_dataset, val_dataset = random_split(true_and_false_sample_dataset, [train_size, val_size])

        # 加载为dataloader
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, shuffle=False, num_workers=0 , pin_memory=True)

        # 计算正负样本权重
        train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
        neg_count = train_labels.count(0)
        pos_count = train_labels.count(1)
        if pos_count == 0:
            pos_weight = 1.0
        else:
            # 计算正样本权重，限制最大值避免过大权重导致训练不稳定，乘以系数增加正样本权重以缓解类别不平衡
            pos_weight = min(neg_count / pos_count, 20.0)*1.3
        
        # 生成权重张量，格式为 [负样本权重, 正样本权重]
        weight_tensor = torch.tensor([1.0, pos_weight])

        return train_dataloader, val_dataloader, weight_tensor


    #-----------------------------------滑动获取区块--------------------------------------------
    def slide_getpatch(self, ct_data, mask_data=None):
        """
        输入原始CT和mask，重采样后滑动裁剪patch。
        - 参数：
            - ct_data: 原始CT数据，可以是mhd文件路径或已经重采样后的numpy数组。
            - mask_data: 原始mask数据，可以是mhd文件路径或已经重采样后的numpy数组，可以为空，但会增加无关的patch，速度减慢。
        - 返回：
            - patches_array: (N,Z,H,W)的patch数组。
            - centers: patch中心坐标列表。
        - 流程:
            - 1. 判断输入类型，必要时先读取并重采样CT与mask。
            - 2. 根据patch大小和步长在三维空间内滑动生成候选中心。
            - 3. 仅保留位于肺部mask内部的中心点。
            - 4. 根据中心点裁剪所有区块，并返回区块数组及中心坐标列表。
        """
        # 读取-ct
        if isinstance(ct_data, str):
            ct_array, ct_information = self.load_data(ct_data)
            # 重采样
            resampled_ct = self.resample_array(ct_array, ct_information)
        elif isinstance(ct_data, np.ndarray):
            resampled_ct = ct_data
        else:
            raise ValueError("输入数据类型不正确，应为mhd文件路径或numpy数组。")
        
        # 读取-mask
        if isinstance(mask_data, str):
            mask_array, _ = self.load_data(mask_data)
            # 重采样
            resampled_mask = self.resample_array(mask_array, ct_information)
        elif isinstance(mask_data, np.ndarray):
            resampled_mask = mask_data
        else:
            resampled_mask = np.ones_like(resampled_ct, dtype=bool)


        # 参数计算
        D, H, W = resampled_ct.shape
        half = self.patch_size//2
        stride = int(self.patch_size/2)
        centers = []
        patches = []

        # 生成待裁剪区块中心
        for z in range(half, D - half, stride):
            for y in range(half, H - half, stride):
                for x in range(half, W - half, stride):

                    # 只在肺内采样
                    if resampled_mask[z, y, x] > 0:
                        centers.append((z, y, x))

        # 裁剪区块
        for center in centers:
            patches.append(self.extract_3d_patch(resampled_ct, center))
        patches_array = np.stack(patches, axis=0)

        return patches_array,centers


    #-----------------------------------获取DataLoader-----------------------------------------
    def get_dataloader(self, patches):
        """
        将patch数组标准化后包装为DataLoader。
        - 参数：
            - patches: (N,Z,H,W)的patch数组。
        - 返回：
            - data_loader: 标准化后的DataLoader。
        - 流程:
            - 1. 按样本分别计算均值和标准差并完成标准化。
            - 2. 将numpy数组转换为张量，并补充通道维度。
            - 3. 构造TensorDataset和DataLoader，便于后续批量预测。
        """
        # 标准化
        mean = patches.mean(axis=(1, 2, 3), keepdims=True)
        std = patches.std(axis=(1, 2, 3), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        patches_normalized = (patches - mean) / std

        # 转为tensor并添加通道维
        patches_tensor = torch.from_numpy(patches_normalized)
        if patches_tensor.ndim == 4:
            patches_tensor = patches_tensor.unsqueeze(1)
        else:
            raise ValueError(f"输入的patches数组维度不正确，期望4维(N,Z,H,W)，但得到{patches_tensor.ndim}维")
        
        # 创建DataLoader
        dataset = TensorDataset(patches_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return data_loader


    #-----------------------------------数据增强------------------------------------------------
    def transform(self, patch):
        """
        数据增强的函数
        - 参数：
            - patch: 输入的4D区块，形状为 (1, Z, H, W)。
        """
        # 随机翻转
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=3)  # 翻转 x 轴
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=2)  # 翻转 y 轴
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1)  # 翻转 z 轴

        # 随机添加高斯噪声
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, patch.shape)
            patch = patch + noise
        
        # 随机旋转
        if np.random.rand() > 0.5:
            axes = [(1, 2), (1, 3), (2, 3)]
            axis = axes[np.random.randint(0, len(axes))]
            patch = np.rot90(patch, k=np.random.randint(1, 4), axes=axis)
        
        # 随机调整亮度
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            patch = patch * factor
        
        # 随机调整对比度
        if np.random.rand() > 0.5:
            mean = patch.mean()
            factor = np.random.uniform(0.9, 1.1)
            patch = (patch - mean) * factor + mean
        
        return patch






if __name__ == "__main__":
    """使用示例"""
    # 全局变量
    dataset_dir = r"D:\learnfile\dataset\LUNA16"
    output_dir = r"D:\Learnfile\Dataset\LUNA16\sample_data"
    data_processor = DataProcessor(dataset_dir, output_dir)

    """重采样以及坐标转换测试"""
    data_processor.batch_resample()
    data_processor.transform_coord()
    
    """生成样本区块测试"""
    data_processor.create_true_sample_patches()
    data_processor.create_false_sample_patches()
    # batch_optimizer_false_samples需要在有训练好模型的ModelOperate对象后才能测试

    """划分数据集并加载测试"""
    # train_loader, val_loader = data_processor.divide_samples_dataset()






