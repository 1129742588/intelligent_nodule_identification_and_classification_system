

import os
import glob
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from tqdm import tqdm


class LUNA16_dataload(Dataset):
    """
    LUNA16 2D切片分割数据集类。
    自动遍历所有subset下的mhd文件，按z轴切片返回2D图像和对应的肺mask。
    - 返回：
        - dataset对象，需用DataLoader加载
    """
    def __init__(self, dataset_dir, img_size=512, transform=None, use_mask=True, cache_npy=True):
        """
        - 参数：
            - data_dir: LUNA16数据集根目录，应包含 subset0/ 等子文件夹及 seg-lungs-LUNA16/ 文件夹
            - img_size: 输出图像尺寸（resize后）
            - transform: 可选的数据增强函数
            - use_mask: 是否加载mask
            - cache_npy: 是否将数据缓存为npy加速后续读取
        """
        self.img_size = img_size
        self.transform = transform
        self.use_mask = use_mask
        self.cache_npy = cache_npy
        self.ct_paths = []
        self.mask_paths = []
        self.slice_index = []  # 每个切片对应的 (ct_idx, z_idx)

        # 缓存目录
        cache_dir = os.path.join(dataset_dir, 'npy_cache')
        os.makedirs(cache_dir, exist_ok=True)

        # 收集所有mhd文件
        for subset in glob.glob(os.path.join(dataset_dir, 'subset*')):
            for mhd_path in glob.glob(os.path.join(subset, '*.mhd')):
                self.ct_paths.append(mhd_path)
                if use_mask:
                    mask_name = os.path.basename(mhd_path)
                    mask_path = os.path.join(dataset_dir, 'seg-lungs-LUNA16', mask_name)
                    self.mask_paths.append(mask_path)

        # 建立所有切片的索引
        for i, ct_path in enumerate(self.ct_paths):
            npy_ct = os.path.join(cache_dir, os.path.basename(ct_path) + '.ct.npy')
            if use_mask:
                npy_mask = os.path.join(cache_dir, os.path.basename(ct_path) + '.mask.npy')

            # 若缓存不存在则生成
            if self.cache_npy and (not os.path.exists(npy_ct) or (use_mask and not os.path.exists(npy_mask))):
                ct_img = sitk.ReadImage(ct_path)
                ct_arr = sitk.GetArrayFromImage(ct_img)
                np.save(npy_ct, ct_arr)
                if use_mask and os.path.exists(self.mask_paths[i]):
                    mask_img = sitk.ReadImage(self.mask_paths[i])
                    mask_arr = sitk.GetArrayFromImage(mask_img)
                    np.save(npy_mask, mask_arr)

            # 获取切片数量
            if self.cache_npy and os.path.exists(npy_ct):
                ct_arr = np.load(npy_ct, mmap_mode='r')
                n_slices = ct_arr.shape[0]
            else:
                ct_img = sitk.ReadImage(ct_path)
                n_slices = ct_img.GetSize()[2]

            for z in range(n_slices):
                self.slice_index.append((i, z))

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        ct_idx, z_idx = self.slice_index[idx]
        ct_path = self.ct_paths[ct_idx]
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(ct_path)), 'npy_cache')
        npy_ct = os.path.join(cache_dir, os.path.basename(ct_path) + '.ct.npy')

        if self.cache_npy and os.path.exists(npy_ct):
            ct_arr = np.load(npy_ct, mmap_mode='r')
        else:
            ct_img = sitk.ReadImage(ct_path)
            ct_arr = sitk.GetArrayFromImage(ct_img)

        img2d = ct_arr[z_idx]
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-8)

        if self.use_mask:
            mask_path = self.mask_paths[ct_idx]
            npy_mask = os.path.join(cache_dir, os.path.basename(ct_path) + '.mask.npy')
            if self.cache_npy and os.path.exists(npy_mask):
                mask_arr = np.load(npy_mask, mmap_mode='r')
            elif os.path.exists(mask_path):
                mask_img = sitk.ReadImage(mask_path)
                mask_arr = sitk.GetArrayFromImage(mask_img)
            else:
                mask_arr = np.zeros_like(ct_arr)
            mask2d = mask_arr[z_idx]
            mask2d = (mask2d > 0).astype(np.int64)
        else:
            mask2d = np.zeros_like(img2d, dtype=np.int64)

        # resize到统一尺寸
        img2d = cv2.resize(img2d, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask2d = cv2.resize(mask2d, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img2d = img2d[np.newaxis, ...]  # 增加通道维度

        if self.transform:
            img2d, mask2d = self.transform(img2d, mask2d)

        return torch.from_numpy(img2d).float(), torch.from_numpy(mask2d).long()
    

class DataProcessor:
    """
    数据处理类，负责加载LUNA16数据集并创建DataLoader。
    """
    #------------------------------------------------------------------------------------
    def __init__(self, dataset_dir):
        """
        - 参数:
            - dataset_dir: LUNA16数据集根目录，应包含 subset*/ 和 seg-lungs-LUNA16/ 文件夹
        """
        # 参数赋值





        # 各个路径构建
        self.dataset_dir = dataset_dir




        # 必要输出目录创建

    #------------------------------------------------------------------------------------
    def load_data(self,data_path):
        """
        可单文件或文件夹读取，返回(N,H,W)numpy数组
        - 参数:
            - data_path: 单个(mhd/dicom)文件路径或包含dicom文件的文件夹路径
        - 返回:
            - image_array: (N, H, W)的numpy数组，N为切片数，H和W为图像尺寸
        """
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
            image = reader.Execute()  # 返回3D图像

            # 转换为numpy数组，维度顺序为 (z, y, x)
            image_array = sitk.GetArrayFromImage(image)   # (切片数, 高度, 宽度)

        else:
            raise ValueError("输入路径既不是文件也不是文件夹，可能不存在该路径")
        
        return image_array

    
