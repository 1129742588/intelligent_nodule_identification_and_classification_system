"""
结节分割推理流程文件。

设计目标：
- 仅负责结节分割（不包含结节识别）。
- 输入已有结节坐标，输出分割后的结节 mask。
- 支持保存全尺寸 mask 与分类用 patch/mask。
- 全流程以类方法组织，不使用独立函数。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import napari
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from .model import EnhancedDenseVNet
# from fenge.predict_utils import EnhancedDenseVNet


class NoduleSegmentationPredictor:
    """
    结节分割推理器（坐标驱动）。

    - 参数:
        - model_path: 分割模型权重路径；为空时使用默认路径。
        - save_root: 输出根目录；为空时使用默认缓存目录。
        - crop_size: 输入模型的立方体 patch 尺寸，默认 96。
        - threshold: 分割概率阈值，默认 0.5。
        - fill_value: 裁剪越界填充值，默认 -1000.0（CT空气值近似）。
        - new_spacing: 重采样目标 spacing，默认 (1.0, 1.0, 1.0)。
        - device: 推理设备，默认自动选择 cuda/cpu。
    - 返回:
        - 类实例本身。
    - 用法:
        - predictor = NoduleSegmentationPredictor()
        - result = predictor.segment_from_coordinate(ct_input=ct_path, center_zyx=(z, y, x))
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        save_root: str | Path | None = None,
        fill_value: float = -1000.0,
        new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: str | torch.device | None = None,
    ) -> None:
        

        self.project_root = Path(__file__).resolve().parents[1]

        default_model_path = os.path.join(os.path.dirname(__file__), "model_data", "nodule_segmentation_model.pth")
        default_save_root = os.path.join(self.project_root, "temp_cache", "nodules_segmentation_cache")

        self.model_path = Path(model_path) if model_path is not None else default_model_path
        self.save_root = Path(save_root) if save_root is not None else default_save_root

        self.crop_size = 96
        self.voxel_threshold = 10       # 前景像素数量阈值，低于该值的 patch 将被视为无效
        self.fill_value = float(fill_value)
        self.new_spacing = tuple(float(v) for v in new_spacing)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


        os.makedirs(self.save_root, exist_ok=True)

    def segment_from_coordinate(
        self,
        ct_input: str | Path | np.ndarray,
        center_zyx: Tuple[float, float, float] | Sequence[Tuple[float, float, float]],
    ) -> List[Dict[str, Any]]:
        """
        根据“已知结节坐标”执行结节分割。

        - 参数:
            - ct_input: CT 输入，可为
                - 文件路径（mhd/dicom 文件），
                - DICOM 序列目录路径，
                - 已加载的 3D numpy 数组（默认视为已重采样）。
            - center_zyx: 重采样空间下的结节中心坐标。
                - 单个中心: (z, y, x)
                - 多个中心: [(z1, y1, x1), (z2, y2, x2), ...]
        - 返回:
            - result_list: 列表，每个元素是一个结节字典，仅包含：
                - patch_center: 生效后的结节中心坐标 (z, y, x)
                - patch: 归一化和标准化后的 patch
                - patch_mask: patch 级别二值 mask
                - full_mask: 回填后的全尺寸二值 mask
        - 流程:
            - 1. 加载 CT，并在必要时重采样到统一 spacing。
            - 2. 将输入坐标归一化为中心列表，逐个约束到体积范围内。
            - 3. 逐个中心裁剪 patch 并推理 patch mask。
            - 4. 回填 full mask，返回结果列表。
        """
        # 1. 加载 CT；若输入为路径则重采样，若输入为数组则默认已重采样
        resampled_ct, _ = self._prepare_ct(ct_input)

        centers_list, _ = self._normalize_centers(center_zyx)
        if len(centers_list) == 0:
            raise ValueError("center_zyx 不能为空，至少需要一个中心坐标")

        result_list: List[Dict[str, Any]] = []

        for one_center in tqdm(centers_list, desc="分割结节进行中", ncols=100):
            center_resampled = (
                int(round(one_center[0])),
                int(round(one_center[1])),
                int(round(one_center[2])),
            )
            center_effective = self._clamp_center_to_volume(center_resampled, resampled_ct.shape)

            patch, roi_box, patch_offsets = self.extract_3d_patch(
                image_array=resampled_ct,
                center=center_effective,
                patch_size=self.crop_size,
            )
            processed_patch = self._preprocess_patch(patch)
            patch_mask = self._segment_patch(processed_patch)

            # 前景像素判断
            if np.sum(patch_mask) < self.voxel_threshold:
                print(f"中心坐标 {center_effective} 的 patch 前景像素数量 {np.sum(patch_mask)} 低于阈值 {self.voxel_threshold}，跳过该结节")
                continue


            full_mask = self._place_patch_mask_back(patch_mask, resampled_ct.shape, roi_box, patch_offsets)

            result_list.append(
                {
                    "patch_center": tuple(int(v) for v in center_effective),
                    "patch": processed_patch,
                    "patch_mask": patch_mask,
                    "full_mask": full_mask,
                }
            )

        return result_list

    def _normalize_centers(
        self,
        center_zyx: Tuple[float, float, float] | Sequence[Tuple[float, float, float]],
    ) -> Tuple[List[Tuple[float, float, float]], bool]:
        """
        统一中心坐标输入格式，兼容单中心和多中心列表。

        - 参数:
            - center_zyx: 单个中心或中心列表。
        - 返回:
            - centers_list: 标准化后的中心列表。
            - is_batch_input: 是否原始批量输入。
        """
        if isinstance(center_zyx, tuple) and len(center_zyx) == 3:
            return [center_zyx], False

        if isinstance(center_zyx, Sequence):
            centers_list: List[Tuple[float, float, float]] = []
            for one_center in center_zyx:
                if not (isinstance(one_center, Sequence) and len(one_center) == 3):
                    raise ValueError("批量中心坐标格式错误，必须是 [(z,y,x), ...]")
                centers_list.append((float(one_center[0]), float(one_center[1]), float(one_center[2])))
            return centers_list, True

        raise ValueError("center_zyx 格式错误，需为 (z,y,x) 或 [(z,y,x), ...]")

    def _clamp_center_to_volume(
        self,
        center_zyx: Tuple[int, int, int],
        volume_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """
        将中心点约束到体积范围内，避免越界裁剪。

        - 参数:
            - center_zyx: 原始中心坐标。
            - volume_shape: 体积形状 (D,H,W)。
        - 返回:
            - 约束后的中心坐标 (z,y,x)。
        """
        cz, cy, cx = center_zyx
        dz, dy, dx = volume_shape
        cz = min(max(int(cz), 0), max(0, dz - 1))
        cy = min(max(int(cy), 0), max(0, dy - 1))
        cx = min(max(int(cx), 0), max(0, dx - 1))
        return cz, cy, cx

    def extract_3d_patch(self, image_array, center, patch_size=96):
        """
        从3D数组中裁剪立方体patch，边界外填充指定值。
        - 参数：
            - image_array: 3D数组。
            - center: patch中心坐标（z,y,x）。
            - patch_size: patch边长，若为None则使用对象默认的patch_size。
        - 返回：
            - patch: 裁剪后的patch数组。
            - roi_box: 原图有效区域范围 (z0,z1,y0,y1,x0,x1)。
            - patch_offsets: patch有效区域偏移 (pz0,pz1,py0,py1,px0,px1)。
        - 流程:
            - 1. 根据中心点和边长计算目标区块在原图中的起止位置。
            - 2. 计算与原图相交的有效区域范围。
            - 3. 创建填充值初始化的目标立方体区块。
            - 4. 将原图中的有效区域复制到目标区块的对应位置。
        """
        if patch_size==None:
            raise ValueError("patch_size不能为None，请提供有效的整数值")


        cz, cy, cx = self._clamp_center_to_volume(center, image_array.shape)
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

        roi_box = (z_src_start, z_src_end, y_src_start, y_src_end, x_src_start, x_src_end)
        patch_offsets = (z_tgt_start, z_tgt_end, y_tgt_start, y_tgt_end, x_tgt_start, x_tgt_end)
        return patch, roi_box, patch_offsets

    def _prepare_ct(self, ct_input: str | Path | np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取并准备 CT 数据。

        - 参数:
            - ct_input: 路径或 numpy 数组。
        - 返回:
            - resampled_ct: 重采样后的 3D 数组。
            - sitk_info: 原始图像空间信息字典。
        """
        if isinstance(ct_input, np.ndarray):
            if ct_input.ndim != 3:
                raise ValueError("ct_input 为 numpy 数组时必须是 3D 形状 (D, H, W)")
            sitk_info = {
                "spacing": self.new_spacing,
                "origin": (0.0, 0.0, 0.0),
                "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                "size": (int(ct_input.shape[2]), int(ct_input.shape[1]), int(ct_input.shape[0])),
            }
            return ct_input.astype(np.float32, copy=False), sitk_info

        ct_path = Path(ct_input)
        ct_array, sitk_info = self._load_data(ct_path)
        resampled_ct = self._resample_array(ct_array, sitk_info)
        return resampled_ct.astype(np.float32, copy=False), sitk_info

    def _load_data(self, data_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        加载单文件或 DICOM 序列目录。

        - 参数:
            - data_path: 文件路径或目录路径。
        - 返回:
            - image_array: 3D numpy 数组 (z, y, x)。
            - sitk_info: spacing/size/direction/origin 信息。
        """
        if data_path.is_file():
            image = sitk.ReadImage(str(data_path))
        elif data_path.is_dir():
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(data_path))
            if not series_ids:
                raise ValueError(f"目录中未找到 DICOM 序列: {data_path}")
            file_names = reader.GetGDCMSeriesFileNames(str(data_path), series_ids[0])
            reader.SetFileNames(file_names)
            image = reader.Execute()
        else:
            raise ValueError(f"输入路径不存在或无效: {data_path}")

        image_array = sitk.GetArrayFromImage(image)
        if image_array.ndim == 2:
            image_array = image_array[np.newaxis, ...]

        sitk_info = {
            "spacing": image.GetSpacing(),
            "size": image.GetSize(),
            "direction": image.GetDirection(),
            "origin": image.GetOrigin(),
        }
        return image_array, sitk_info

    def _resample_array(self, image_array: np.ndarray, sitk_info: Dict[str, Any]) -> np.ndarray:
        """
        将输入 3D 数组重采样到统一 spacing（默认 1mm）。

        - 参数:
            - image_array: 原始 3D 数组。
            - sitk_info: 原始空间信息。
        - 返回:
            - 重采样后的 3D 数组。
        """
        image = sitk.GetImageFromArray(image_array)
        image.SetSpacing(tuple(sitk_info["spacing"]))
        image.SetOrigin(tuple(sitk_info["origin"]))
        image.SetDirection(tuple(sitk_info["direction"]))

        new_size = [
            int(round(size * spacing / new_spacing))
            for size, spacing, new_spacing in zip(sitk_info["size"], sitk_info["spacing"], self.new_spacing)
        ]

        resampled_image = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            self.new_spacing,
            image.GetDirection(),
        )
        return sitk.GetArrayFromImage(resampled_image)

    def _preprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        patch 预处理（标准化 + 归一化）。

        - 参数:
            - patch: 原始 patch。
        - 返回:
            - 预处理后的 patch（float32, 值范围约 [0,1]）。
        """
        patch = patch.astype(np.float32, copy=False)
        patch = (patch - np.mean(patch)) / np.std(patch)
        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch) + 1e-8)
        return patch.astype(np.float32, copy=False)
    


    def _segment_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        对单个 patch 执行分割推理。

        - 参数:
            - patch: 预处理后的 patch，形状 (D,H,W)。
        - 返回:
            - patch_mask: 二值 mask，形状 (D,H,W)。
        """
        model = EnhancedDenseVNet()
        # 加载权重
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        # 移动到设备
        model = model.to(self.device)
        model.eval()

        # 转换为tensor并添加batch和通道维度
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        patch_tensor = patch_tensor.to(self.device)
        
        # 预测

        with torch.no_grad():
            output = model(patch_tensor)
        
        # 转换为numpy并二值化
        mask = output.cpu().numpy()[0, 0]
        mask = (mask > 0.5).astype(np.float32)

        return mask

    def _place_patch_mask_back(
        self,
        patch_mask: np.ndarray,
        full_shape: Tuple[int, int, int],
        roi_box: Tuple[int, int, int, int, int, int],
        patch_offsets: Tuple[int, int, int, int, int, int],
    ) -> np.ndarray:
        """
        将 patch mask 回填到全尺寸 CT 坐标系。

        - 参数:
            - patch_mask: patch 级二值 mask。
            - full_shape: 全尺寸体积形状。
            - roi_box: 原图有效区域范围。
            - patch_offsets: patch 有效区域偏移。
        - 返回:
            - full_mask: 全尺寸二值 mask。
        """
        rz0, rz1, ry0, ry1, rx0, rx1 = roi_box
        pz0, pz1, py0, py1, px0, px1 = patch_offsets

        full_mask = np.zeros(full_shape)
        full_mask[rz0:rz1, ry0:ry1, rx0:rx1] = patch_mask[pz0:pz1, py0:py1, px0:px1]
        return full_mask

    def _save_classification_data(
        self,
        case_id: str,
        coord_tag: str,
        patch: np.ndarray,
        patch_mask: np.ndarray,
    ) -> Dict[str, str]:
        """
        保存后续分类需要的 patch 与 patch-mask。

        - 参数:
            - case_id: 病例标识。
            - coord_tag: 坐标标签（z_y_x）。
            - patch: 结节 patch。
            - patch_mask: 对应 patch mask。
        - 返回:
            - 路径字典。
        """
        case_dir = os.path.join(self.save_root, case_id)
        nodules_dir = os.path.join(case_dir, "nodules")
        masks_dir = os.path.join(case_dir, "masks")

        # 确保目录存在
        os.makedirs(nodules_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        patch_path = os.path.join(nodules_dir, f"{case_id}_{coord_tag}_patch.npy")
        mask_path = os.path.join(masks_dir, f"{case_id}_{coord_tag}_patch_mask.npy")

        np.save(patch_path, patch.astype(np.float32))
        np.save(mask_path, patch_mask)

        return {
            "classification_patch": str(patch_path),
            "classification_mask": str(mask_path),
        }
