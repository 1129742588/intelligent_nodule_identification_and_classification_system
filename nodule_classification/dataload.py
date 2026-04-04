from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pickle
import xml.etree.ElementTree as ET

import napari
import numpy as np
import pydicom
from tqdm import tqdm


class LIDCNoduleParser:
    """
    LIDC-IDRI 解析器（XML + DICOM）。

    `parse()` 的返回结构为 `List[scan_dict]`，每个 `scan_dict` 的核心字段：
    - `xml_path: str`
    - `patient_id: Optional[str]`
    - `study_uid: Optional[str]`
    - `series_uid: str`
    - `volume_shape: Tuple[int, int, int]`，格式为 `(depth, height, width)`
    - `nodules: List[nodule_dict]`

    每个 `nodule_dict` 的核心字段：
    - `nodule_index: int`
    - `n_annotations: int`（该结节由多少医生标注融合而来）
    - `voxel_count: int`
    - `bbox_xyzwhd: Tuple[int, int, int, int, int, int]`，格式为 `(x, y, z, w, h, d)`
    - `centroid_zyx: Tuple[float, float, float]`
    - `mask_crop: np.ndarray[bool]`
    - `hu_crop: np.ndarray[float32]`
    - `features: Dict[str, Optional[int]]`（XML 里的特征分级，仍为原始分级）

    常用访问示例：
    - `results = parser.parse()`
    - `first_scan = results[0]`
    - `first_nodule = first_scan["nodules"][0]`
    - `hu = first_nodule["hu_crop"]`
    - `calc = first_nodule["features"].get("calcification")`
    """

    def __init__(self, dataset_dir: str, output_dir: str) -> None:
        """
        初始化解析器与缓存目录。

        - 参数:
          - dataset_dir: LIDC 数据集根目录，通常包含 LIDC-IDRI-xxxx 病例目录。
          - output_dir: 输出目录，用于保存病例级缓存（`output_dir/cache/*.pkl`）。
        - 返回:
          - None
        - 流程:
          - 保存路径参数。
          - 构造 cache 目录。
          - 初始化运行时缓存。
        """
        self.dataset_dir = Path(dataset_dir)
        if output_dir is None:
          self.output_dir = None
          self.cache_dir = None
        else:
          self.output_dir = output_dir
          self.cache_dir = os.path.join(self.output_dir, "cache")
        self.cache_version = 4
        self._dicom_count_cache: Dict[Path, int] = {}

    @staticmethod
    def _strip_ns(tag: str) -> str:
        """
        - 参数:
          - tag: XML 标签字符串。
        - 返回:
          - 去除命名空间后的标签名。
        - 流程:
          - 检查是否包含命名空间分隔符 `}`。
          - 若存在则截取后半段。
        """
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @classmethod
    def _child_text(cls, element: ET.Element, name: str) -> Optional[str]:
        """
        - 参数:
          - element: 父节点。
          - name: 子节点名。
        - 返回:
          - 匹配子节点的文本，若不存在则返回 None。
        - 流程:
          - 遍历子节点。
          - 按去命名空间后的标签匹配。
          - 读取并清理文本。
        """
        for child in list(element):
            if cls._strip_ns(child.tag) == name:
                if child.text is None:
                    return None
                return child.text.strip()
        return None

    @classmethod
    def _find_children(cls, element: ET.Element, name: str) -> List[ET.Element]:
        """
        - 参数:
          - element: 父节点。
          - name: 目标子节点名。
        - 返回:
          - 目标子节点列表。
        - 流程:
          - 遍历全部子节点。
          - 保留标签名匹配项。
        """
        return [child for child in list(element) if cls._strip_ns(child.tag) == name]

    @staticmethod
    def _safe_int(text: Optional[str]) -> Optional[int]:
        """
        - 参数:
          - text: 待转换文本。
        - 返回:
          - 成功则返回 int，失败返回 None。
        - 流程:
          - 空值直接返回 None。
          - 尝试 int 转换并捕获异常。
        """
        if text is None or text == "":
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _safe_float(text: Optional[str]) -> Optional[float]:
        """
        - 参数:
          - text: 待转换文本。
        - 返回:
          - 成功则返回 float，失败返回 None。
        - 流程:
          - 空值直接返回 None。
          - 尝试 float 转换并捕获异常。
        """
        if text is None or text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _looks_like_dicom(path: Path) -> bool:
        """
        - 参数:
          - path: 文件路径。
        - 返回:
          - 是否看起来是 DICOM 文件。
        - 流程:
          - 先按常见扩展名判断。
          - 对无扩展名文件读取 DICM 头判断。
        """
        suffix = path.suffix.lower()
        if suffix in {".dcm", ".dicom", ".ima"}:
            return True

        if suffix not in {"", ".img"}:
            return False

        try:
            with path.open("rb") as f:
                header = f.read(132)
            return len(header) >= 132 and header[128:132] == b"DICM"
        except OSError:
            return False

    @staticmethod
    def _polygon_to_mask(height: int, width: int, points_xy: Sequence[Tuple[int, int]]) -> np.ndarray:
        """
        - 参数:
          - height: 图像高。
          - width: 图像宽。
          - points_xy: 多边形顶点坐标列表。
        - 返回:
          - 2D 布尔 mask。
        - 流程:
          - 计算多边形包围框。
          - 在局部网格上做射线法点内判断。
          - 写回整幅二维 mask。
        """
        if len(points_xy) < 3:
            return np.zeros((height, width), dtype=bool)

        polygon = np.array(points_xy, dtype=np.float32)
        xs = polygon[:, 0]
        ys = polygon[:, 1]

        min_x = max(0, int(np.floor(xs.min())))
        max_x = min(width - 1, int(np.ceil(xs.max())))
        min_y = max(0, int(np.floor(ys.min())))
        max_y = min(height - 1, int(np.ceil(ys.max())))

        if min_x > max_x or min_y > max_y:
            return np.zeros((height, width), dtype=bool)

        grid_x, grid_y = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float32),
            np.arange(min_y, max_y + 1, dtype=np.float32),
        )
        test_x = grid_x + 0.5
        test_y = grid_y + 0.5

        x1 = polygon[:, 0]
        y1 = polygon[:, 1]
        x2 = np.roll(x1, -1)
        y2 = np.roll(y1, -1)

        inside = np.zeros(test_x.shape, dtype=bool)
        for idx in range(len(polygon)):
            intersects = ((y1[idx] > test_y) != (y2[idx] > test_y))
            x_intersect = (x2[idx] - x1[idx]) * (test_y - y1[idx]) / (y2[idx] - y1[idx] + 1e-8) + x1[idx]
            inside ^= intersects & (test_x < x_intersect)

        mask = np.zeros((height, width), dtype=bool)
        mask[min_y:max_y + 1, min_x:max_x + 1] = inside
        return mask

    @staticmethod
    def _to_bbox_from_slice_masks(slice_masks: Dict[int, np.ndarray]) -> Optional[Dict[str, int]]:
        """
        - 参数:
          - slice_masks: 键为 z 索引、值为该层 2D mask 的字典。
        - 返回:
          - 3D 包围框字典（x,y,z,w,h,d），若为空则返回 None。
        - 流程:
          - 收集所有非空切片。
          - 聚合全局 x/y/z 最小最大边界。
          - 组装 bbox。
        """
        non_empty = [(z, m) for z, m in slice_masks.items() if np.any(m)]
        if not non_empty:
            return None

        z_vals = [z for z, _ in non_empty]
        x_mins: List[int] = []
        x_maxs: List[int] = []
        y_mins: List[int] = []
        y_maxs: List[int] = []

        for _, mask in non_empty:
            ys, xs = np.where(mask)
            x_mins.append(int(xs.min()))
            x_maxs.append(int(xs.max()))
            y_mins.append(int(ys.min()))
            y_maxs.append(int(ys.max()))

        x0, x1 = min(x_mins), max(x_maxs)
        y0, y1 = min(y_mins), max(y_maxs)
        z0, z1 = min(z_vals), max(z_vals)

        return {
            "x": x0,
            "y": y0,
            "z": z0,
            "w": int(x1 - x0 + 1),
            "h": int(y1 - y0 + 1),
            "d": int(z1 - z0 + 1),
        }

    @staticmethod
    def _build_mask_crop_from_slice_masks(slice_masks: Dict[int, np.ndarray], bbox: Dict[str, int]) -> np.ndarray:
        """
        - 参数:
          - slice_masks: 全部有效切片 mask。
          - bbox: 结节 3D 包围框。
        - 返回:
          - 仅结节包围框内的 3D 布尔 mask（d,h,w）。
        - 流程:
          - 按 bbox 初始化裁剪体积。
          - 将每个切片映射到局部 z 位置并裁剪 x/y。
        """
        crop = np.zeros((bbox["d"], bbox["h"], bbox["w"]), dtype=bool)
        x0, y0, z0 = bbox["x"], bbox["y"], bbox["z"]

        for z, m2d in slice_masks.items():
            if z < z0 or z >= z0 + bbox["d"]:
                continue
            local_z = z - z0
            crop[local_z] = m2d[y0:y0 + bbox["h"], x0:x0 + bbox["w"]]

        return crop

    @staticmethod
    def _read_hu_slice(dicom_file: str) -> Optional[np.ndarray]:
        """
        - 参数:
          - dicom_file: DICOM 文件路径。
        - 返回:
          - Optional[np.ndarray]: HU 2D 切片，失败返回 None。
        - 流程:
          - 读取 pixel_array。
          - 应用 RescaleSlope/RescaleIntercept 转换为 HU。
        """
        try:
            ds = pydicom.dcmread(dicom_file, force=True)
            pixel = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            return pixel * slope + intercept
        except Exception:
            return None

    def _build_hu_crop_from_bbox(self, series_info: Dict[str, Any], bbox: Dict[str, int]) -> np.ndarray:
        """
        - 参数:
          - series_info: 序列信息，包含排序后的 DICOM 切片列表。
          - bbox: 结节 3D 包围框。
        - 返回:
          - np.ndarray: 与 mask_crop 同尺寸的 HU 裁剪体积 (d,h,w)。
        - 流程:
          - 按 bbox 的 z 范围读取对应 DICOM。
          - 转 HU 并裁剪 x/y 区域。
          - 组装 3D HU 裁剪数组。
        """
        d, h, w = bbox["d"], bbox["h"], bbox["w"]
        x0, y0, z0 = bbox["x"], bbox["y"], bbox["z"]
        hu_crop = np.zeros((d, h, w), dtype=np.float32)

        slices = series_info.get("slices", [])
        for local_z in range(d):
            z_idx = z0 + local_z
            if z_idx < 0 or z_idx >= len(slices):
                continue

            dicom_file = slices[z_idx].get("file")
            if dicom_file is None:
                continue

            hu_slice = self._read_hu_slice(dicom_file)
            if hu_slice is None:
                continue

            hu_crop[local_z] = hu_slice[y0:y0 + h, x0:x0 + w]

        return hu_crop

    @staticmethod
    def _centroid_from_crop(mask_crop: np.ndarray, bbox: Dict[str, int]) -> Optional[List[float]]:
        """
        - 参数:
          - mask_crop: 裁剪后 3D mask。
          - bbox: 对应全局包围框。
        - 返回:
          - 全局坐标系下的质心 [z,y,x]，为空则返回 None。
        - 流程:
          - 在 crop 上找前景体素坐标。
          - 计算局部均值。
          - 加上 bbox 偏移还原全局坐标。
        """
        coords = np.argwhere(mask_crop)
        if len(coords) == 0:
            return None
        local_centroid = coords.mean(axis=0)
        return [
            float(local_centroid[0] + bbox["z"]),
            float(local_centroid[1] + bbox["y"]),
            float(local_centroid[2] + bbox["x"]),
        ]

    @staticmethod
    def _bbox_iou_3d(a: Dict[str, int], b: Dict[str, int]) -> float:
        """
        - 参数:
          - a: bbox A（x,y,z,w,h,d）。
          - b: bbox B（x,y,z,w,h,d）。
        - 返回:
          - 两个 3D 包围框的 IoU。
        - 流程:
          - 计算交集体积。
          - 计算并集体积。
          - 返回交并比。
        """
        ax0, ay0, az0 = a["x"], a["y"], a["z"]
        ax1, ay1, az1 = ax0 + a["w"], ay0 + a["h"], az0 + a["d"]
        bx0, by0, bz0 = b["x"], b["y"], b["z"]
        bx1, by1, bz1 = bx0 + b["w"], by0 + b["h"], bz0 + b["d"]

        ix0, iy0, iz0 = max(ax0, bx0), max(ay0, by0), max(az0, bz0)
        ix1, iy1, iz1 = min(ax1, bx1), min(ay1, by1), min(az1, bz1)

        iw, ih, idp = max(0, ix1 - ix0), max(0, iy1 - iy0), max(0, iz1 - iz0)
        inter = iw * ih * idp
        if inter <= 0:
            return 0.0

        va = a["w"] * a["h"] * a["d"]
        vb = b["w"] * b["h"] * b["d"]
        union = va + vb - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    @staticmethod
    def _mean_features(feature_list: List[Dict[str, Optional[int]]]) -> Dict[str, Optional[float]]:
        """
        - 参数:
          - feature_list: 多位医生的特征字典列表。
        - 返回:
          - 各特征均值字典，空值自动忽略。
        - 流程:
          - 按键聚合非空值。
          - 计算均值。
          - 无有效值时返回 None。
        """
        if not feature_list:
            return {}

        keys = list(feature_list[0].keys())
        out: Dict[str, Optional[float]] = {}
        for k in keys:
            vals = [float(v[k]) for v in feature_list if v.get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
        return out

    @staticmethod
    def _extract_characteristics(characteristics_node: Optional[ET.Element]) -> Dict[str, Optional[int]]:
        """
        - 参数:
          - characteristics_node: XML 中 characteristics 节点。
        - 返回:
          - 特征字典（subtlety / texture / malignancy 等）。
        - 流程:
          - 预置全部键。
          - 按字段读取并转 int。
        """
        keys = [
            "subtlety",
            "internalStructure",
            "calcification",
            "sphericity",
            "margin",
            "lobulation",
            "spiculation",
            "texture",
            "malignancy",
        ]
        output: Dict[str, Optional[int]] = {k: None for k in keys}
        if characteristics_node is None:
            return output

        for child in list(characteristics_node):
            key = LIDCNoduleParser._strip_ns(child.tag)
            if key in output:
                output[key] = LIDCNoduleParser._safe_int(child.text.strip() if child.text else None)

        return output

    def _list_case_dirs(self) -> List[Path]:
        """
        - 参数:
          - 无。
        - 返回:
          - 病例目录列表。
        - 流程:
          - 若 dataset_dir 下存在子目录则按子目录作为病例。
          - 否则将 dataset_dir 本身视为单病例目录。
        """
        case_dirs = sorted([p for p in self.dataset_dir.iterdir() if p.is_dir()])
        if case_dirs:
            return case_dirs
        return [self.dataset_dir]

    @staticmethod
    def _case_id(case_dir: Path) -> str:
        """
        - 参数:
          - case_dir: 病例目录。
        - 返回:
          - 病例名（目录名）。
        - 流程:
          - 直接读取目录名。
        """
        return case_dir.name

    def _cache_path_for_case(self, case_id: str) -> Path:
        """
        - 参数:
          - case_id: 病例名。
        - 返回:
          - 对应病例缓存文件路径。
        - 流程:
          - 拼接 output_dir/cache/{case_id}.pkl。
        """
        return self.cache_dir / f"{case_id}.pkl"

    @staticmethod
    def _find_xml_files(case_dir: Path) -> List[Path]:
        """
        - 参数:
          - case_dir: 病例目录。
        - 返回:
          - 病例下全部 XML 文件。
        - 流程:
          - 递归搜索 .xml。
        """
        return sorted(case_dir.rglob("*.xml"))

    def _count_dicoms_in_dir(self, folder: Path) -> int:
        """
        - 参数:
          - folder: 统计目录。
        - 返回:
          - 目录下 DICOM 文件数量。
        - 流程:
          - 命中本地计数缓存则直接返回。
          - 否则递归遍历并累计。
        """
        if folder in self._dicom_count_cache:
            return self._dicom_count_cache[folder]

        count = 0
        for p in folder.rglob("*"):
            if p.is_file() and self._looks_like_dicom(p):
                count += 1

        self._dicom_count_cache[folder] = count
        return count

    def _select_valid_xml_files(self, xml_files: List[Path]) -> List[Path]:
        """
        - 参数:
          - xml_files: 病例内找到的 XML 列表。
        - 返回:
          - 仅保留有效目录（DICOM 数最多目录）下的 XML 列表。
        - 流程:
          - 提取 XML 所在父目录。
          - 若多个目录，选择 DICOM 数最多目录。
          - 返回该目录下 XML。
        """
        if not xml_files:
            return []

        xml_dirs = sorted({x.parent for x in xml_files})
        if len(xml_dirs) == 1:
            valid_dir = xml_dirs[0]
        else:
            valid_dir = max(xml_dirs, key=self._count_dicoms_in_dir)

        return sorted([x for x in xml_files if x.parent == valid_dir])

    def _build_series_dicom_index(self, valid_xml_files: List[Path]) -> Dict[str, Dict[str, Any]]:
        """
        - 参数:
          - valid_xml_files: 有效 XML 列表。
        - 返回:
          - 以 SeriesUID 为键的切片索引（rows/cols/sop->zindex）。
        - 流程:
          - 在 XML 所在目录下扫描 DICOM。
          - 读取关键元信息。
          - 建立 SOP 与 z 索引映射。
        """
        series_index: Dict[str, Dict[str, Any]] = {}
        scan_roots = sorted({x.parent for x in valid_xml_files})

        for scan_root in scan_roots:
            for file_path in scan_root.rglob("*"):
                if not file_path.is_file() or not self._looks_like_dicom(file_path):
                    continue

                try:
                    ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
                except Exception:
                    continue

                series_uid = getattr(ds, "SeriesInstanceUID", None)
                sop_uid = getattr(ds, "SOPInstanceUID", None)
                rows = getattr(ds, "Rows", None)
                cols = getattr(ds, "Columns", None)

                if series_uid is None or sop_uid is None or rows is None or cols is None:
                    continue

                z_value = None
                if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                    z_value = float(ds.ImagePositionPatient[2])
                elif hasattr(ds, "SliceLocation"):
                    z_value = float(ds.SliceLocation)
                elif hasattr(ds, "InstanceNumber"):
                    z_value = float(ds.InstanceNumber)

                if z_value is None:
                    continue

                if series_uid not in series_index:
                    series_index[series_uid] = {"rows": int(rows), "cols": int(cols), "slices": []}

                series_index[series_uid]["slices"].append(
                  {"sop_uid": str(sop_uid), "z": z_value, "file": str(file_path)}
                )

        for info in series_index.values():
            slices = sorted(info["slices"], key=lambda item: item["z"])
            info["slices"] = slices
            info["z_to_index"] = {round(item["z"], 4): idx for idx, item in enumerate(slices)}
            info["sop_to_index"] = {item["sop_uid"]: idx for idx, item in enumerate(slices)}

        return series_index

    def _build_annotation_entry(
        self,
        roi_nodes: List[ET.Element],
        series_info: Dict[str, Any],
        width: int,
        height: int,
    ) -> Optional[Dict[str, Any]]:
        """
        - 参数:
          - roi_nodes: 单个医生的 ROI 节点列表。
          - series_info: 当前序列切片索引信息。
          - width: 图像宽。
          - height: 图像高。
        - 返回:
          - 单标注条目（仅包含 mask_crop，不含 mask_full）；若为空则返回 None。
        - 流程:
          - 逐 ROI 生成每层 2D mask 并处理 inclusion/exclusion。
          - 汇总为 3D bbox。
          - 构造裁剪后的 3D mask 与统计量。
        """
        slice_masks: Dict[int, np.ndarray] = {}

        for roi in roi_nodes:
            inclusion_text = self._child_text(roi, "inclusion")
            is_inclusion = str(inclusion_text).strip().upper() != "FALSE"

            sop_uid = self._child_text(roi, "imageSOP_UID")
            z_position = self._safe_float(self._child_text(roi, "imageZposition"))

            z_index = None
            if sop_uid is not None:
                z_index = series_info["sop_to_index"].get(sop_uid)
            if z_index is None and z_position is not None:
                z_index = series_info["z_to_index"].get(round(z_position, 4))
            if z_index is None:
                continue

            points_xy: List[Tuple[int, int]] = []
            for edge_map in self._find_children(roi, "edgeMap"):
                x = self._safe_int(self._child_text(edge_map, "xCoord"))
                y = self._safe_int(self._child_text(edge_map, "yCoord"))
                if x is None or y is None:
                    continue
                if 0 <= x < width and 0 <= y < height:
                    points_xy.append((x, y))

            if len(points_xy) < 3:
                continue

            polygon_mask = self._polygon_to_mask(height, width, points_xy)
            if z_index not in slice_masks:
                slice_masks[z_index] = np.zeros((height, width), dtype=bool)

            if is_inclusion:
                slice_masks[z_index] |= polygon_mask
            else:
                slice_masks[z_index] &= ~polygon_mask

        bbox = self._to_bbox_from_slice_masks(slice_masks)
        if bbox is None:
            return None

        mask_crop = self._build_mask_crop_from_slice_masks(slice_masks, bbox)
        voxel_count = int(mask_crop.sum())
        if voxel_count <= 0:
            return None

        centroid_zyx = self._centroid_from_crop(mask_crop, bbox)
        hu_crop = self._build_hu_crop_from_bbox(series_info, bbox)

        return {
            "voxel_count": voxel_count,
            "bbox_xyzwhd": bbox,
            "centroid_zyx": centroid_zyx,
            "mask_crop": mask_crop,
          "hu_crop": hu_crop,
        }

    def _merge_reader_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        - 参数:
          - annotations: 同一扫描内的医生标注列表。
        - 返回:
          - 融合后的结节列表（每个结节仅保留一个医生 mask，特征取均值）。
        - 流程:
          - 用 nodule_id 或 3D bbox IoU 做聚类。
          - 每簇选 voxel_count 最大的 mask 作为代表。
          - 对簇内特征按字段求均值。
        """
        if not annotations:
            return []

        clusters: List[List[Dict[str, Any]]] = []
        iou_thr = 0.10

        for ann in annotations:
            placed = False
            for cluster in clusters:
                rep = cluster[0]

                same_id = (
                    ann.get("nodule_id")
                    and rep.get("nodule_id")
                    and str(ann["nodule_id"]) == str(rep["nodule_id"])
                )
                iou = self._bbox_iou_3d(ann["bbox_xyzwhd"], rep["bbox_xyzwhd"])

                if same_id or iou >= iou_thr:
                    cluster.append(ann)
                    placed = True
                    break

            if not placed:
                clusters.append([ann])

        merged_nodules: List[Dict[str, Any]] = []
        for idx, cluster in enumerate(clusters):
            representative = max(cluster, key=lambda x: int(x.get("voxel_count", 0)))
            features_avg = self._mean_features([c["features"] for c in cluster])

            merged_nodules.append(
                {
                    "nodule_index": idx,
                    "n_annotations": len(cluster),
                    "reader_id": representative.get("reader_id"),
                    "nodule_id": representative.get("nodule_id"),
                    "voxel_count": representative["voxel_count"],
                    "bbox_xyzwhd": representative["bbox_xyzwhd"],
                    "centroid_zyx": representative["centroid_zyx"],
                    "mask_crop": representative["mask_crop"],
                    "hu_crop": representative["hu_crop"],
                    "features": features_avg,
                }
            )

        return merged_nodules

    def _parse_single_xml(self, xml_path: Path, series_index: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        - 参数:
          - xml_path: 单个 XML 路径。
          - series_index: 序列 DICOM 索引。
        - 返回:
          - 单扫描解析结果（结节已融合）；失败返回 None。
        - 流程:
          - 解析 XML 头和 readingSession。
          - 先提取每位医生标注。
          - 再做结节级融合。
        """
        try:
            root = ET.parse(str(xml_path)).getroot()
        except Exception:
            return None

        response_header = None
        reading_sessions: List[ET.Element] = []
        for child in list(root):
            tag = self._strip_ns(child.tag)
            if tag == "ResponseHeader":
                response_header = child
            elif tag == "readingSession":
                reading_sessions.append(child)

        if response_header is None:
            return None

        patient_id = (
            self._child_text(response_header, "PatientId")
            or self._child_text(response_header, "PatientID")
            or self._child_text(response_header, "patientId")
        )
        study_uid = self._child_text(response_header, "StudyInstanceUID")
        series_uid = self._child_text(response_header, "SeriesInstanceUid")

        if series_uid is None or series_uid not in series_index:
            return None

        series_info = series_index[series_uid]
        depth = len(series_info["slices"])
        height = int(series_info["rows"])
        width = int(series_info["cols"])

        annotations: List[Dict[str, Any]] = []
        for session_idx, session in enumerate(reading_sessions):
            for nodule_node in self._find_children(session, "unblindedReadNodule"):
                nodule_id = self._child_text(nodule_node, "noduleID")
                characteristics_node = None
                roi_nodes: List[ET.Element] = []

                for child in list(nodule_node):
                    tag = self._strip_ns(child.tag)
                    if tag == "characteristics":
                        characteristics_node = child
                    elif tag == "roi":
                        roi_nodes.append(child)

                if not roi_nodes:
                    continue

                ann_entry = self._build_annotation_entry(
                    roi_nodes=roi_nodes,
                    series_info=series_info,
                    width=width,
                    height=height,
                )
                if ann_entry is None:
                    continue

                ann_entry.update(
                    {
                        "reader_id": session_idx,
                        "nodule_id": nodule_id,
                        "features": self._extract_characteristics(characteristics_node),
                    }
                )
                annotations.append(ann_entry)

        nodules = self._merge_reader_annotations(annotations)

        return {
            "xml_path": str(xml_path),
            "patient_id": patient_id,
            "study_uid": study_uid,
            "series_uid": series_uid,
            "volume_shape": (depth, height, width),
            "nodules": nodules,
        }

    def _save_case_results(self, case_id: str, results: List[Dict[str, Any]]) -> None:
        """
        - 参数:
          - case_id: 病例名。
          - results: 病例解析结果列表（即 `parse()` 返回列表中的该病例部分）。
        - 返回:
          - None
        - 流程:
          - 写入病例缓存文件。
          - 带版本号保存，便于后续兼容。

        - 缓存文件结构 (`output_dir/cache/{case_id}.pkl`):
          - 顶层: `{"version": int, "data": List[scan_dict]}`
          - `data` 内的 `scan_dict`/`nodule_dict` 字段与 `parse()` 返回结构一致。
        """
        cache_path = self._cache_path_for_case(case_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.cache_version, "data": results}
        with cache_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _sanitize_loaded_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        - 参数:
          - results: 从缓存加载出的结果。
        - 返回:
          - 清洗后的结果。
        - 流程:
          - 删除历史缓存中的 mask_full 字段。
          - 保留当前所需字段。
        """
        for scan in results:
            cleaned_nodules: List[Dict[str, Any]] = []
            for nodule in scan.get("nodules", []):
                if "mask_full" in nodule:
                    nodule.pop("mask_full", None)
                if "hu_crop" in nodule:
                    cleaned_nodules.append(nodule)
            scan["nodules"] = cleaned_nodules
        return results

    def _load_case_results(self, case_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        - 参数:
          - case_id: 病例名。
        - 返回:
          - 有效缓存数据（`List[scan_dict]`）；若缓存过旧或异常则返回 None。
        - 流程:
          - 读取缓存文件。
          - 检查版本。
          - 旧格式或异常时返回 None 触发重解析。

        - 说明:
          - 返回值可直接拼接到 `all_results`。
          - 返回前会做字段清洗（例如移除历史字段、过滤无 `hu_crop` 结节）。
        """
        cache_path = self._cache_path_for_case(case_id)
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        if int(payload.get("version", -1)) != self.cache_version:
            return None

        data = payload.get("data", [])
        if not isinstance(data, list):
            return None

        return self._sanitize_loaded_results(data)

    def parse(self) -> List[Dict[str, Any]]:
        """
        - 参数:
          - 无。
        - 返回:
          - 全部病例解析结果列表 `List[scan_dict]`。
        - 流程:
          - 遍历病例目录。
          - 每病例优先读取缓存。
          - 无缓存时解析并回写病例缓存。

        - `scan_dict` 字段:
          - `xml_path: str`，原 XML 文件路径。
          - `patient_id: Optional[str]`，患者 ID（可能缺失）。
          - `study_uid: Optional[str]`，研究 UID（可能缺失）。
          - `series_uid: str`，序列 UID。
          - `volume_shape: Tuple[int, int, int]`，原始体积形状 (depth, height, width)。
          - `nodules: List[nodule_dict]`，结节列表（可能为空）。

        - `nodule_dict` 字段:
          - `nodule_index: int`，结节索引（同一扫描内唯一）。
          - `n_annotations: int`，标注该结节的医生数量。
          - `reader_id: Optional[int]`，标注该结节的医生索引（0-3），若无则为 None。
          - `nodule_id: Optional[str]`，原 XML 中的 noduleID，可能缺失或不唯一。
          - `voxel_count: int`，结节体素数量。
          - `bbox_xyzwhd: Tuple[int, int, int, int, int, int]`，结节 3D 包围框（x,y,z,w,h,d）。
          - `centroid_zyx: Tuple[float, float, float]`，结节质心在全局坐标系下的 (z,y,x)。
          - `mask_crop: np.ndarray[bool]`，结节包围框内的 3D mask（d,h,w）。
          - `hu_crop: np.ndarray[float32]`，结节包围框内的 HU 裁剪体积（d,h,w）。
          - `features: Dict[str, Optional[int]]`，结节特征字典（subtlety / texture / malignancy 等）。

        - 快速访问示例:
          - `results = parser.parse()`
          - `for scan in results:`
          - `    for nodule in scan["nodules"]:`
          - `        hu = nodule["hu_crop"]`
          - `        texture = nodule["features"].get("texture")`
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_dir}")

        all_results: List[Dict[str, Any]] = []
        case_dirs = self._list_case_dirs()

        for case_dir in tqdm(case_dirs, desc="遍历病例", ncols=100):
            case_id = self._case_id(case_dir)
            cached = self._load_case_results(case_id)
            if cached is not None:
                all_results.extend(cached)
                continue

            xml_files = self._find_xml_files(case_dir)
            if not xml_files:
                self._save_case_results(case_id, [])
                continue

            valid_xml_files = self._select_valid_xml_files(xml_files)
            if not valid_xml_files:
                self._save_case_results(case_id, [])
                continue

            series_index = self._build_series_dicom_index(valid_xml_files)
            if not series_index:
                self._save_case_results(case_id, [])
                continue

            case_results: List[Dict[str, Any]] = []
            for xml_path in valid_xml_files:
                parsed = self._parse_single_xml(xml_path, series_index)
                if parsed is not None:
                    case_results.append(parsed)

            self._save_case_results(case_id, case_results)
            all_results.extend(case_results)

        return all_results



if __name__ == "__main__":
    DATASET_DIR = r"D:\Learnfile\Dataset\LIDC-IDRI"
    OUTPUT_DIR = r"D:\Tempfile"

    parser = LIDCNoduleParser(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR)
    results = parser.parse()

    print(f"已解析 xml 数量: {len(results)}")
    total_nodules = sum(len(item["nodules"]) for item in results)
    empty_scans = sum(1 for item in results if len(item.get("nodules", [])) == 0)
    print(f"总结节标注数量: {total_nodules}")
    print(f"无结节 xml 数量: {empty_scans}")
    print(f"缓存目录路径: {parser.cache_dir}")

    if results and results[0]["nodules"]:
        first = results[0]["nodules"][0]
        print("首个结节 bbox:", first["bbox_xyzwhd"])
        print("首个结节属性均值:", first["features"])
    
    print(results[25]["nodules"][0]["mask_crop"].shape)
    # view = napari.view_image(results[33]["nodules"][0]["hu_crop"], name="首个结节裁剪 Mask")
    # napari.run()
