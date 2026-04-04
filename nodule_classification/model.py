from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import pickle
import random

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


DEFAULT_TASKS = ("calcification", "margin", "texture")


@dataclass
class TaskModelBundle:
    task_name: str
    models: Dict[str, Any]
    classes_: np.ndarray
    selector: Any


class MultiTaskNoduleClassifier:
    """
    多任务结节分类器：
    - 每个任务训练多个基分类器（LR / RF / GBDT / KNN / SVM）
    - 每个任务独立做特征筛选（SelectKBest）
    - 支持任务独立预测与融合概率预测
    """

    def __init__(
        self,
        tasks: Sequence[str] = DEFAULT_TASKS,
        random_state: int = 42,
      binary_mode: bool = False,
        positive_threshold: int = 4,
        max_selected_features: int = 12,
      task_thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
      target_block_size: int = 32,
      fill_ratio: float = 0.88,
      enable_augmentation: bool = True,
      augment_times: int = 2,
    ) -> None:
        """
        - 参数:
          - tasks: 多任务名称列表，如 calcification/margin/texture。
          - random_state: 随机种子。
          - binary_mode: 兼容旧逻辑的全局二分类开关。
          - positive_threshold: 旧逻辑下二分类阈值，raw_value >= threshold 记为正类 1。
          - max_selected_features: 每个任务最多保留的筛选特征数。
          - task_thresholds: 任务独立阈值配置，支持自动二/三分类。
          - target_block_size: 固定重采样块大小（默认 32）。
          - fill_ratio: 重采样后结节占满比例（默认 0.88）。
          - enable_augmentation: 训练时是否启用增强。
          - augment_times: 每个结节额外增强样本数量。
        - 返回:
          - None
        - 流程:
          - 保存配置参数。
          - 初始化任务模型容器。
        """
        self.tasks = tuple(tasks)
        self.random_state = random_state
        self.binary_mode = binary_mode
        self.positive_threshold = positive_threshold
        self.max_selected_features = max_selected_features
        self.target_block_size = int(target_block_size)
        self.fill_ratio = float(fill_ratio)
        self.enable_augmentation = bool(enable_augmentation)
        self.augment_times = int(augment_times)
        self._rng = random.Random(self.random_state)
        self.task_thresholds = task_thresholds or self._default_task_thresholds()
        self.task_bundles: Dict[str, TaskModelBundle] = {}
        self.feature_dim: Optional[int] = None

    def _default_task_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """
        - 参数:
          - 无。
        - 返回:
          - 默认任务阈值配置。
        - 流程:
          - 为每个任务创建 low/mid/high 区间。
        """
        return {
            task: {
                "low": (1, 2),
                "mid": 3,
                "high": (4, 5),
            }
            for task in self.tasks
        }

    def _base_estimators(self) -> Dict[str, Any]:
        """
        - 参数:
          - 无。
        - 返回:
          - Dict[str, Any]: 基分类器字典。
            - lr: LogisticRegression 管道
            - rf: RandomForestClassifier
            - gbdt: GradientBoostingClassifier
            - knn: KNeighborsClassifier 管道
            - svm: SVC 管道
        - 流程:
          - 构建并返回每个基分类器实例。
        """
        return {
            "lr": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=self.random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "rf": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=self.random_state,
                class_weight="balanced_subsample",
            ),
            "gbdt": GradientBoostingClassifier(random_state=self.random_state),
            "knn": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
                ]
            ),
            "svm": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=2.0,
                            gamma="scale",
                            probability=True,
                            random_state=self.random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        }

    @staticmethod
    def _in_closed_interval(value: float, interval: Tuple[float, float]) -> bool:
        low, high = float(interval[0]), float(interval[1])
        return low <= value <= high

    @staticmethod
    def _as_interval(value: Any) -> Tuple[float, float]:
        if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
            low = float(value[0])
            high = float(value[1])
            if low <= high:
                return low, high
            return high, low
        mid = float(value)
        return mid, mid

    @staticmethod
    def _intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return not (a[1] < b[0] or b[1] < a[0])

    def _is_binary_task_by_threshold(self, task: str) -> bool:
        cfg = self.task_thresholds.get(task, {})
        low = self._as_interval(cfg.get("low", (1, 2)))
        mid = self._as_interval(cfg.get("mid", 3))
        high = self._as_interval(cfg.get("high", (4, 5)))

        overlap_low_mid = self._intervals_overlap(mid, low)
        overlap_high_mid = self._intervals_overlap(mid, high)
        return bool(overlap_low_mid or overlap_high_mid)

    def _map_label(self, raw_value: float, task: Optional[str] = None) -> Optional[int]:
        """
        - 参数:
          - raw_value: 原始标签值（通常来自 1~6 评分）。
        - 返回:
            - Optional[int]: 映射后的标签；若落在阈值外则返回 None。
            - 任务阈值不重叠: 三分类 low/mid/high -> 0/1/2。
            - 任务阈值与 mid 重叠: 自动退化为二分类 0/1。
            - binary_mode=True: 使用旧阈值逻辑返回 0/1。
        - 流程:
            - 优先按任务独立阈值映射。
            - 若启用 binary_mode 则走旧逻辑。
        """
        if self.binary_mode:
            return int(raw_value >= self.positive_threshold)

        if task is None or task not in self.task_thresholds:
            return int(round(raw_value))

        cfg = self.task_thresholds[task]
        low = self._as_interval(cfg.get("low", (1, 2)))
        mid = self._as_interval(cfg.get("mid", 3))
        high = self._as_interval(cfg.get("high", (4, 5)))

        in_low = self._in_closed_interval(raw_value, low)
        in_mid = self._in_closed_interval(raw_value, mid)
        in_high = self._in_closed_interval(raw_value, high)

        overlap_low_mid = self._intervals_overlap(mid, low)
        overlap_high_mid = self._intervals_overlap(mid, high)

        if overlap_low_mid and overlap_high_mid:
            if in_low or in_mid:
                return 0
            if in_high:
                return 1
            return None

        if overlap_low_mid:
            if in_low or in_mid:
                return 0
            if in_high:
                return 1
            return None

        if overlap_high_mid:
            if in_low:
                return 0
            if in_high or in_mid:
                return 1
            return None

        if in_low:
            return 0
        if in_mid:
            return 1
        if in_high:
            return 2
        return None

    @staticmethod
    def _safe_float(v: Any) -> Optional[float]:
        """
        - 参数:
          - v: 任意输入值。
        - 返回:
          - Optional[float]: 可转换则返回 float，否则返回 None。
        - 流程:
          - 空值直接返回 None。
          - 尝试 float 转换。
        """
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _hu_intensity_features(self, hu_crop: np.ndarray) -> List[float]:
        """
        - 参数:
          - hu_crop: 单结节 3D HU 体积。
        - 返回:
          - List[float]: HU 统计特征列表。
        - 流程:
          - 计算 HU 一阶统计（均值/方差/分位数/极值）。
          - 计算能量与熵特征。
          - 计算三维梯度粗糙度特征。
        """
        arr = np.asarray(hu_crop, dtype=np.float32)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return [0.0] * 16

        mean_hu = float(np.mean(flat))
        std_hu = float(np.std(flat))
        min_hu = float(np.min(flat))
        max_hu = float(np.max(flat))
        p10 = float(np.percentile(flat, 10))
        p25 = float(np.percentile(flat, 25))
        p50 = float(np.percentile(flat, 50))
        p75 = float(np.percentile(flat, 75))
        p90 = float(np.percentile(flat, 90))
        iqr = p75 - p25

        energy = float(np.mean(np.square(flat)))
        clipped = np.clip(flat, -1000.0, 1400.0)
        hist, _ = np.histogram(clipped, bins=64, range=(-1000.0, 1400.0), density=True)
        hist = hist + 1e-8
        entropy = float(-np.sum(hist * np.log(hist)))

        if min(arr.shape) < 2:
            grad_mean = 0.0
            grad_std = 0.0
        else:
            gz, gy, gx = np.gradient(arr)
            grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
            grad_mean = float(np.mean(grad_mag))
            grad_std = float(np.std(grad_mag))

        d, h, w = arr.shape
        anis_dh = float(d / h) if h > 0 else 0.0
        anis_dw = float(d / w) if w > 0 else 0.0
        anis_hw = float(h / w) if w > 0 else 0.0

        return [
            mean_hu,
            std_hu,
            min_hu,
            max_hu,
            p10,
            p25,
            p50,
            p75,
            p90,
            iqr,
            energy,
            entropy,
            grad_mean,
            grad_std,
            anis_dh,
            anis_dw,
            anis_hw,
        ]

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        return int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)

    def _trim_to_foreground(self, volume: np.ndarray, mask_crop: Optional[np.ndarray], margin: int = 2) -> np.ndarray:
        """
        去除黑色/无效背景，仅保留结节附近区域并留少量边缘。
        """
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"volume 必须是3D数组，当前 ndim={volume.ndim}")

        if mask_crop is not None and np.asarray(mask_crop).shape == volume.shape:
            mask = np.asarray(mask_crop, dtype=bool)
        else:
          bg = float(np.percentile(volume, 1))
          eps = max(1e-5, 0.05 * float(np.std(volume)))
          mask = volume > (bg + eps)

        bbox = self._bbox_from_mask(mask)
        if bbox is None:
            return volume
        else:
            z0, y0, x0, z1, y1, x1 = bbox

        margin = max(0, int(margin))
        z0 = max(0, z0 - margin)
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        z1 = min(volume.shape[0] - 1, z1 + margin)
        y1 = min(volume.shape[1] - 1, y1 + margin)
        x1 = min(volume.shape[2] - 1, x1 + margin)

        crop = volume[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        if crop.size == 0:
            return volume
        return crop.astype(np.float32)

    @staticmethod
    def _standardize_and_normalize(volume: np.ndarray) -> np.ndarray:
        """
        统一预处理：先标准化，再归一化到 [0, 1]。
        """
        arr = np.asarray(volume, dtype=np.float32)
        arr = np.clip(arr, -1200.0, 1200.0)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr))
        if std_v < 1e-6:
            z = arr - mean_v
        else:
            z = (arr - mean_v) / std_v

        z_min = float(np.min(z))
        z_max = float(np.max(z))
        if (z_max - z_min) < 1e-6:
            return np.zeros_like(z, dtype=np.float32)
        return ((z - z_min) / (z_max - z_min)).astype(np.float32)

    def _preprocess_volume(self, hu_volume: np.ndarray, mask_crop: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预处理流程：值域对齐(HU) -> 局部去背景裁剪 -> 标准化+归一化。
        不再执行固定尺寸铺满。
        """
        hu = self._to_hu_volume(hu_volume)
        local_crop = self._trim_to_foreground(hu, mask_crop, margin=2)
        return self._standardize_and_normalize(local_crop)

    def _augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        轻量增强：翻转 + 小噪声。
        """
        aug = np.array(volume, copy=True)
        axis = self._rng.choice([0, 1, 2])
        aug = np.flip(aug, axis=axis)

        noise = np.random.normal(loc=0.0, scale=0.02, size=aug.shape).astype(np.float32)
        aug = np.clip(aug + noise, 0.0, 1.0)
        return aug.astype(np.float32)

    def _bbox_features(self, bbox: Dict[str, Any]) -> List[float]:
        """
        - 参数:
          - bbox: 结节框字典（x/y/z/w/h/d）。
        - 返回:
          - List[float]: bbox 数值特征列表。
        - 流程:
          - 提取并安全转换 x/y/z/w/h/d。
          - 对缺失值使用 0.0。
        """
        x = self._safe_float(bbox.get("x")) or 0.0
        y = self._safe_float(bbox.get("y")) or 0.0
        z = self._safe_float(bbox.get("z")) or 0.0
        w = self._safe_float(bbox.get("w")) or 0.0
        h = self._safe_float(bbox.get("h")) or 0.0
        d = self._safe_float(bbox.get("d")) or 0.0
        return [x, y, z, w, h, d]

    def extract_feature_vector(self, nodule: Dict[str, Any]) -> np.ndarray:
        """
        - 参数:
          - nodule: 单个结节字典。
            - 必需键: hu_crop
            - 可选键: bbox_xyzwhd
        - 返回:
          - np.ndarray: 1D 浮点特征向量。
        - 流程:
          - 从 hu_crop 提取 HU 强度统计特征。
          - 从 bbox 提取位置/尺度特征。
          - 拼接并转为 float32。
        """
        hu_crop = np.asarray(nodule["hu_crop"], dtype=np.float32)
        mask_crop = nodule.get("mask_crop")
        hu_crop = self._preprocess_volume(hu_crop, mask_crop)
        bbox = nodule.get("bbox_xyzwhd", {})
        feature = self._hu_intensity_features(hu_crop) + self._bbox_features(bbox)
        return np.asarray(feature, dtype=np.float32)

    @staticmethod
    def _to_hu_volume(arr: np.ndarray) -> np.ndarray:
        """
        - 参数:
          - arr: 输入 3D 结节数组（原始 HU 或归一化值）。
        - 返回:
          - np.ndarray: 标准化到 HU 近似范围的 3D 数组。
        - 流程:
          - 判断输入值域。
          - 识别 [0,1]/[-1,1]/[0,255] 等归一化场景并反归一化到 HU。
          - 其余数值按 HU 原值处理。
        """
        volume = np.asarray(arr, dtype=np.float32)
        finite_mask = np.isfinite(volume)
        if not np.any(finite_mask):
            return np.zeros_like(volume, dtype=np.float32)

        finite_vals = volume[finite_mask]
        min_v = float(np.min(finite_vals))
        max_v = float(np.max(finite_vals))

        if 0.0 <= min_v and max_v <= 1.0:
            return volume * 2400.0 - 1000.0
        if -1.0 <= min_v and max_v <= 1.0:
            return ((volume + 1.0) / 2.0) * 2400.0 - 1000.0
        if 0.0 <= min_v and max_v <= 255.0 and np.issubdtype(arr.dtype, np.integer):
            return volume / 255.0 * 2400.0 - 1000.0
        return volume

    def _nodule_dicts_from_arrays(self, nodule_arrays: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        """
        - 参数:
          - nodule_arrays: 结节数组列表。
            - 每个元素必须是 3D 数组，shape 可不一致。
            - 支持 bool / int / float 输入。
        - 返回:
          - List[Dict[str, Any]]: 标准化结节字典列表。
            - 每个元素结构:
              - hu_crop: np.ndarray(float32), HU 体积
              - bbox_xyzwhd: Dict[str,int]，默认以数组尺寸构造的局部框
        - 流程:
          - 校验维度。
          - 自动识别输入是否归一化并转换到 HU。
          - 构造可供内部预测使用的结节字典。
        """
        nodules: List[Dict[str, Any]] = []
        for idx, arr in enumerate(nodule_arrays):
            mask = np.asarray(arr)
            if mask.ndim != 3:
                raise ValueError(f"第 {idx} 个结节数组不是 3D，当前 ndim={mask.ndim}。")

            hu_crop = self._preprocess_volume(mask, mask_crop=None)

            d, h, w = hu_crop.shape
            nodules.append(
                {
                    "hu_crop": hu_crop.astype(np.float32),
                    "bbox_xyzwhd": {"x": 0, "y": 0, "z": 0, "w": int(w), "h": int(h), "d": int(d)},
                }
            )
        return nodules

    def build_samples(self, parsed_results: Sequence[Dict[str, Any]], augment: bool = False) -> List[Dict[str, Any]]:
        """
        - 参数:
          - parsed_results: 解析器输出结果列表（scan 列表）。
        - 返回:
          - List[Dict[str, Any]]: 训练样本列表。
            - 每个样本结构:
              - patient_id: Any
              - series_uid: Any
              - feature: np.ndarray
              - labels: Dict[str, Any]
        - 流程:
          - 遍历 scan/nodule。
          - 提取特征向量。
          - 汇总标签字典。
            - 按需进行增强样本扩展。
        """
        samples: List[Dict[str, Any]] = []
        for scan in parsed_results:
            patient_id = scan.get("patient_id")
            series_uid = scan.get("series_uid")
            for nodule in scan.get("nodules", []):
                if "hu_crop" not in nodule:
                    continue
                processed_volume = self._preprocess_volume(
                    np.asarray(nodule["hu_crop"], dtype=np.float32),
                    nodule.get("mask_crop"),
                )

                feature = self._hu_intensity_features(processed_volume) + self._bbox_features(
                    nodule.get("bbox_xyzwhd", {})
                )
                samples.append(
                    {
                        "patient_id": patient_id,
                        "series_uid": series_uid,
                        "feature": np.asarray(feature, dtype=np.float32),
                        "labels": dict(nodule.get("features", {})),
                    }
                )

                if augment and self.enable_augmentation and self.augment_times > 0:
                    for _ in range(self.augment_times):
                        aug_vol = self._augment_volume(processed_volume)
                        aug_feature = self._hu_intensity_features(aug_vol) + self._bbox_features(
                            nodule.get("bbox_xyzwhd", {})
                        )
                        samples.append(
                            {
                                "patient_id": patient_id,
                                "series_uid": series_uid,
                                "feature": np.asarray(aug_feature, dtype=np.float32),
                                "labels": dict(nodule.get("features", {})),
                            }
                        )
        return samples

    def _build_xy_for_task(self, samples: Sequence[Dict[str, Any]], task: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        - 参数:
          - samples: 样本列表。
          - task: 任务名。
        - 返回:
          - Tuple[np.ndarray, np.ndarray]:
            - x: 特征矩阵
            - y: 标签数组
        - 流程:
          - 过滤该任务缺失标签样本。
          - 标签按当前策略映射（二分类或多分类）。
          - 组装并返回 x/y。
        """
        x_list: List[np.ndarray] = []
        y_list: List[int] = []

        for item in samples:
            value = item["labels"].get(task)
            value_float = self._safe_float(value)
            if value_float is None:
                continue
            mapped = self._map_label(value_float, task=task)
            if mapped is None:
                continue
            x_list.append(item["feature"])
            y_list.append(mapped)

        if not x_list:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.asarray(y_list, dtype=np.int64)
        return x, y

    def fit(
      self,
      parsed_results: Sequence[Dict[str, Any]],
      show_progress: bool = True,
      progress_callback: Optional[Callable[[str], None]] = None,
    ) -> "MultiTaskNoduleClassifier":
        """
        - 参数:
          - parsed_results: 解析结果列表。
        - 返回:
          - MultiTaskNoduleClassifier: 训练后的自身对象。
        - 流程:
          - 构建样本。
          - 对每个任务构建 X/y。
          - 执行特征筛选（SelectKBest）。
          - 训练 5 个基分类器并保存任务模型包。
        """
        samples = self.build_samples(parsed_results, augment=True)
        if not samples:
            raise ValueError("没有可用于训练的结节样本。")

        self.feature_dim = int(samples[0]["feature"].shape[0])
        self.task_bundles.clear()

        task_iter = self.tasks
        if show_progress and progress_callback is None:
            task_iter = tqdm(self.tasks, desc="训练任务", ncols=100, unit="task")

        for task in task_iter:
            x, y = self._build_xy_for_task(samples, task)
            if x.shape[0] == 0:
                if progress_callback is not None:
                    progress_callback(task)
                continue

            unique_classes = np.unique(y)
            if unique_classes.shape[0] < 2:
                if progress_callback is not None:
                    progress_callback(task)
                continue

            k = min(int(self.max_selected_features), int(x.shape[1]))
            selector = SelectKBest(score_func=f_classif, k=k)
            x_selected = selector.fit_transform(x, y)

            trained_models: Dict[str, Any] = {}
            for name, est in self._base_estimators().items():
                model = clone(est)
                model.fit(x_selected, y)
                trained_models[name] = model

            self.task_bundles[task] = TaskModelBundle(
                task_name=task,
                models=trained_models,
                classes_=unique_classes,
                selector=selector,
            )

            if progress_callback is not None:
              progress_callback(task)

        if not self.task_bundles:
            raise ValueError("所有任务都没有足够的有效标签（至少需要2个类别）。")

        return self

    @staticmethod
    def _average_proba(models: Dict[str, Any], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        - 参数:
          - models: 任务下的基分类器字典。
          - x: 已对齐到该任务特征空间的输入特征。
        - 返回:
          - Tuple[np.ndarray, np.ndarray]:
            - classes_ref: 类别数组。
            - avg_proba: 平均后的类别概率矩阵，shape=(n_samples,n_classes)。
        - 流程:
          - 逐模型计算 predict_proba。
          - 将不同模型类别索引对齐。
          - 对概率做算术平均。
        """
        probas = []
        classes_ref = None

        for model in models.values():
            p = model.predict_proba(x)
            c = model.classes_
            if classes_ref is None:
                classes_ref = c
                probas.append(p)
                continue

            aligned = np.zeros((x.shape[0], classes_ref.shape[0]), dtype=np.float64)
            class_to_idx = {int(v): i for i, v in enumerate(c)}
            for j, cls_val in enumerate(classes_ref):
                if int(cls_val) in class_to_idx:
                    aligned[:, j] = p[:, class_to_idx[int(cls_val)]]
            probas.append(aligned)

        avg = np.mean(np.stack(probas, axis=0), axis=0)
        return classes_ref, avg

    def predict_tasks(self, nodules: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        - 参数:
          - nodules: 结节字典列表（需包含 mask_crop，可选 bbox_xyzwhd）。
        - 返回:
          - Dict[str, np.ndarray]: 多任务预测标签。
            - key: 任务名
            - value: 该任务预测类别数组（长度 = 输入结节数）
        - 流程:
          - 提取全量特征。
          - 按任务 selector 做特征筛选变换。
          - 基模型软投票并取最大概率类别。
        """
        if not self.task_bundles:
            raise RuntimeError("模型尚未训练，请先调用 fit。")

        x = np.stack([self.extract_feature_vector(n) for n in nodules], axis=0).astype(np.float32)
        outputs: Dict[str, np.ndarray] = {}

        for task, bundle in self.task_bundles.items():
            x_task = bundle.selector.transform(x)
            classes_, avg_proba = self._average_proba(bundle.models, x_task)
            pred_idx = np.argmax(avg_proba, axis=1)
            outputs[task] = classes_[pred_idx]

        return outputs

    def predict_task_proba(self, nodules: Sequence[Dict[str, Any]], task: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        - 参数:
          - nodules: 结节字典列表。
          - task: 任务名。
        - 返回:
          - Tuple[np.ndarray, np.ndarray]:
            - classes_: 类别数组。
            - avg_proba: 概率矩阵，shape=(n_samples,n_classes)。
        - 流程:
          - 校验任务是否已训练。
          - 特征提取与任务特征筛选。
          - 输出软投票概率。
        """
        if task not in self.task_bundles:
            raise KeyError(f"任务 {task} 未训练或不存在。")

        x = np.stack([self.extract_feature_vector(n) for n in nodules], axis=0).astype(np.float32)
        bundle = self.task_bundles[task]
        x_task = bundle.selector.transform(x)
        classes_, avg_proba = self._average_proba(bundle.models, x_task)
        return classes_, avg_proba

    def evaluate_consistency_score(self, nodules: Sequence[Dict[str, Any]]) -> np.ndarray:
        """
        - 参数:
          - nodules: 结节字典列表。
        - 返回:
          - np.ndarray: 一致性置信度数组（长度=输入结节数）。
        - 流程:
          - 对每个任务取软投票最大概率。
          - 在任务维度上取均值。
        """
        if not self.task_bundles:
            raise RuntimeError("模型尚未训练，请先调用 fit。")

        x = np.stack([self.extract_feature_vector(n) for n in nodules], axis=0).astype(np.float32)
        task_confidences = []

        for bundle in self.task_bundles.values():
            x_task = bundle.selector.transform(x)
            _, avg_proba = self._average_proba(bundle.models, x_task)
            task_confidences.append(np.max(avg_proba, axis=1))

        return np.mean(np.stack(task_confidences, axis=0), axis=0)

    def predict_tasks_from_arrays(self, nodule_arrays: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        - 参数:
          - nodule_arrays: 3D 结节数组列表（大小可不一致）。
        - 返回:
          - Dict[str, np.ndarray]: 多任务预测标签。
            - key: 任务名
            - value: 对应任务预测标签数组
        - 流程:
          - 将数组标准化为内部结节字典。
          - 复用 predict_tasks 完成推理。
        """
        nodules = self._nodule_dicts_from_arrays(nodule_arrays)
        return self.predict_tasks(nodules)

    def predict_task_proba_from_arrays(
        self,
        nodule_arrays: Sequence[np.ndarray],
        task: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        - 参数:
          - nodule_arrays: 3D 结节数组列表（大小可不一致）。
          - task: 任务名。
        - 返回:
          - Tuple[np.ndarray, np.ndarray]:
            - classes_: 类别数组
            - avg_proba: 概率矩阵
        - 流程:
          - 数组标准化。
          - 复用 predict_task_proba 输出概率。
        """
        nodules = self._nodule_dicts_from_arrays(nodule_arrays)
        return self.predict_task_proba(nodules, task)

    def save(self, path: str) -> None:
        """
        - 参数:
          - path: 模型保存路径。
        - 返回:
          - None
        - 流程:
          - 打包配置与训练后的任务模型。
          - 使用 pickle 序列化保存。
        """
        payload = {
            "tasks": self.tasks,
            "random_state": self.random_state,
            "binary_mode": self.binary_mode,
            "positive_threshold": self.positive_threshold,
            "max_selected_features": self.max_selected_features,
          "task_thresholds": self.task_thresholds,
          "target_block_size": self.target_block_size,
          "fill_ratio": self.fill_ratio,
          "enable_augmentation": self.enable_augmentation,
          "augment_times": self.augment_times,
            "feature_dim": self.feature_dim,
            "task_bundles": self.task_bundles,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def _load_pickle_compat(cls, file_obj: Any) -> Any:
        """
        兼容不同模块路径下的 pickle 反序列化。
        """

        class _CompatUnpickler(pickle.Unpickler):
            module_aliases = {
                "model": __name__,
                "nodule_classification.model": __name__,
            }

            def find_class(self, module: str, name: str) -> Any:
                remapped_module = self.module_aliases.get(module, module)
                return super().find_class(remapped_module, name)

        try:
            return pickle.load(file_obj)
        except ModuleNotFoundError:
            file_obj.seek(0)
            return _CompatUnpickler(file_obj).load()

    @classmethod
    def load(cls, path: str) -> "MultiTaskNoduleClassifier":
        """
        - 参数:
          - path: 模型文件路径。
        - 返回:
          - MultiTaskNoduleClassifier: 从磁盘恢复的模型对象。
        - 流程:
          - 读取 pickle。
          - 按保存参数重建对象并回填训练产物。
        """
        with open(path, "rb") as f:
            payload = cls._load_pickle_compat(f)

        if isinstance(payload, cls):
            return payload

        if not isinstance(payload, dict):
            raise TypeError(f"不支持的模型文件格式: {type(payload)}")

        obj = cls(
            tasks=payload["tasks"],
            random_state=payload["random_state"],
            binary_mode=payload.get("binary_mode", True),
            positive_threshold=payload.get("positive_threshold", 4),
            max_selected_features=payload.get("max_selected_features", 12),
          task_thresholds=payload.get("task_thresholds"),
          target_block_size=payload.get("target_block_size", 32),
          fill_ratio=payload.get("fill_ratio", 0.88),
          enable_augmentation=payload.get("enable_augmentation", True),
          augment_times=payload.get("augment_times", 2),
        )
        obj.feature_dim = payload.get("feature_dim")
        obj.task_bundles = payload["task_bundles"]
        return obj


if __name__ == "__main__":
    from dataload import LIDCNoduleParser

    DATASET_DIR = r"D:\Learnfile\Dataset\LIDC-IDRI"
    OUTPUT_DIR = r"D:\Tempfile"

    parser = LIDCNoduleParser(DATASET_DIR, OUTPUT_DIR)
    results = parser.parse()

    clf = MultiTaskNoduleClassifier(tasks=("calcification", "margin", "texture"))
    clf.fit(results)

    all_nodules = [n for scan in results for n in scan.get("nodules", [])]
    preds = clf.predict_tasks(all_nodules[:10])
    score = clf.evaluate_consistency_score(all_nodules[:10])

    print("任务预测示例:", {k: v.tolist() for k, v in preds.items()})
    print("综合置信度示例:", score.tolist())
