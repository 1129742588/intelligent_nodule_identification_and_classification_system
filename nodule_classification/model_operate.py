from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import napari
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import label_binarize

from .dataload import LIDCNoduleParser
from .model import MultiTaskNoduleClassifier


class NoduleModelService:
    def __init__(self, dataset_dir: str, output_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        self.model_dir = os.path.join(os.path.dirname(__file__), "model_data")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = Path(os.path.join(self.model_dir, "nodule_multitask_model.pkl"))

        self.tasks = ("calcification", "margin", "texture", "spiculation", "sphericity", "malignancy", "lobulation")

        # 任务标签映射阈值定义可随意划分，有一边值相同则转化为二分类
        self.task_thresholds: Dict[str, Dict[str, Any]] = {
            "calcification": {"low": (1, 5), "mid": 5, "high": (6, 6)},
            "margin": {"low": (1, 3), "mid": 4, "high": 5},
            "texture": {"low": 1, "mid": (2, 4), "high": 5},
            "spiculation": {"low": 1, "mid": 2, "high": (3, 5)},
            "sphericity": {"low": (1, 2), "mid": 3, "high": (4, 5)},
            "malignancy": {"low": (1, 2), "mid": 3, "high": (4, 5)},
            "lobulation": {"low": 1, "mid": 2, "high": (3, 5)},
        }


        self.target_block_size = 32
        self.fill_ratio = 0.88
        self.augment_times = 2

        self.parser = LIDCNoduleParser(dataset_dir, output_dir)
        self.model = MultiTaskNoduleClassifier(
            tasks=self.tasks,
            random_state=42,
            binary_mode=False,
            positive_threshold=4,
            max_selected_features=12,
            task_thresholds=self.task_thresholds,
            target_block_size=self.target_block_size,
            fill_ratio=self.fill_ratio,
            enable_augmentation=True,
            augment_times=self.augment_times,
        )

    @staticmethod
    def _flatten_nodules(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nodules: List[Dict[str, Any]] = []
        for scan in results:
            patient_id = str(scan.get("patient_id") or scan.get("series_uid") or "UNKNOWN")
            series_uid = scan.get("series_uid")
            for nodule in scan.get("nodules", []):
                item = dict(nodule)
                item["patient_id"] = patient_id
                item["series_uid"] = series_uid
                nodules.append(item)
        return nodules

    def train(self, val_ratio: float = 0.2) -> Dict[str, Any]:
        results = self.parser.parse()
        all_nodules = self._flatten_nodules(results)

        if len(all_nodules) < 10:
            raise RuntimeError("有效结节样本太少，无法稳定训练。")

        split_labels: List[int] = []
        for nodule in all_nodules:
            raw = nodule.get("features", {}).get("calcification")
            raw_float = self.model._safe_float(raw)
            if raw_float is None:
                split_labels.append(-1)
            else:
                mapped = self.model._map_label(raw_float, task="calcification")
                split_labels.append(-1 if mapped is None else int(mapped))

        split_labels_np = np.asarray(split_labels, dtype=np.int64)
        use_stratify = np.unique(split_labels_np).shape[0] > 1

        train_nodules, val_nodules = train_test_split(
            all_nodules,
            test_size=val_ratio,
            random_state=42,
            shuffle=True,
            stratify=split_labels_np if use_stratify else None,
        )
        # 随机预览若干训练结节的预处理结果（napari窗口），帮助验证预处理流程是否合理。
        # self.preview_preprocessed_nodules(train_nodules, n_samples=4)

        cv_repeats = 5
        total_fit_steps = len(self.tasks) * (1 + cv_repeats)
        with tqdm(total=total_fit_steps, desc="总训练进度", ncols=100, unit="task") as pbar:
            self.model.fit(
                [{"nodules": train_nodules}],
                show_progress=False,
                progress_callback=lambda _: pbar.update(1),
            )
            self.save_model()

            grouped_cv_summary = self._grouped_repeated_cv(
                all_nodules=all_nodules,
                test_size=val_ratio,
                n_repeats=cv_repeats,
                progress_callback=lambda _: pbar.update(1),
            )

        val_arrays = [np.asarray(n["hu_crop"], dtype=np.float32) for n in val_nodules if "hu_crop" in n]
        val_preds = self.model.predict_tasks_from_arrays(val_arrays[:8]) if val_arrays else {}
        val_metrics = self._evaluate_on_validation(val_nodules)
        return {
            "n_all_nodules": len(all_nodules),
            "n_train": len(train_nodules),
            "n_val": len(val_nodules),
            "model_path": str(self.model_path),
            "val_metrics": val_metrics,
            "grouped_cv_summary": grouped_cv_summary,
            "preview_predictions": {k: v.tolist() for k, v in val_preds.items()},
        }

    def preview_preprocessed_nodules(self, nodules: List[Dict[str, Any]], n_samples: int = 4) -> None:
        """
        训练前随机展示若干预处理后的结节数组（napari）。
        """
        valid = [n for n in nodules if "hu_crop" in n]
        if not valid:
            print("[预览] 无可展示结节（未找到 hu_crop）。")
            return

        count = min(max(int(n_samples), 1), len(valid))
        idxs = np.random.choice(len(valid), size=count, replace=False)

        viewer = napari.Viewer(title="Preprocessed Nodule Preview")
        for rank, idx in enumerate(idxs):
            nodule = valid[int(idx)]
            processed = self.model._preprocess_volume(
                np.asarray(nodule["hu_crop"], dtype=np.float32),
                nodule.get("mask_crop"),
            )
            viewer.add_image(processed, name=f"nodule_{rank}", blending="additive")

        print(f"[预览] 已随机展示 {count} 个预处理后结节（napari窗口）。")
        napari.run()

    def _evaluate_on_validation(
        self,
        val_nodules: List[Dict[str, Any]],
        model: MultiTaskNoduleClassifier | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        metrics: Dict[str, Dict[str, Any]] = {}
        eval_model = model if model is not None else self.model

        eval_nodules = [n for n in val_nodules if "hu_crop" in n]
        if not eval_nodules:
            return metrics

        eval_arrays = [np.asarray(n["hu_crop"], dtype=np.float32) for n in eval_nodules]
        pred_map = eval_model.predict_tasks_from_arrays(eval_arrays)

        for task in self.tasks:
            if task not in pred_map:
                continue

            y_true: List[int] = []
            y_pred: List[int] = []
            y_score_rows: List[np.ndarray] = []

            classes_, proba = eval_model.predict_task_proba_from_arrays(eval_arrays, task)
            class_to_idx = {int(classes_[j]): j for j in range(len(classes_))}
            task_preds = pred_map[task]

            for i, nodule in enumerate(eval_nodules):
                raw = nodule.get("features", {}).get(task)
                raw_float = eval_model._safe_float(raw)
                if raw_float is None:
                    continue

                mapped = eval_model._map_label(raw_float, task=task)
                if mapped is None:
                    continue

                y_true.append(int(mapped))
                y_pred.append(int(task_preds[i]))
                y_score_rows.append(np.asarray(proba[i], dtype=np.float64))

            if not y_true:
                continue

            y_true_np = np.asarray(y_true, dtype=np.int64)
            y_pred_np = np.asarray(y_pred, dtype=np.int64)
            y_score = np.stack(y_score_rows, axis=0)

            bundle = eval_model.task_bundles.get(task)
            if bundle is not None:
                labels = np.asarray(bundle.classes_, dtype=np.int64)
            else:
                labels = np.unique(np.concatenate([y_true_np, y_pred_np]))
            cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)

            if labels.shape[0] <= 2:
                positive_label = int(labels[-1])
                precision = float(
                    precision_score(y_true_np, y_pred_np, pos_label=positive_label, zero_division=0)
                )
                recall = float(
                    recall_score(y_true_np, y_pred_np, pos_label=positive_label, zero_division=0)
                )
                f1 = float(f1_score(y_true_np, y_pred_np, pos_label=positive_label, zero_division=0))
            else:
                precision = float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0))
                recall = float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0))
                f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))

            bal_acc = float(balanced_accuracy_score(y_true_np, y_pred_np))
            mcc = float(matthews_corrcoef(y_true_np, y_pred_np))

            score_matrix = np.zeros((len(y_true), len(labels)), dtype=np.float64)
            for col, cls in enumerate(labels):
                cls_int = int(cls)
                if cls_int in class_to_idx:
                    score_matrix[:, col] = y_score[:, class_to_idx[cls_int]]

            if labels.shape[0] == 2:
                positive_label = int(labels[-1])
                positive_col = int(np.where(labels == positive_label)[0][0])
                y_true_binary = (y_true_np == positive_label)
                if np.unique(y_true_binary).shape[0] < 2:
                    auprc = float("nan")
                else:
                    auprc = float(average_precision_score(y_true_binary, score_matrix[:, positive_col]))
            else:
                y_bin = label_binarize(y_true_np, classes=labels)
                valid_scores: List[float] = []
                for col in range(y_bin.shape[1]):
                    col_true = y_bin[:, col]
                    if np.unique(col_true).shape[0] < 2:
                        continue
                    valid_scores.append(float(average_precision_score(col_true, score_matrix[:, col])))
                auprc = float(np.mean(valid_scores)) if valid_scores else float("nan")

            metrics[task] = {
                "n_samples": int(y_true_np.size),
                "classes": labels.astype(int).tolist(),
                "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": bal_acc,
                "mcc": mcc,
                "auprc": auprc,
                "confusion_matrix": cm.astype(int).tolist(),
            }

        return metrics

    def _grouped_repeated_cv(
        self,
        all_nodules: List[Dict[str, Any]],
        test_size: float = 0.2,
        n_repeats: int = 5,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if len(all_nodules) < 10:
            return {}

        groups = np.asarray([str(n.get("patient_id") or "UNKNOWN") for n in all_nodules])
        if np.unique(groups).shape[0] < 2:
            return {}

        splitter = GroupShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=42)
        agg: Dict[str, Dict[str, List[float]]] = {}

        for train_idx, val_idx in splitter.split(np.arange(len(all_nodules)), groups=groups):
            train_nodules = [all_nodules[i] for i in train_idx]
            val_nodules = [all_nodules[i] for i in val_idx]

            fold_model = MultiTaskNoduleClassifier(
                tasks=self.tasks,
                random_state=42,
                binary_mode=False,
                positive_threshold=4,
                max_selected_features=12,
                task_thresholds=self.task_thresholds,
                target_block_size=self.target_block_size,
                fill_ratio=self.fill_ratio,
                enable_augmentation=True,
                augment_times=self.augment_times,
            )

            try:
                fold_model.fit(
                    [{"nodules": train_nodules}],
                    show_progress=False,
                    progress_callback=progress_callback,
                )
            except Exception:
                continue

            fold_metrics = self._evaluate_on_validation(val_nodules, model=fold_model)
            for task_name, m in fold_metrics.items():
                if task_name not in agg:
                    agg[task_name] = {
                        "accuracy": [],
                        "f1": [],
                        "balanced_accuracy": [],
                        "mcc": [],
                        "auprc": [],
                    }

                for k in agg[task_name].keys():
                    val = m.get(k)
                    if val is not None and np.isfinite(val):
                        agg[task_name][k].append(float(val))

        summary: Dict[str, Dict[str, Any]] = {}
        for task_name, metric_series in agg.items():
            task_summary: Dict[str, Any] = {}
            for metric_name, values in metric_series.items():
                if values:
                    task_summary[f"{metric_name}_mean"] = float(np.mean(values))
                    task_summary[f"{metric_name}_std"] = float(np.std(values))
                    task_summary[f"{metric_name}_n_folds"] = int(len(values))
            if task_summary:
                summary[task_name] = task_summary

        return summary

    def save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_path))

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")
        self.model = MultiTaskNoduleClassifier.load(str(self.model_path))

    def predict_one(self, nodule_array: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        对单个结节进行预测
        - 参数: 
            - nodule_array - 预处理后的结节数组（应与训练时的预处理方式一致标准化+归一化）
        - 返回: 包含每个任务的预测类别和各类别概率的字典
            - 结构示例:
                - key : value
                - "margin": {}
                    - key : value
                    - "pred_label": 2
                    - "class_probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}
            - 使用示例：
                - output["margin"]["pred_label"] 获取 margin 任务的预测类别标签
                - output["margin"]["class_probabilities"]["0"] 获取 margin 任务的0类别概率字典。

        """
        if not self.model.task_bundles:
            self.load_model()

        pred_map = self.model.predict_tasks_from_arrays([nodule_array])
        output: Dict[str, Dict[str, Any]] = {}

        for task in self.tasks:
            if task not in self.model.task_bundles:
                continue

            classes_, proba = self.model.predict_task_proba_from_arrays([nodule_array], task)
            pred_label = int(pred_map[task][0])
            class_to_prob = {int(classes_[i]): float(proba[0, i]) for i in range(len(classes_))}

            output[task] = {
                "pred_label": int(pred_label),
                "class_probabilities": class_to_prob,
            }

        return output


if __name__ == "__main__":
    DATASET_DIR = r"D:\Learnfile\Dataset\LIDC-IDRI"
    OUTPUT_DIR = r"D:\Tempfile"

    service = NoduleModelService(DATASET_DIR, OUTPUT_DIR)

    # train_info = service.train(val_ratio=0.2)
    # print("========= 训练完成 =========")
    # print(f"样本统计: all={train_info['n_all_nodules']}, train={train_info['n_train']}, val={train_info['n_val']}")
    # print(f"模型保存路径: {train_info['model_path']}")

    # print("========= 验证集指标 =========")
    # val_metrics = train_info.get("val_metrics", {})
    # if not val_metrics:
    #     print("无可评估指标（可能验证集中没有可用标签）。")
    # else:
    #     for task_name, m in val_metrics.items():
    #         print(f"任务: {task_name}")
    #         print(f"  样本数: {m['n_samples']}")
    #         print(f"  类别: {m['classes']}")
    #         print(f"  Accuracy : {m['accuracy']:.4f}")
    #         print(f"  Precision: {m['precision']:.4f}")
    #         print(f"  Recall   : {m['recall']:.4f}")
    #         print(f"  F1-score : {m['f1']:.4f}")
    #         print(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}")
    #         print(f"  MCC: {m['mcc']:.4f}")
    #         print(f"  AUPRC: {m['auprc']:.4f}")
    #         print(f"  Confusion Matrix: {m['confusion_matrix']}")
    #         print("-" * 50)

    # print("========= 病人分组重复验证(5次) =========")
    # grouped_cv = train_info.get("grouped_cv_summary", {})
    # if not grouped_cv:
    #     print("无分组重复验证结果（可能病人分组不足或任务标签过于单一）。")
    # else:
    #     for task_name, m in grouped_cv.items():
    #         print(f"任务: {task_name}")
    #         print(
    #             f"  Balanced Accuracy: {m.get('balanced_accuracy_mean', float('nan')):.4f} ± "
    #             f"{m.get('balanced_accuracy_std', float('nan')):.4f} (n={m.get('balanced_accuracy_n_folds', 0)})"
    #         )
    #         print(
    #             f"  MCC: {m.get('mcc_mean', float('nan')):.4f} ± "
    #             f"{m.get('mcc_std', float('nan')):.4f} (n={m.get('mcc_n_folds', 0)})"
    #         )
    #         print(
    #             f"  AUPRC: {m.get('auprc_mean', float('nan')):.4f} ± "
    #             f"{m.get('auprc_std', float('nan')):.4f} (n={m.get('auprc_n_folds', 0)})"
    #         )
    #         print("-" * 50)

    # print("========= 预测预览(验证集前8个) =========")
    # print(train_info.get("preview_predictions", {}))

    parser = LIDCNoduleParser(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR)
    results = parser.parse()
    one_arr = results[22]["nodules"][0]["hu_crop"]

    print("========= 单结节数组预测 =========")
    pred = service.predict_one(one_arr)
    print("注: 任务可能是二分类或三分类，类别含义由 task_thresholds 决定。")
    print("=" * 50)
    for task_name, result in pred.items():
        print(f"任务: {task_name}")
        print(f"  预测类别: {result['pred_label']}")
        print(f"  各类别概率: {result['class_probabilities']}")
        print("=" * 50)
