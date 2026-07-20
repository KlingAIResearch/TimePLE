"""
QVHighlights 数据集专用评估指标

实现标准的 Moment Retrieval 评估指标:
- R@n,θ (Recall@n at IoU threshold θ): 在 top-n 预测中是否有与任意 GT 的 IoU ≥ θ
- mAP@θ (mean Average Precision at IoU threshold θ): 基于所有候选片段的 AP 计算

参考:
- QVHighlights 论文: https://arxiv.org/abs/2107.09609
- Moment-DETR 官方实现: https://github.com/jayleicn/moment_detr
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


GT_DURATION_GROUP_SPECS = [
    {
        "group": "Short",
        "metric_key": "mIoU_S",
        "duration_range": "(0,10]s",
        "min_duration_sec": 0.0,
        "max_duration_sec": 10.0,
    },
    {
        "group": "Medium",
        "metric_key": "mIoU_M",
        "duration_range": "(10,30]s",
        "min_duration_sec": 10.0,
        "max_duration_sec": 30.0,
    },
    {
        "group": "Long",
        "metric_key": "mIoU_L",
        "duration_range": "(30,∞)s",
        "min_duration_sec": 30.0,
        "max_duration_sec": None,
    },
]


def calculate_total_gt_duration(gt_timestamps: List[Tuple[float, float]]) -> float:
    """Use the sum of all GT segment lengths as the sample duration."""
    return float(sum(max(0.0, end - start) for start, end in gt_timestamps))


def find_gt_duration_group(gt_duration_sec: float) -> Optional[Dict]:
    for group in GT_DURATION_GROUP_SPECS:
        start = group["min_duration_sec"]
        end = group["max_duration_sec"]
        if end is None:
            if gt_duration_sec > start:
                return group
        elif start < gt_duration_sec <= end:
            return group
    return None


def calculate_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """
    计算两个时间区间的 IoU（Intersection over Union）

    参数:
        pred: (start, end) 预测的时间区间
        gt: (start, end) 真实的时间区间

    返回:
        IoU 值 (0-1)
    """
    pred_start, pred_end = pred
    gt_start, gt_end = gt

    # 计算交集
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    # 计算并集
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start

    if union == 0:
        return 0.0

    return intersection / union


def compute_temporal_iou_batch(pred_windows: np.ndarray, gt_windows: np.ndarray) -> np.ndarray:
    """
    批量计算预测窗口和 GT 窗口之间的 IoU

    参数:
        pred_windows: shape (N, 2), N 个预测窗口 [start, end]
        gt_windows: shape (M, 2), M 个 GT 窗口 [start, end]

    返回:
        iou_matrix: shape (N, M), IoU 矩阵
    """
    # pred_windows: (N, 2) -> (N, 1, 2)
    # gt_windows: (M, 2) -> (1, M, 2)
    pred_windows = pred_windows[:, np.newaxis, :]  # (N, 1, 2)
    gt_windows = gt_windows[np.newaxis, :, :]      # (1, M, 2)

    # 计算交集
    inter_start = np.maximum(pred_windows[:, :, 0], gt_windows[:, :, 0])  # (N, M)
    inter_end = np.minimum(pred_windows[:, :, 1], gt_windows[:, :, 1])    # (N, M)
    inter = np.maximum(0, inter_end - inter_start)  # (N, M)

    # 计算并集
    union_start = np.minimum(pred_windows[:, :, 0], gt_windows[:, :, 0])  # (N, M)
    union_end = np.maximum(pred_windows[:, :, 1], gt_windows[:, :, 1])    # (N, M)
    union = union_end - union_start  # (N, M)

    # 避免除零
    iou = inter / np.maximum(union, 1e-10)

    return iou


def calculate_recall_at_k(
    pred_windows: List[Tuple[float, float]],
    gt_windows: List[Tuple[float, float]],
    iou_threshold: float,
    top_k: int = 1
) -> bool:
    """
    计算 R@k,θ: 在 top-k 个预测中是否有与任意 GT 的 IoU ≥ θ

    参数:
        pred_windows: 预测的时间窗口列表（已按置信度排序，从高到低）
        gt_windows: GT 时间窗口列表
        iou_threshold: IoU 阈值 θ
        top_k: 取 top-k 个预测

    返回:
        bool: 是否命中（True/False）
    """
    if not pred_windows or not gt_windows:
        return False

    # 只取 top-k 个预测
    top_k_preds = pred_windows[:top_k]

    # 转换为 numpy 数组
    pred_array = np.array(top_k_preds)  # (k, 2)
    gt_array = np.array(gt_windows)     # (M, 2)

    # 计算 IoU 矩阵: (k, M)
    iou_matrix = compute_temporal_iou_batch(pred_array, gt_array)

    # 检查是否有任何预测与任意 GT 的 IoU ≥ threshold
    # 对于每个预测，取与所有 GT 的最大 IoU
    max_ious = np.max(iou_matrix, axis=1)  # (k,)

    # 如果任意一个预测的最大 IoU ≥ threshold，则命中
    return bool(np.any(max_ious >= iou_threshold))


def calculate_average_precision(
    pred_windows: List[Tuple[float, float]],
    pred_scores: List[float],
    gt_windows: List[Tuple[float, float]],
    iou_threshold: float
) -> float:
    """
    计算 Average Precision (AP) at IoU threshold θ

    AP 的计算步骤:
    1. 按照预测分数从高到低排序所有候选片段
    2. 对于每个候选，判断是否与任意 GT 的 IoU ≥ θ（正样本）
    3. 计算 Precision-Recall 曲线下的面积

    参数:
        pred_windows: 预测的时间窗口列表
        pred_scores: 对应的置信度分数列表（越高越好）
        gt_windows: GT 时间窗口列表
        iou_threshold: IoU 阈值 θ

    返回:
        AP 值 (0-1)
    """
    if not pred_windows or not gt_windows:
        return 0.0

    # 按分数从高到低排序
    sorted_indices = np.argsort(pred_scores)[::-1]
    sorted_pred_windows = [pred_windows[i] for i in sorted_indices]

    # 转换为 numpy 数组
    pred_array = np.array(sorted_pred_windows)  # (N, 2)
    gt_array = np.array(gt_windows)              # (M, 2)

    # 计算 IoU 矩阵: (N, M)
    iou_matrix = compute_temporal_iou_batch(pred_array, gt_array)

    # 对于每个预测，取与所有 GT 的最大 IoU
    max_ious = np.max(iou_matrix, axis=1)  # (N,)

    # 标记正样本（IoU ≥ threshold）
    is_positive = (max_ious >= iou_threshold).astype(float)  # (N,)

    num_gt = len(gt_windows)

    # 计算累积的 TP 和 precision
    tp_cumsum = np.cumsum(is_positive)  # (N,)
    num_predictions = np.arange(1, len(is_positive) + 1)  # (N,)
    precision = tp_cumsum / num_predictions  # (N,)

    # 计算 recall
    recall = tp_cumsum / num_gt  # (N,)

    # 计算 AP（使用 11-point interpolation 或者所有点）
    # 这里使用所有点的方法（更精确）
    # AP = sum(precision[i] * is_positive[i]) / num_gt
    ap = np.sum(precision * is_positive) / num_gt

    return float(ap)


def calculate_sample_best_iou(
    pred_windows: Optional[List[Tuple[float, float]]],
    gt_windows: List[Tuple[float, float]],
) -> float:
    """Return the best IoU over all predicted and GT window pairs."""
    if not pred_windows or not gt_windows:
        return 0.0
    pred_array = np.array(pred_windows)
    gt_array = np.array(gt_windows)
    iou_matrix = compute_temporal_iou_batch(pred_array, gt_array)
    return float(np.max(iou_matrix)) if iou_matrix.size else 0.0


class QVHighlightsMetrics:
    """
    QVHighlights 数据集的 Moment Retrieval 评估指标

    主要指标:
    - R@1,θ: Recall@1 at various IoU thresholds (通常使用 0.5, 0.7)
    - mAP@θ: mean Average Precision at various IoU thresholds (通常使用 0.5, 0.75)
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        recall_thresholds: Optional[List[float]] = None,
        map_thresholds: Optional[List[float]] = None
    ):
        """
        初始化评估指标

        参数:
            iou_thresholds: 兼容参数，如果指定则同时用于 R@1 和 mAP
            recall_thresholds: R@1 的 IoU 阈值列表（默认 [0.5, 0.7]）
            map_thresholds: mAP 的 IoU 阈值列表（默认 [0.5, 0.75]）
        """
        # 如果指定了 iou_thresholds，则同时用于 R@1 和 mAP（向后兼容）
        if iou_thresholds is not None:
            self.recall_thresholds = sorted(iou_thresholds)
            self.map_thresholds = sorted(iou_thresholds)
        else:
            # 使用不同的默认阈值
            self.recall_thresholds = sorted(recall_thresholds if recall_thresholds else [0.5, 0.7])
            self.map_thresholds = sorted(map_thresholds if map_thresholds else [0.5, 0.75])

        self.reset()

    def reset(self):
        """重置所有指标"""
        self.total = 0
        self.parsed = 0  # 成功解析出至少一个预测片段的样本数
        self.iou_list = []  # 样本级 best IoU；无合法预测时记为 0

        # R@1 统计（使用 recall_thresholds）
        self.recall_at_1 = {th: 0 for th in self.recall_thresholds}

        # AP 列表（用于计算 mAP，使用 map_thresholds）
        self.ap_lists = {th: [] for th in self.map_thresholds}
        self.gt_duration_buckets = {
            group["group"]: {
                "group": group["group"],
                "metric_key": group["metric_key"],
                "duration_range": group["duration_range"],
                "min_duration_sec": group["min_duration_sec"],
                "max_duration_sec": group["max_duration_sec"],
                "total_samples": 0,
                "parsed_samples": 0,
                "evaluated_samples": 0,
                "iou_sum": 0.0,
            }
            for group in GT_DURATION_GROUP_SPECS
        }

    def _update_gt_duration_breakdown(
        self,
        gt_windows: List[Tuple[float, float]],
        pred_windows: Optional[List[Tuple[float, float]]],
        sample_iou: float,
    ) -> None:
        group = find_gt_duration_group(calculate_total_gt_duration(gt_windows))
        if group is None:
            return

        bucket = self.gt_duration_buckets[group["group"]]
        bucket["total_samples"] += 1
        if pred_windows:
            bucket["parsed_samples"] += 1
        bucket["evaluated_samples"] += 1
        bucket["iou_sum"] += float(sample_iou)

    def add(
        self,
        pred_windows: Optional[List[Tuple[float, float]]],
        pred_scores: Optional[List[float]],
        gt_windows: List[Tuple[float, float]]
    ):
        """
        添加一个样本的预测结果

        参数:
            pred_windows: 预测的时间窗口列表 [(start, end), ...]
                         如果为 None 或空列表，表示解析失败
            pred_scores: 对应的置信度分数列表（与 pred_windows 长度相同）
                        分数越高表示越可能是正确的片段
                        如果为 None，将使用递减的默认分数 [1.0, 0.9, 0.8, ...]
            gt_windows: GT 时间窗口列表 [(start, end), ...]
        """
        self.total += 1
        sample_iou = calculate_sample_best_iou(pred_windows, gt_windows)
        self.iou_list.append(sample_iou)
        self._update_gt_duration_breakdown(gt_windows, pred_windows, sample_iou)

        # 处理解析失败的情况
        if pred_windows is None or len(pred_windows) == 0:
            # 解析失败：R@1 = 0, AP = 0
            for th in self.map_thresholds:
                self.ap_lists[th].append(0.0)
            return

        self.parsed += 1

        # 如果没有提供分数，使用递减的默认分数
        if pred_scores is None or len(pred_scores) != len(pred_windows):
            pred_scores = [1.0 - i * 0.1 for i in range(len(pred_windows))]

        # 按分数排序预测（从高到低）
        sorted_indices = np.argsort(pred_scores)[::-1]
        sorted_pred_windows = [pred_windows[i] for i in sorted_indices]
        sorted_pred_scores = [pred_scores[i] for i in sorted_indices]

        # 计算 R@1 指标
        for th in self.recall_thresholds:
            hit = calculate_recall_at_k(
                sorted_pred_windows,
                gt_windows,
                iou_threshold=th,
                top_k=1
            )
            if hit:
                self.recall_at_1[th] += 1

        # 计算 mAP 指标
        for th in self.map_thresholds:
            ap = calculate_average_precision(
                sorted_pred_windows,
                sorted_pred_scores,
                gt_windows,
                iou_threshold=th
            )
            self.ap_lists[th].append(ap)

    def get_summary(self) -> Dict:
        """
        获取评估指标摘要

        返回:
            指标字典，包含 R@1 和 mAP
        """
        summary = {
            'total_samples': self.total,
            'parsed_samples': self.parsed,
            'parse_rate': self.parsed / max(self.total, 1) * 100,
            'evaluated_samples': len(self.iou_list),
            'mean_iou_definition': 'sample_best_iou; invalid_or_empty_prediction_counts_as_0',
            'gt_duration_definition': 'sum_of_gt_segment_lengths_sec',
        }

        if self.iou_list:
            summary['mean_iou'] = float(np.mean(self.iou_list))
            summary['median_iou'] = float(np.median(self.iou_list))
            summary['std_iou'] = float(np.std(self.iou_list))

        # R@1 指标
        for th in self.recall_thresholds:
            r1_count = self.recall_at_1[th]
            r1_rate = r1_count / max(self.total, 1) * 100
            summary[f'R@1,IoU={th}'] = r1_rate
            summary[f'R@1,IoU={th}_count'] = r1_count

        # mAP 指标
        for th in self.map_thresholds:
            ap_list = self.ap_lists[th]
            if ap_list:
                mean_ap = np.mean(ap_list)
                summary[f'mAP@IoU={th}'] = mean_ap * 100  # 转换为百分比
            else:
                summary[f'mAP@IoU={th}'] = 0.0

        summary['gt_duration_breakdown'] = self._get_gt_duration_breakdown_summary()
        for group_summary in summary['gt_duration_breakdown'].values():
            summary[group_summary['metric_key']] = group_summary.get('mean_iou')

        return summary

    def _get_gt_duration_breakdown_summary(self) -> Dict:
        breakdown = {}
        for group in GT_DURATION_GROUP_SPECS:
            stats = self.gt_duration_buckets[group["group"]]
            bucket_summary = {
                "group": stats["group"],
                "metric_key": stats["metric_key"],
                "duration_range": stats["duration_range"],
                "min_duration_sec": stats["min_duration_sec"],
                "max_duration_sec": stats["max_duration_sec"],
                "total_samples": stats["total_samples"],
                "parsed_samples": stats["parsed_samples"],
                "parse_rate": stats["parsed_samples"] / max(stats["total_samples"], 1) * 100,
                "evaluated_samples": stats["evaluated_samples"],
                "mean_iou": (
                    stats["iou_sum"] / stats["evaluated_samples"]
                    if stats["evaluated_samples"] > 0
                    else None
                ),
            }
            breakdown[group["group"]] = bucket_summary
        return breakdown

    def print_summary(self):
        """打印评估指标摘要"""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("QVHighlights Moment Retrieval 评估结果")
        print("=" * 70)
        print(f"总样本数: {summary['total_samples']}")
        print(f"成功解析样本数: {summary['parsed_samples']} ({summary['parse_rate']:.2f}%)")
        if 'mean_iou' in summary:
            print(f"样本级 best mIoU: {summary['mean_iou']:.4f}")

        print(f"\nRecall@1 (R@1) 指标:")
        for th in self.recall_thresholds:
            r1 = summary[f'R@1,IoU={th}']
            count = summary[f'R@1,IoU={th}_count']
            print(f"  R@1, IoU={th}: {r1:.2f}% ({count}/{summary['total_samples']})")

        print(f"\nmean Average Precision (mAP) 指标:")
        for th in self.map_thresholds:
            map_value = summary[f'mAP@IoU={th}']
            print(f"  mAP@IoU={th}: {map_value:.2f}%")

        print("\n按 GT 时长分组的 mIoU:")
        for group, bucket_summary in summary.get('gt_duration_breakdown', {}).items():
            mean_iou = bucket_summary.get('mean_iou')
            mean_text = "N/A" if mean_iou is None else f"{mean_iou:.4f}"
            print(
                f"  {group} {bucket_summary['duration_range']}: "
                f"samples={bucket_summary['total_samples']}, mIoU={mean_text}"
            )

        print("=" * 70)

    def get_detailed_stats(self) -> Dict:
        """
        获取详细统计信息

        返回:
            详细统计字典
        """
        stats = self.get_summary()

        if self.iou_list:
            stats['iou_distribution'] = {
                'min': float(np.min(self.iou_list)),
                'max': float(np.max(self.iou_list)),
                'median': float(np.median(self.iou_list)),
                'std': float(np.std(self.iou_list)),
                'q25': float(np.percentile(self.iou_list, 25)),
                'q75': float(np.percentile(self.iou_list, 75)),
            }

        # 添加 AP 分布统计
        for th in self.map_thresholds:
            ap_list = self.ap_lists[th]
            if ap_list:
                stats[f'AP@IoU={th}_distribution'] = {
                    'min': float(np.min(ap_list)),
                    'max': float(np.max(ap_list)),
                    'median': float(np.median(ap_list)),
                    'std': float(np.std(ap_list)),
                    'q25': float(np.percentile(ap_list, 25)),
                    'q75': float(np.percentile(ap_list, 75)),
                }

        return stats
