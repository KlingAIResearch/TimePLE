"""
Evaluation metrics for temporal grounding
"""

import logging
import math
from typing import Tuple, List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GT_DURATION_BUCKET_EDGES = [0.0, 10.0, 30.0]
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


def _format_duration_value(value: float) -> str:
    """格式化时长边界，避免标签里出现多余的小数点。"""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def normalize_gt_duration_bucket_edges(
    bucket_edges: Optional[List[float]]
) -> List[float]:
    """
    规范化 GT 时长分桶边界。

    约定:
    - 输入为递增边界列表，例如 [0, 10, 30]
    - 若首个边界大于 0，会自动补 0
    - 若最后一个边界不是无穷大，会自动补一个开放尾桶（+inf）
    """
    if not bucket_edges:
        normalized_edges = list(DEFAULT_GT_DURATION_BUCKET_EDGES)
    else:
        normalized_edges = []
        for idx, edge in enumerate(bucket_edges):
            if edge is None:
                if idx != len(bucket_edges) - 1:
                    raise ValueError("GT时长分桶边界中的 None 只能出现在最后一个位置")
                normalized_edges.append(float("inf"))
            else:
                normalized_edges.append(float(edge))

    if not normalized_edges:
        normalized_edges = list(DEFAULT_GT_DURATION_BUCKET_EDGES)

    if normalized_edges[0] > 0:
        normalized_edges.insert(0, 0.0)

    deduped_edges = [normalized_edges[0]]
    for edge in normalized_edges[1:]:
        if edge == deduped_edges[-1]:
            continue
        deduped_edges.append(edge)

    for i in range(1, len(deduped_edges)):
        if deduped_edges[i] < deduped_edges[i - 1]:
            raise ValueError("GT时长分桶边界必须严格递增")

    if not math.isinf(deduped_edges[-1]):
        deduped_edges.append(float("inf"))

    if len(deduped_edges) < 2:
        raise ValueError("GT时长分桶至少需要两个边界")

    return deduped_edges


def build_gt_duration_bucket_specs(bucket_edges: Optional[List[float]]) -> List[Dict]:
    """构建固定的 S/M/L GT 时长分组。

    `bucket_edges` 参数保留用于兼容旧配置；当前评测统一使用论文常见的
    Short (0,10]s、Medium (10,30]s、Long (30,∞)s 三组。
    """
    return [dict(bucket) for bucket in GT_DURATION_GROUP_SPECS]


def calculate_total_gt_duration(gt_timestamps: List[Tuple[float, float]]) -> float:
    """
    计算一个样本的 GT 总时长。

    对于多段 GT，使用所有 GT 片段长度之和作为难度分桶依据。
    """
    return float(sum(max(0.0, end - start) for start, end in gt_timestamps))


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


def calculate_multi_segment_iou(
    pred_list: List[Tuple[float, float]],
    gt_list: List[Tuple[float, float]]
) -> Tuple[List[float], Dict]:
    """
    计算多段时间戳的IoU（段级别统计）

    策略：
    1. 每个预测段与所有gt段计算IoU，取最大值作为该段的IoU
    2. 如果预测段数 < gt段数，漏掉的gt段贡献IoU=0
    3. 返回所有段的IoU列表（而非平均值），用于段级别统计

    参数:
        pred_list: 预测的时间段列表 [(start1, end1), (start2, end2), ...]
        gt_list: ground truth时间段列表 [(start1, end1), (start2, end2), ...]

    返回:
        (segment_ious, match_info): 段级别IoU列表和匹配详情
    """
    if not gt_list:
        return [], {
            'matched': False,
            'num_pred': len(pred_list) if pred_list else 0,
            'num_gt': 0,
            'reason': 'empty_gt'
        }

    if not pred_list:
        # 如果没有预测，所有gt段的IoU都是0
        num_gt = len(gt_list)
        return [0.0] * num_gt, {
            'matched': False,
            'num_pred': 0,
            'num_gt': num_gt,
            'reason': 'empty_prediction'
        }

    num_pred = len(pred_list)
    num_gt = len(gt_list)

    # 记录段数量是否匹配
    count_matched = (num_pred == num_gt)

    # 计算所有配对的IoU矩阵
    iou_matrix = np.zeros((num_pred, num_gt))
    for i, pred in enumerate(pred_list):
        for j, gt in enumerate(gt_list):
            iou_matrix[i, j] = calculate_iou(pred, gt)

    # 每个预测段取最佳gt段的IoU
    segment_ious = []
    for i in range(num_pred):
        best_iou = np.max(iou_matrix[i, :])
        segment_ious.append(float(best_iou))

    # 如果预测段数 < gt段数，补充漏预测的gt段（IoU=0）
    if num_pred < num_gt:
        num_missing = num_gt - num_pred
        segment_ious.extend([0.0] * num_missing)

    match_info = {
        'matched': count_matched,
        'num_pred': num_pred,
        'num_gt': num_gt,
        'num_segments': len(segment_ious),  # 实际统计的段数
        'segment_ious': segment_ious,
        'reason': 'matched' if count_matched else 'count_mismatch'
    }

    return segment_ious, match_info


class EvaluationMetrics:
    """
    评估指标管理类

    功能:
        - 累积样本结果
        - 支持单段和多段ground truth评估
        - 计算各种指标 (IoU, IoU@thresholds, 解析成功率等)
        - 生成评估报告
    """

    def __init__(
        self,
        iou_thresholds: List[float] = [0.3, 0.5, 0.7],
        gt_duration_bucket_edges: Optional[List[float]] = None
    ):
        """
        初始化评估指标

        参数:
            iou_thresholds: IoU阈值列表
            gt_duration_bucket_edges: GT时长分桶边界（单位：秒）
        """
        self.iou_thresholds = sorted(iou_thresholds)
        self.gt_duration_bucket_specs = build_gt_duration_bucket_specs(gt_duration_bucket_edges)
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.total = 0
        self.parsed = 0
        # 每个样本贡献一个 IoU。无合法预测时该样本 IoU 记为 0。
        self.iou_list = []
        # 保留段级 IoU，便于排查多段样本内部匹配情况。
        self.segment_iou_list = []
        self.threshold_counts = {th: 0 for th in self.iou_thresholds}

        # 多段评估相关统计
        self.multi_segment_count = 0  # 多段gt的样本数
        self.count_matched = 0  # 段数量匹配的样本数
        self.match_info_list = []  # 匹配详情列表

        # GT 时长分桶统计（用于难度分析）
        self.gt_duration_buckets = {
            bucket['group']: {
                'group': bucket['group'],
                'metric_key': bucket['metric_key'],
                'duration_range': bucket['duration_range'],
                'min_duration_sec': bucket['min_duration_sec'],
                'max_duration_sec': bucket['max_duration_sec'],
                'total_samples': 0,
                'parsed_samples': 0,
                'evaluated_samples': 0,
                'iou_sum': 0.0,
                'threshold_counts': {th: 0 for th in self.iou_thresholds},
            }
            for bucket in self.gt_duration_bucket_specs
        }

    def _find_gt_duration_bucket(self, gt_duration_sec: float) -> Optional[Dict]:
        """根据 GT 总时长找到对应分桶。

        分桶按论文常用的闭右开左区间定义：(0, 10]、(10, 30]、(30, ∞)。
        """
        for bucket in self.gt_duration_bucket_specs:
            start = bucket['min_duration_sec']
            end = bucket['max_duration_sec']
            if end is None:
                if gt_duration_sec > start:
                    return bucket
            elif start < gt_duration_sec <= end:
                return bucket
        return None

    def _update_gt_duration_breakdown(
        self,
        gt_timestamps: List[Tuple[float, float]],
        pred_timestamps: List[Tuple[float, float]],
        sample_iou: Optional[float]
    ):
        """更新按 GT 时长分桶的难度统计。"""
        gt_duration_sec = calculate_total_gt_duration(gt_timestamps)
        bucket_spec = self._find_gt_duration_bucket(gt_duration_sec)
        if bucket_spec is None:
            return

        bucket_stats = self.gt_duration_buckets[bucket_spec['group']]
        bucket_stats['total_samples'] += 1

        if pred_timestamps:
            bucket_stats['parsed_samples'] += 1

        if sample_iou is None:
            return

        bucket_stats['evaluated_samples'] += 1
        bucket_stats['iou_sum'] += float(sample_iou)

        for threshold in self.iou_thresholds:
            if sample_iou >= threshold:
                bucket_stats['threshold_counts'][threshold] += 1

    def add(
        self,
        pred_timestamps: Optional[List[Tuple[float, float]]],
        gt_timestamps: List[Tuple[float, float]]
    ) -> Optional[List[float]]:
        """
        添加一个样本的结果。

        参数:
            pred_timestamps: 预测的时间戳列表，None或空列表表示解析失败
            gt_timestamps: 真实的时间戳列表

        返回:
            段级别 IoU 列表。完全解析失败时也返回 [0.0]。
        """
        self.total += 1

        # 处理解析失败的情况
        if pred_timestamps is None:
            pred_timestamps = []

        # 判断是单段还是多段ground truth
        num_gt = len(gt_timestamps)

        if num_gt == 1:
            # 单段ground truth：取第一个预测段计算IoU
            if len(pred_timestamps) == 0:
                # 解析失败：该样本 IoU 按 0 计入整体 mIoU 和分桶 mIoU。
                sample_iou = 0.0
                self.iou_list.append(sample_iou)
                self.segment_iou_list.append(sample_iou)
                self._update_gt_duration_breakdown(gt_timestamps, pred_timestamps, sample_iou)
                return [sample_iou]

            self.parsed += 1
            pred = pred_timestamps[0]
            iou = calculate_iou(pred, gt_timestamps[0])
            self.iou_list.append(iou)
            self.segment_iou_list.append(iou)

            # 更新阈值计数
            for threshold in self.iou_thresholds:
                if iou >= threshold:
                    self.threshold_counts[threshold] += 1

            self._update_gt_duration_breakdown(gt_timestamps, pred_timestamps, iou)
            return [iou]

        else:
            # 多段ground truth：先计算段级 IoU，再聚合为一个样本级 IoU。
            self.multi_segment_count += 1

            segment_ious, match_info = calculate_multi_segment_iou(
                pred_timestamps, gt_timestamps
            )

            # 如果有任何有效的预测（即使不完整），也算解析成功
            if len(pred_timestamps) > 0:
                self.parsed += 1

            # 每个样本贡献一个 IoU 到整体 mIoU，同时保留段级 IoU 供诊断。
            sample_iou = float(np.mean(segment_ious)) if segment_ious else 0.0
            self.iou_list.append(sample_iou)
            self.segment_iou_list.extend(segment_ious)
            self.match_info_list.append(match_info)

            # 记录段数量是否匹配
            if match_info['matched']:
                self.count_matched += 1

            # 更新阈值计数（样本级）
            for threshold in self.iou_thresholds:
                if sample_iou >= threshold:
                    self.threshold_counts[threshold] += 1

            self._update_gt_duration_breakdown(gt_timestamps, pred_timestamps, sample_iou)
            return segment_ious if segment_ious else None

    def _get_gt_duration_breakdown_summary(self) -> Dict:
        """汇总按 GT 时长分桶的难度统计。"""
        breakdown = {}

        for bucket in self.gt_duration_bucket_specs:
            group = bucket['group']
            stats = self.gt_duration_buckets[group]

            bucket_summary = {
                'group': stats['group'],
                'metric_key': stats['metric_key'],
                'duration_range': stats['duration_range'],
                'min_duration_sec': stats['min_duration_sec'],
                'max_duration_sec': stats['max_duration_sec'],
                'total_samples': stats['total_samples'],
                'parsed_samples': stats['parsed_samples'],
                'parse_rate': stats['parsed_samples'] / max(stats['total_samples'], 1) * 100,
                'evaluated_samples': stats['evaluated_samples'],
            }

            if stats['evaluated_samples'] > 0:
                bucket_summary['mean_iou'] = stats['iou_sum'] / stats['evaluated_samples']
                for threshold in self.iou_thresholds:
                    count = stats['threshold_counts'][threshold]
                    bucket_summary[f'iou@{threshold}'] = count
                    bucket_summary[f'iou@{threshold}_rate'] = count / stats['evaluated_samples'] * 100
            else:
                bucket_summary['mean_iou'] = None

            breakdown[group] = bucket_summary

        return breakdown

    def get_summary(self) -> Dict:
        """
        获取评估指标摘要（样本级 mIoU）

        返回:
            指标字典
        """
        summary = {
            'total_samples': self.total,
            'parsed_samples': self.parsed,
            'parse_rate': self.parsed / max(self.total, 1) * 100,
            'evaluated_samples': len(self.iou_list),
            'total_segments': len(self.segment_iou_list),
            'mean_iou_definition': 'sample_mean_iou; invalid_or_empty_prediction_counts_as_0',
            'gt_duration_definition': 'sum_of_gt_segment_lengths_sec',
        }

        if len(self.iou_list) > 0:
            summary['mean_iou'] = np.mean(self.iou_list)
            summary['median_iou'] = np.median(self.iou_list)
            summary['std_iou'] = np.std(self.iou_list)

            # 样本级阈值统计
            for threshold in self.iou_thresholds:
                count = self.threshold_counts[threshold]
                summary[f'iou@{threshold}'] = count
                summary[f'iou@{threshold}_rate'] = count / len(self.iou_list) * 100

            if self.segment_iou_list:
                summary['segment_mean_iou'] = np.mean(self.segment_iou_list)

        # 添加多段评估统计
        if self.multi_segment_count > 0:
            summary['multi_segment_samples'] = self.multi_segment_count
            summary['count_matched'] = self.count_matched
            summary['count_match_rate'] = self.count_matched / self.multi_segment_count * 100

        summary['gt_duration_breakdown'] = self._get_gt_duration_breakdown_summary()
        for group_summary in summary['gt_duration_breakdown'].values():
            summary[group_summary['metric_key']] = group_summary.get('mean_iou')

        return summary

    def print_summary(self):
        """打印评估指标摘要（样本级 mIoU）"""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("评估结果汇总（样本级 mIoU）")
        print("=" * 70)
        print(f"总样本数: {summary['total_samples']}")
        print(f"成功解析样本数: {summary['parsed_samples']} ({summary['parse_rate']:.2f}%)")
        print(f"评估样本数: {summary['evaluated_samples']}")
        print(f"总段数: {summary['total_segments']}")

        if 'mean_iou' in summary:
            print(f"\n样本级 IoU 统计:")
            print(f"  平均 IoU: {summary['mean_iou']:.4f}")
            print(f"  中位数 IoU: {summary['median_iou']:.4f}")
            print(f"  IoU 标准差: {summary['std_iou']:.4f}")

            print(f"\n样本级 IoU 阈值统计:")
            for threshold in self.iou_thresholds:
                count = summary[f'iou@{threshold}']
                rate = summary[f'iou@{threshold}_rate']
                print(f"  IoU@{threshold}: {count} / {summary['evaluated_samples']} ({rate:.2f}%)")

        # 打印多段评估统计
        if 'multi_segment_samples' in summary:
            print("\n多段时序定位统计:")
            print(f"  多段样本数: {summary['multi_segment_samples']}")
            print(f"  段数量匹配: {summary['count_matched']} ({summary['count_match_rate']:.2f}%)")

        gt_duration_breakdown = summary.get('gt_duration_breakdown', {})
        non_empty_buckets = [
            (label, bucket_summary)
            for label, bucket_summary in gt_duration_breakdown.items()
            if bucket_summary['total_samples'] > 0
        ]
        if non_empty_buckets:
            print("\n按 GT 时长分桶的难度统计:")
            for label, bucket_summary in non_empty_buckets:
                line_parts = [
                    f"{label}",
                    f"samples={bucket_summary['total_samples']}",
                    f"parsed={bucket_summary['parsed_samples']} ({bucket_summary['parse_rate']:.2f}%)",
                    f"evaluated={bucket_summary['evaluated_samples']}",
                ]

                if 'mean_iou' in bucket_summary:
                    line_parts.append(f"mIoU={bucket_summary['mean_iou']:.4f}")
                    for threshold in self.iou_thresholds:
                        line_parts.append(
                            f"IoU@{threshold}={bucket_summary[f'iou@{threshold}_rate']:.2f}%"
                        )
                else:
                    line_parts.append("mIoU=N/A")

                print("  " + ", ".join(line_parts))

        print("=" * 70)

    def get_detailed_stats(self) -> Dict:
        """
        获取详细统计信息

        返回:
            详细统计字典
        """
        stats = self.get_summary()

        if self.iou_list:
            # 添加更多统计信息
            stats['iou_distribution'] = {
                'min': float(np.min(self.iou_list)),
                'max': float(np.max(self.iou_list)),
                'q25': float(np.percentile(self.iou_list, 25)),
                'q50': float(np.percentile(self.iou_list, 50)),
                'q75': float(np.percentile(self.iou_list, 75)),
                'q90': float(np.percentile(self.iou_list, 90)),
                'q95': float(np.percentile(self.iou_list, 95)),
            }

            # IoU分布区间统计
            bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            hist, _ = np.histogram(self.iou_list, bins=bins)
            stats['iou_histogram'] = {
                f'{bins[i]:.1f}-{bins[i+1]:.1f}': int(hist[i])
                for i in range(len(hist))
            }

        return stats
