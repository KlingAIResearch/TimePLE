"""
Loss utilities for TimePLE-Codec.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_target_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0 or not torch.any(mask):
        return values.new_tensor(0.0)
    masked_values = values[mask]
    if masked_values.numel() == 0:
        return values.new_tensor(0.0)
    return masked_values.mean()


def interval_iou(
    pred_start: torch.Tensor,
    pred_end: torch.Tensor,
    target_start: torch.Tensor,
    target_end: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_start = torch.minimum(pred_start, pred_end)
    pred_end = torch.maximum(pred_start, pred_end)
    target_start = torch.minimum(target_start, target_end)
    target_end = torch.maximum(target_start, target_end)

    inter = (torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start)).clamp(min=0.0)
    union = (pred_end - pred_start) + (target_end - target_start) - inter
    return inter / (union + eps)


def interval_giou_loss(
    pred_start: torch.Tensor,
    pred_end: torch.Tensor,
    target_start: torch.Tensor,
    target_end: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_start = torch.minimum(pred_start, pred_end)
    pred_end = torch.maximum(pred_start, pred_end)
    target_start = torch.minimum(target_start, target_end)
    target_end = torch.maximum(target_start, target_end)

    inter = (torch.minimum(pred_end, target_end) - torch.maximum(pred_start, target_start)).clamp(min=0.0)
    union = (pred_end - pred_start) + (target_end - target_start) - inter
    enclosing = torch.maximum(pred_end, target_end) - torch.minimum(pred_start, target_start)
    iou = inter / (union + eps)
    giou = iou - (enclosing - union) / (enclosing + eps)
    return 1.0 - giou
