from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
        return value.tolist()
    return value


def _normalize_nested_sequences(value: Any) -> list[list[Any]]:
    value = _to_list(value)
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [[value]]
    if len(value) == 0:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return [list(item) for item in value]
    return [list(value)]


def _extract_timespan_labels_per_sample(micro_batch: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    timespan_labels = _to_list(micro_batch.get("timespan_labels"))
    if isinstance(timespan_labels, list) and len(timespan_labels) > 0 and isinstance(timespan_labels[0], dict):
        starts_per_sample = [list(item.get("start", [])) for item in timespan_labels]
        ends_per_sample = [list(item.get("end", [])) for item in timespan_labels]
        return starts_per_sample, ends_per_sample
    if isinstance(timespan_labels, dict):
        return _normalize_nested_sequences(timespan_labels.get("start")), _normalize_nested_sequences(
            timespan_labels.get("end")
        )
    return [], []


def _extract_timespan_video_durations_per_sample(micro_batch: dict[str, Any]) -> list[list[float]]:
    timespan_labels = _to_list(micro_batch.get("timespan_labels"))
    if isinstance(timespan_labels, list) and len(timespan_labels) > 0 and isinstance(timespan_labels[0], dict):
        durations_per_sample: list[list[float]] = []
        for item in timespan_labels:
            sample_duration = _to_list(item.get("video_duration", []))
            if isinstance(sample_duration, tuple):
                sample_duration = list(sample_duration)
            if isinstance(sample_duration, list):
                durations_per_sample.append([float(value) for value in sample_duration])
            elif sample_duration is None:
                durations_per_sample.append([])
            else:
                durations_per_sample.append([float(sample_duration)])
        return durations_per_sample
    if isinstance(timespan_labels, dict) and "video_duration" in timespan_labels:
        return _normalize_nested_sequences(timespan_labels.get("video_duration"))

    explicit_durations = _to_list(micro_batch.get("timespan_video_durations"))
    if explicit_durations is not None:
        return _normalize_nested_sequences(explicit_durations)
    return []


def _resolve_decode_video_durations(
    *,
    target_durations: list[float],
    target_ends: list[float],
    prediction_count: int,
) -> list[float]:
    if prediction_count <= 0:
        return []
    if len(target_durations) > 0:
        return [float(target_durations[0])] * prediction_count
    if len(target_ends) > 0:
        return [max(float(max(target_ends)), 1.0)] * prediction_count
    return [1.0] * prediction_count


def resolve_tr_spd_duration_tensor(
    *,
    micro_batch: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts_per_sample, ends_per_sample = _extract_timespan_labels_per_sample(micro_batch)
    durations_per_sample = _extract_timespan_video_durations_per_sample(micro_batch)
    durations: list[float] = []
    target_valid: list[float] = []
    for sample_idx in range(batch_size):
        target_starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
        target_ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
        target_count = min(len(target_starts), len(target_ends))
        if target_count <= 0:
            durations.append(1.0)
            target_valid.append(0.0)
            continue
        target_durations = durations_per_sample[sample_idx] if sample_idx < len(durations_per_sample) else []
        resolved = _resolve_decode_video_durations(
            target_durations=target_durations,
            target_ends=target_ends[:target_count],
            prediction_count=1,
        )
        durations.append(float(resolved[0]) if len(resolved) > 0 else max(max(target_ends[:target_count]), 1.0))
        target_valid.append(1.0)
    return (
        torch.as_tensor(durations, device=device, dtype=torch.float32).clamp_min(1e-6),
        torch.as_tensor(target_valid, device=device, dtype=torch.float32),
    )


def _tensor_interval_iou(
    pred_start: torch.Tensor,
    pred_end: torch.Tensor,
    target_start: torch.Tensor,
    target_end: torch.Tensor,
) -> torch.Tensor:
    pred_lo = torch.minimum(pred_start, pred_end)
    pred_hi = torch.maximum(pred_start, pred_end)
    target_lo = torch.minimum(target_start, target_end)
    target_hi = torch.maximum(target_start, target_end)
    inter = (torch.minimum(pred_hi, target_hi) - torch.maximum(pred_lo, target_lo)).clamp_min(0.0)
    pred_len = (pred_hi - pred_lo).clamp_min(0.0)
    target_len = (target_hi - target_lo).clamp_min(0.0)
    union = (pred_len + target_len - inter).clamp_min(1e-8)
    return torch.where((pred_len > 0) & (target_len > 0), inter / union, torch.zeros_like(inter))


def _spod_reward_from_seconds(
    *,
    pred_start_sec: torch.Tensor,
    pred_end_sec: torch.Tensor,
    micro_batch: dict[str, Any],
    durations: torch.Tensor,
    reward_type: str,
    boundary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts_per_sample, ends_per_sample = _extract_timespan_labels_per_sample(micro_batch)
    device = pred_start_sec.device
    reward = torch.zeros_like(pred_start_sec, dtype=torch.float32)
    valid = torch.zeros(pred_start_sec.shape[0], device=device, dtype=torch.float32)
    reward_type = str(reward_type)
    if reward_type not in {"iou", "iou_boundary"}:
        raise ValueError(f"Unsupported tr_spd_reward_type: {reward_type!r}.")

    for sample_idx in range(pred_start_sec.shape[0]):
        target_starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
        target_ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
        target_count = min(len(target_starts), len(target_ends))
        if target_count <= 0:
            continue

        pred_start_flat = pred_start_sec[sample_idx].float().reshape(1, -1)
        pred_end_flat = pred_end_sec[sample_idx].float().reshape(1, -1)
        target_start = torch.as_tensor(target_starts[:target_count], device=device, dtype=torch.float32).reshape(-1, 1)
        target_end = torch.as_tensor(target_ends[:target_count], device=device, dtype=torch.float32).reshape(-1, 1)
        candidate_reward = _tensor_interval_iou(pred_start_flat, pred_end_flat, target_start, target_end)
        if reward_type == "iou_boundary" and float(boundary_weight) > 0.0:
            duration = durations[sample_idx].float().clamp_min(1e-6)
            boundary = (pred_start_flat - target_start).abs() + (pred_end_flat - target_end).abs()
            candidate_reward = candidate_reward - float(boundary_weight) * boundary / duration
        reward[sample_idx] = candidate_reward.amax(dim=0).reshape_as(pred_start_sec[sample_idx]).to(reward)
        valid[sample_idx] = 1.0
    return reward, valid


def _spod_reward_from_uv(
    *,
    codec: nn.Module,
    features: torch.Tensor,
    uv: torch.Tensor,
    micro_batch: dict[str, Any],
    durations: torch.Tensor,
    reward_type: str,
    boundary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(codec, "decode_uv_with_features_seconds"):
        raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires codec.decode_uv_with_features_seconds for decoder parity.")
    pred_start_sec, pred_end_sec, _ = codec.decode_uv_with_features_seconds(
        features=features,
        uv=uv,
        video_duration_sec=durations,
        hard=True,
    )
    return _spod_reward_from_seconds(
        pred_start_sec=pred_start_sec,
        pred_end_sec=pred_end_sec,
        micro_batch=micro_batch,
        durations=durations,
        reward_type=reward_type,
        boundary_weight=boundary_weight,
    )


def _cell_coordinates(decoder: nn.Module, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    u_centers = getattr(decoder, "u_centers", None)
    v_centers = getattr(decoder, "v_centers", None)
    if not isinstance(u_centers, torch.Tensor) or not isinstance(v_centers, torch.Tensor):
        raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires decoder.u_centers and decoder.v_centers buffers.")
    num_u_bins = int(getattr(decoder, "num_u_bins", u_centers.numel()))
    num_v_bins = int(getattr(decoder, "num_v_bins", v_centers.numel()))
    flat_u = u_centers.to(device=device, dtype=dtype).reshape(-1).repeat_interleave(num_v_bins)
    flat_v = v_centers.to(device=device, dtype=dtype).reshape(-1).repeat(num_u_bins)
    return torch.stack([flat_u, flat_v], dim=-1)


def _canonical_gt_uv(
    *,
    codec: nn.Module,
    micro_batch: dict[str, Any],
    durations: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts_per_sample, ends_per_sample = _extract_timespan_labels_per_sample(micro_batch)
    gt_uv = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
    valid = torch.zeros(batch_size, device=device, dtype=torch.float32)
    if not hasattr(codec, "_to_relative"):
        raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires codec._to_relative to map GT seconds to canonical coordinates.")
    transform = getattr(codec, "transform", None)
    if transform is None or not hasattr(transform, "interval_to_square"):
        raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires codec.transform.interval_to_square.")

    for sample_idx in range(batch_size):
        target_starts = starts_per_sample[sample_idx] if sample_idx < len(starts_per_sample) else []
        target_ends = ends_per_sample[sample_idx] if sample_idx < len(ends_per_sample) else []
        target_count = min(len(target_starts), len(target_ends))
        if target_count <= 0:
            continue
        start_sec = torch.as_tensor([float(target_starts[0])], device=device, dtype=torch.float32)
        end_sec = torch.as_tensor([float(target_ends[0])], device=device, dtype=torch.float32)
        start_rel, end_rel, _ = codec._to_relative(start_sec, end_sec, durations[sample_idx : sample_idx + 1])
        u, v, _ = transform.interval_to_square(start_rel, end_rel)
        gt_uv[sample_idx, 0] = u.reshape(-1)[0].float()
        gt_uv[sample_idx, 1] = v.reshape(-1)[0].float()
        valid[sample_idx] = 1.0
    return gt_uv, valid


def _support_threshold(support_mode: str) -> Optional[float]:
    mode = str(support_mode).replace("-", "_").lower()
    if mode in {"none", "global", ""}:
        return None
    if mode in {"gaussian_95", "gaussian95"}:
        return 5.99
    if mode in {"gaussian_99", "gaussian99"}:
        return 9.21
    raise ValueError(f"Unsupported tr_spd_support_mode: {support_mode!r}.")


def _masked_mean_float(values: torch.Tensor, mask: torch.Tensor) -> float:
    valid_count = mask.float().sum().clamp_min(1.0)
    return float(((values.float().reshape(-1) * mask.float().reshape(-1)).sum() / valid_count).detach().item())


def _zero_metrics(tau_candidates: Sequence[float]) -> dict[str, float]:
    metrics = {
        "tr_spd_loss": 0.0,
        "tr_spd_span_ce": 0.0,
        "tr_spd_reject_retention_loss": 0.0,
        "tr_spd_accept_rate": 0.0,
        "tr_spd_valid_frac": 0.0,
        "tr_spd_reward_prior": 0.0,
        "tr_spd_reward_posterior": 0.0,
        "tr_spd_reward_current": 0.0,
        "tr_spd_delta_reward_mean": 0.0,
        "tr_spd_target_entropy": 0.0,
        "tr_spd_prior_entropy": 0.0,
        "tr_spd_current_entropy": 0.0,
        "tr_spd_kl_target_to_prior": 0.0,
        "tr_spd_kl_target_to_student": 0.0,
        "tr_spd_kl_prior_to_student_reject": 0.0,
        "tr_spd_trust_region_pass_rate": 0.0,
        "tr_spd_trust_region_reject_rate": 0.0,
        "tr_spd_trust_region_kl_budget": 0.0,
        "tr_spd_support_mass_prior": 1.0,
        "tr_spd_improvement_weight_mean": 0.0,
    }
    for tau in tau_candidates:
        metrics[f"tr_spd_best_tau_{float(tau):g}"] = 0.0
    return metrics


def _as_tau_list(tau_candidates: Iterable[float]) -> list[float]:
    taus = [float(tau) for tau in tau_candidates]
    if len(taus) == 0:
        raise ValueError("tr_spd_tau_candidates must not be empty.")
    for tau in taus:
        if tau <= 0.0:
            raise ValueError(f"tr_spd_tau_candidates must be positive, got {tau}.")
    return taus


def compute_tr_spd_loss(
    *,
    current_logits: torch.Tensor,
    current_features: torch.Tensor,
    old_logits: torch.Tensor,
    old_features: torch.Tensor,
    micro_batch: dict[str, Any],
    sample_mask: torch.Tensor,
    codec: nn.Module,
    tau_candidates: Iterable[float],
    support_mode: str,
    accept_delta: float,
    trust_region_kl_budget: Optional[float],
    rejected_retention_weight: float,
    use_improvement_weight: bool,
    improvement_gamma: float,
    improvement_scale: float,
    improvement_max_extra_weight: float,
    reward_type: str,
    boundary_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if current_logits is None or old_logits is None or current_features is None or old_features is None:
        raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires current/old logits and current/old decoder features.")
    if current_logits.shape != old_logits.shape:
        raise ValueError(f"CIS Trust-Region Span Posterior Distillation (TR-SPD) current/old logits shape mismatch: {current_logits.shape} vs {old_logits.shape}.")
    if current_logits.dim() != 3:
        raise ValueError(f"CIS Trust-Region Span Posterior Distillation (TR-SPD) logits must have shape [B, U, V], got {tuple(current_logits.shape)}.")
    if current_features.shape != old_features.shape or current_features.size(0) != current_logits.size(0):
        raise ValueError(
            "CIS Trust-Region Span Posterior Distillation (TR-SPD) feature shape mismatch: "
            f"current={tuple(current_features.shape)}, old={tuple(old_features.shape)}, logits={tuple(current_logits.shape)}."
        )
    if not hasattr(codec, "decode_uv_with_features_seconds"):
        raise ValueError("CIS Trust-Region Span Posterior Distillation (TR-SPD) requires codec.decode_uv_with_features_seconds for decoder parity.")

    taus = _as_tau_list(tau_candidates)
    if trust_region_kl_budget is not None and float(trust_region_kl_budget) <= 0.0:
        raise ValueError("tr_spd_trust_region_kl_budget must be positive when set.")
    device = current_logits.device
    zero_loss = current_logits.sum() * 0.0
    metrics = _zero_metrics(taus)
    if trust_region_kl_budget is not None:
        metrics["tr_spd_trust_region_kl_budget"] = float(trust_region_kl_budget)
    batch_size = current_logits.size(0)
    sample_mask = (sample_mask.float().reshape(-1).to(device=device) > 0).float()
    if sample_mask.numel() != batch_size:
        raise ValueError(f"CIS Trust-Region Span Posterior Distillation (TR-SPD) sample_mask size mismatch: {sample_mask.numel()} vs {batch_size}.")

    durations, target_valid = resolve_tr_spd_duration_tensor(
        micro_batch=micro_batch,
        batch_size=batch_size,
        device=device,
    )
    gt_uv, gt_valid = _canonical_gt_uv(
        codec=codec,
        micro_batch=micro_batch,
        durations=durations,
        batch_size=batch_size,
        device=device,
    )
    mask = sample_mask * target_valid * gt_valid
    metrics["tr_spd_valid_frac"] = float(mask.mean().detach().item()) if batch_size > 0 else 0.0
    if mask.sum().item() <= 0.0:
        return zero_loss, metrics

    old_logits = old_logits.to(device=device).float().detach()
    old_features = old_features.to(device=device).detach()
    current_logp = F.log_softmax(current_logits.float().flatten(1), dim=-1)
    current_p = current_logp.exp()
    old_logp = F.log_softmax(old_logits.flatten(1), dim=-1).detach()
    old_p = old_logp.exp()
    coords = _cell_coordinates(codec.decoder, device=device, dtype=current_logp.dtype)
    sigma_u = max(float(getattr(codec.decoder, "span_sigma_u", getattr(codec.encoder, "span_sigma_u", 0.1))), 1e-6)
    sigma_v = max(float(getattr(codec.decoder, "span_sigma_v", getattr(codec.encoder, "span_sigma_v", 0.1))), 1e-6)

    with torch.no_grad():
        du = (coords[:, 0].unsqueeze(0) - gt_uv[:, 0:1]) / sigma_u
        dv = (coords[:, 1].unsqueeze(0) - gt_uv[:, 1:2]) / sigma_v
        distance_sq = du.square() + dv.square()
        energy = 0.5 * distance_sq
        threshold = _support_threshold(support_mode)
        if threshold is None:
            support = torch.ones_like(energy, dtype=torch.bool)
        else:
            support = distance_sq <= float(threshold)
        support_mass_prior = (old_p * support.float()).sum(dim=-1)

        mu_prior = old_p @ coords
        reward_prior, reward_valid = _spod_reward_from_uv(
            codec=codec,
            features=old_features,
            uv=mu_prior,
            micro_batch=micro_batch,
            durations=durations,
            reward_type=reward_type,
            boundary_weight=boundary_weight,
        )
        mask = mask * reward_valid
        if mask.sum().item() <= 0.0:
            metrics["tr_spd_valid_frac"] = float(mask.mean().detach().item()) if batch_size > 0 else 0.0
            return zero_loss, metrics

        candidate_qs: list[torch.Tensor] = []
        candidate_rewards: list[torch.Tensor] = []
        candidate_kls: list[torch.Tensor] = []
        candidate_entropies: list[torch.Tensor] = []
        for tau in taus:
            posterior_logits = old_logp - energy / max(float(tau), 1e-6)
            posterior_logits = posterior_logits.masked_fill(~support, -torch.inf)
            empty_support = ~torch.isfinite(posterior_logits).any(dim=-1, keepdim=True)
            posterior_logits = torch.where(empty_support, old_logp, posterior_logits)
            q_tau = F.softmax(posterior_logits, dim=-1)
            q_log_tau = q_tau.clamp_min(1e-12).log()
            mu_tau = q_tau @ coords
            reward_tau, _ = _spod_reward_from_uv(
                codec=codec,
                features=old_features,
                uv=mu_tau,
                micro_batch=micro_batch,
                durations=durations,
                reward_type=reward_type,
                boundary_weight=boundary_weight,
            )
            candidate_qs.append(q_tau)
            candidate_rewards.append(reward_tau)
            candidate_kls.append((q_tau * (q_log_tau - old_logp)).sum(dim=-1))
            candidate_entropies.append(-(q_tau * q_log_tau).sum(dim=-1))

        reward_stack = torch.stack(candidate_rewards, dim=-1)
        kl_stack = torch.stack(candidate_kls, dim=-1)
        entropy_stack = torch.stack(candidate_entropies, dim=-1)
        if trust_region_kl_budget is None:
            candidate_allowed = torch.ones_like(reward_stack, dtype=torch.bool)
        else:
            candidate_allowed = kl_stack <= float(trust_region_kl_budget)
        selection_reward = reward_stack.masked_fill(~candidate_allowed, -torch.inf)
        has_trust_region_candidate = torch.isfinite(selection_reward).any(dim=-1)
        fallback_reward = torch.where(has_trust_region_candidate.unsqueeze(-1), selection_reward, reward_stack)
        raw_best_reward, best_idx = fallback_reward.max(dim=-1)
        best_reward = torch.where(has_trust_region_candidate, raw_best_reward, reward_prior)
        q_stack = torch.stack(candidate_qs, dim=1)
        best_gather_index = best_idx.reshape(-1, 1, 1).expand(-1, 1, q_stack.size(-1))
        q_best = q_stack.gather(dim=1, index=best_gather_index).squeeze(1).detach()
        q_kl_best = kl_stack.gather(dim=1, index=best_idx.reshape(-1, 1)).squeeze(1)
        q_entropy_best = entropy_stack.gather(dim=1, index=best_idx.reshape(-1, 1)).squeeze(1)
        delta_reward = best_reward - reward_prior
        trust_region_mask = has_trust_region_candidate.float() * mask
        accept_mask = (delta_reward > float(accept_delta)).float() * trust_region_mask

        if use_improvement_weight:
            scaled = (delta_reward - float(accept_delta)) / max(float(improvement_scale), 1e-6)
            extra = scaled.clamp(min=0.0, max=float(improvement_max_extra_weight))
            improvement_weight = 1.0 + float(improvement_gamma) * extra
        else:
            improvement_weight = torch.ones_like(delta_reward)
        improvement_weight = improvement_weight * accept_mask

        old_logp_safe = old_logp
        q_entropy = q_entropy_best
        prior_entropy = -(old_p * old_logp_safe).sum(dim=-1)
        q_kl_old = q_kl_best
        mu_cur = current_p.detach() @ coords
        reward_cur, _ = _spod_reward_from_uv(
            codec=codec,
            features=current_features.detach(),
            uv=mu_cur,
            micro_batch=micro_batch,
            durations=durations,
            reward_type=reward_type,
            boundary_weight=boundary_weight,
        )

    ce_target = -(q_best * current_logp).sum(dim=-1)
    target_entropy = -(q_best * q_best.clamp_min(1e-12).log()).sum(dim=-1)
    true_kl_target = ce_target - target_entropy
    retention_kl = (old_p * (old_logp - current_logp)).sum(dim=-1)

    accepted_weight = improvement_weight
    reject_weight = (1.0 - accept_mask) * mask * max(float(rejected_retention_weight), 0.0)
    denom = (accepted_weight + reject_weight).sum().clamp_min(1e-8)
    loss = ((accepted_weight * true_kl_target) + (reject_weight * retention_kl)).sum() / denom

    with torch.no_grad():
        accepted_count = accept_mask.sum().clamp_min(1.0)
        rejected_count = ((1.0 - accept_mask) * mask).sum().clamp_min(1.0)
        metrics.update(
            {
                "tr_spd_loss": float(loss.detach().item()),
                "tr_spd_span_ce": _masked_mean_float(ce_target, accept_mask),
                "tr_spd_reject_retention_loss": _masked_mean_float(
                    retention_kl,
                    (1.0 - accept_mask) * mask,
                ),
                "tr_spd_accept_rate": float((accept_mask.sum() / mask.sum().clamp_min(1.0)).detach().item()),
                "tr_spd_valid_frac": float(mask.mean().detach().item()) if batch_size > 0 else 0.0,
                "tr_spd_reward_prior": _masked_mean_float(reward_prior, mask),
                "tr_spd_reward_posterior": _masked_mean_float(best_reward, mask),
                "tr_spd_reward_current": _masked_mean_float(reward_cur, mask),
                "tr_spd_delta_reward_mean": _masked_mean_float(delta_reward, mask),
                "tr_spd_target_entropy": _masked_mean_float(q_entropy, accept_mask),
                "tr_spd_prior_entropy": _masked_mean_float(prior_entropy, mask),
                "tr_spd_current_entropy": _masked_mean_float(
                    -(current_p.detach() * current_logp.detach()).sum(dim=-1),
                    mask,
                ),
                "tr_spd_kl_target_to_prior": _masked_mean_float(q_kl_old, accept_mask),
                "tr_spd_kl_target_to_student": _masked_mean_float(true_kl_target, accept_mask),
                "tr_spd_kl_prior_to_student_reject": _masked_mean_float(
                    retention_kl,
                    (1.0 - accept_mask) * mask,
                ),
                "tr_spd_trust_region_pass_rate": float(
                    (trust_region_mask.sum() / mask.sum().clamp_min(1.0)).detach().item()
                ),
                "tr_spd_trust_region_reject_rate": float(
                    (((1.0 - has_trust_region_candidate.float()) * mask).sum() / mask.sum().clamp_min(1.0))
                    .detach()
                    .item()
                ),
                "tr_spd_support_mass_prior": _masked_mean_float(support_mass_prior, mask),
                "tr_spd_improvement_weight_mean": float(
                    (accepted_weight.sum() / accepted_count).detach().item()
                ),
            }
        )
        for idx, tau in enumerate(taus):
            tau_mask = (best_idx == idx).float() * accept_mask
            metrics[f"tr_spd_best_tau_{float(tau):g}"] = float(
                (tau_mask.sum() / mask.sum().clamp_min(1.0)).detach().item()
            )

    return loss, metrics
