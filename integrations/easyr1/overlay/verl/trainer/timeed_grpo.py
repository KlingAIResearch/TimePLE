from __future__ import annotations

from typing import Any

import numpy as np
import torch


def span_samples_per_response_from_config(config: Any) -> int:
    actor_config = getattr(getattr(config, "worker", None), "actor", None)
    if actor_config is None:
        return 1
    if not bool(getattr(actor_config, "timeed_span_grpo_enabled", False)):
        return 1
    # The current TimeED span policy enumerates all canonical cells inside the
    # actor update, so the rollout batch should stay at one row per response.
    return 1


def expand_batch_indices_for_span_samples(batch_size: int, span_samples_per_response: int) -> torch.Tensor:
    span_samples_per_response = max(int(span_samples_per_response), 1)
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}.")
    return torch.arange(batch_size, dtype=torch.long).repeat_interleave(span_samples_per_response)


def expand_dataproto_for_span_samples(data: Any, span_samples_per_response: int) -> Any:
    """Legacy utility retained for old configs; enumerated TimeED returns 1."""
    span_samples_per_response = max(int(span_samples_per_response), 1)
    if span_samples_per_response == 1:
        return data

    import copy

    indices = expand_batch_indices_for_span_samples(len(data), span_samples_per_response)
    expanded = data.index_select(indices)
    expanded.meta_info = copy.deepcopy(data.meta_info)
    expanded.meta_info["timeed_span_samples_per_response"] = span_samples_per_response
    if "timeed_response_uid" not in expanded.non_tensor_batch:
        prompt_uids = data.non_tensor_batch.get("uid")
        if prompt_uids is None:
            base_response_uids = np.asarray([f"timeed-response-{idx}" for idx in range(len(data))], dtype=object)
        else:
            base_response_uids = np.asarray(
                [f"{str(prompt_uid)}::timeed-response-{idx}" for idx, prompt_uid in enumerate(prompt_uids)],
                dtype=object,
            )
        expanded.non_tensor_batch["timeed_response_uid"] = np.repeat(
            base_response_uids,
            span_samples_per_response,
            axis=0,
        )
    if "timeed_span_sample_index" not in expanded.batch:
        sample_index = torch.arange(span_samples_per_response, dtype=torch.long).repeat(len(data))
        expanded.batch["timeed_span_sample_index"] = sample_index
    return expanded


def _put_response_rewards_on_last_token(data: Any, response_rewards: torch.Tensor) -> torch.Tensor:
    token_scores = torch.zeros_like(data.batch["token_level_scores"])
    response_mask = data.batch.get("response_mask")
    if response_mask is None:
        token_scores[:, -1] = response_rewards.to(device=token_scores.device, dtype=token_scores.dtype)
        return token_scores

    lengths = response_mask.long().sum(dim=-1).clamp_min(1) - 1
    row_idx = torch.arange(token_scores.size(0), device=token_scores.device, dtype=torch.long)
    token_scores[row_idx, lengths.to(device=token_scores.device)] = response_rewards.to(
        device=token_scores.device,
        dtype=token_scores.dtype,
    )
    return token_scores


def prepare_timeed_span_policy_batch(
    data: Any,
    *,
    text_reward_aggregation: str = "mean",
    preference_reward_gap: float = 0.05,
    eps: float = 1e-6,
) -> None:
    del text_reward_aggregation, preference_reward_gap, eps
    if "token_level_scores" not in data.batch:
        return

    # Reward function is text-format-only. Span rewards are recomputed as a
    # canonical-cell reward map inside the actor update from p_old, p_new and GT.
    rewards = data.batch["token_level_scores"].sum(dim=-1)
    data.batch["timeed_span_token_level_scores"] = data.batch["token_level_scores"].clone()
    data.batch["timeed_text_response_rewards"] = rewards.to(device=rewards.device, dtype=torch.float32)
    data.batch["token_level_scores"] = _put_response_rewards_on_last_token(data, rewards)


def attach_span_advantages_from_scores(data: Any, eps: float = 1e-6) -> None:
    prepare_timeed_span_policy_batch(data, eps=eps)
