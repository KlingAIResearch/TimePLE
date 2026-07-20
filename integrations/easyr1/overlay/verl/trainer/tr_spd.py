from __future__ import annotations

from typing import Any

import torch


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


def prepare_tr_spd_batch(data: Any) -> None:
    if "token_level_scores" not in data.batch:
        return

    rewards = data.batch["token_level_scores"].sum(dim=-1)
    data.batch["tr_spd_token_level_scores"] = data.batch["token_level_scores"].clone()
    data.batch["tr_spd_text_response_rewards"] = rewards.to(device=rewards.device, dtype=torch.float32)
    data.batch["token_level_scores"] = _put_response_rewards_on_last_token(data, rewards)
