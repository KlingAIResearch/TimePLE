from __future__ import annotations

from typing import Any

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.worker.block_table import BlockTable


def compute_required_block_counts(
    *,
    model_runner: Any,
    seq_len: int,
) -> dict[int, int]:
    required: dict[int, int] = {}
    for kv_cache_gid, kv_cache_group in enumerate(model_runner.kv_cache_config.kv_cache_groups):
        if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        base_block_table = model_runner.input_batch.block_table[kv_cache_gid]
        original_block_size = (
            base_block_table.block_size * base_block_table.blocks_per_kv_block
        )
        required[kv_cache_gid] = max(1, (seq_len + original_block_size - 1) // original_block_size)
    return required


def build_temp_block_table(
    *,
    source_block_table: BlockTable,
    block_ids: list[int],
    seq_len: int,
) -> BlockTable:
    original_block_size = (
        source_block_table.block_size * source_block_table.blocks_per_kv_block
    )
    temp_block_table = BlockTable(
        block_size=original_block_size,
        max_num_reqs=1,
        max_num_blocks_per_req=max(1, len(block_ids)),
        max_num_batched_tokens=max(1, seq_len),
        pin_memory=source_block_table.pin_memory,
        device=source_block_table.device,
        kernel_block_size=source_block_table.block_size,
        cp_kv_cache_interleave_size=source_block_table.cp_kv_cache_interleave_size,
    )
    temp_block_table.add_row(block_ids, 0)
    temp_block_table.commit_block_table(1)
    return temp_block_table


def collect_full_mm_embeddings(
    model_runner: Any,
    req_state: Any,
    seq_len: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    is_mm_embed = torch.zeros(seq_len, dtype=torch.bool, device=model_runner.device)
    mm_embeds: list[torch.Tensor] = []

    for mm_feature in req_state.mm_features:
        pos_info = mm_feature.mm_position
        start_pos = int(pos_info.offset)
        num_encoder_tokens = int(pos_info.length)
        end_pos = start_pos + num_encoder_tokens
        if end_pos > seq_len:
            raise ValueError(
                f"Multimodal placeholder overflows sequence: end_pos={end_pos} seq_len={seq_len}"
            )

        encoder_output = model_runner.encoder_cache.get(mm_feature.identifier)
        if encoder_output is None:
            raise KeyError(f"Missing encoder cache for mm feature {mm_feature.identifier}")
        if torch.is_tensor(encoder_output) and encoder_output.device != model_runner.device:
            encoder_output = encoder_output.to(model_runner.device, non_blocking=True)

        if (embed_mask := pos_info.is_embed) is not None:
            if torch.is_tensor(embed_mask):
                embed_mask = embed_mask[:num_encoder_tokens].to(
                    device=model_runner.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
            else:
                embed_mask = torch.as_tensor(
                    embed_mask[:num_encoder_tokens],
                    device=model_runner.device,
                    dtype=torch.bool,
                )
            curr_start, curr_end = pos_info.get_embeds_indices_in_range(0, num_encoder_tokens)
            if curr_start != curr_end:
                mm_embeds.append(encoder_output[curr_start:curr_end])
            is_mm_embed[start_pos:end_pos] |= embed_mask
        else:
            mm_embeds.append(encoder_output[:num_encoder_tokens])
            is_mm_embed[start_pos:end_pos] = True

    return mm_embeds, is_mm_embed


def build_full_forward_positions(
    *,
    model_runner: Any,
    model: Any,
    req_state: Any,
    full_token_ids: list[int],
    mm_embeds: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    seq_len = len(full_token_ids)
    prompt_len = len(req_state.prompt_token_ids)
    completion_len = seq_len - prompt_len

    if model_runner.uses_mrope:
        prompt_positions = getattr(req_state, "mrope_positions", None)
        if prompt_positions is None:
            positions, _ = model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                req_state.mm_features,
            )
            prompt_positions = positions

        prompt_positions = prompt_positions[:, :prompt_len]
        positions = torch.empty(
            (prompt_positions.shape[0], seq_len),
            dtype=prompt_positions.dtype,
            device=prompt_positions.device,
        )
        positions[:, :prompt_len] = prompt_positions
        if completion_len > 0:
            mrope_position_delta = int(getattr(req_state, "mrope_position_delta", 0))
            completion_positions = torch.arange(
                mrope_position_delta + prompt_len,
                mrope_position_delta + prompt_len + completion_len,
                dtype=prompt_positions.dtype,
                device=prompt_positions.device,
            )
            positions[:, prompt_len:] = completion_positions.unsqueeze(0).expand(
                prompt_positions.shape[0],
                -1,
            )
        positions = positions.to(model_runner.device, non_blocking=True)
        return mm_embeds, positions

    if model_runner.uses_xdrope_dim > 0:
        positions = model.get_xdrope_input_positions(
            full_token_ids,
            req_state.mm_features,
        )
        return mm_embeds, positions.to(model_runner.device, non_blocking=True)

    positions = torch.arange(seq_len, device=model_runner.device, dtype=torch.int64)
    return mm_embeds, positions


def build_full_forward_attention(
    *,
    model_runner: Any,
    req_state: Any,
    seq_len: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    query_start_loc_cpu = torch.tensor([0, seq_len], dtype=torch.int32, device="cpu")
    query_start_loc = query_start_loc_cpu.to(model_runner.device, non_blocking=True)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32, device="cpu")
    seq_lens = seq_lens_cpu.to(model_runner.device, non_blocking=True)
    num_computed_tokens_cpu = torch.tensor([0], dtype=torch.int32, device="cpu")
    is_prefilling = torch.tensor([True], dtype=torch.bool, device=model_runner.device)
    scalar_positions = torch.arange(seq_len, device=model_runner.device, dtype=torch.int64)

    attn_metadata: dict[str, Any] = {}
    slot_mapping_by_layer: dict[str, torch.Tensor] = {}

    for kv_cache_gid, kv_cache_group in enumerate(model_runner.kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue

        base_block_table = model_runner.input_batch.block_table[kv_cache_gid]
        temp_block_table = build_temp_block_table(
            source_block_table=base_block_table,
            block_ids=req_state.block_ids[kv_cache_gid],
            seq_len=seq_len,
        )
        temp_block_table.compute_slot_mapping(
            1,
            query_start_loc,
            scalar_positions,
        )
        block_table_tensor = temp_block_table.get_device_tensor(1)
        slot_mapping = temp_block_table.slot_mapping.gpu[:seq_len]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=1,
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
            is_prefilling=is_prefilling,
        )

        for attn_group in model_runner.attn_groups[kv_cache_gid]:
            builder = attn_group.get_metadata_builder()
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
                slot_mapping_by_layer[layer_name] = slot_mapping

    return attn_metadata, slot_mapping_by_layer


def extract_real_step_sampled_token_ids(output: Any) -> list[list[int]] | None:
    if output is None:
        return None

    if hasattr(output, "sampled_token_ids_cpu") and hasattr(output, "async_copy_ready_event"):
        output.async_copy_ready_event.synchronize()
        raw_step_ids = output.sampled_token_ids_cpu.tolist()
        invalid_req_indices = set(getattr(output, "_invalid_req_indices", []) or [])
        normalized_step_ids: list[list[int]] = []
        for req_idx, row in enumerate(raw_step_ids):
            if req_idx in invalid_req_indices:
                normalized_step_ids.append([])
                continue
            if -1 in row:
                row = row[: row.index(-1)]
            normalized_step_ids.append([int(token_id) for token_id in row])
        return normalized_step_ids

    raw_step_ids = getattr(output, "sampled_token_ids", None)
    if raw_step_ids is None:
        return None
    if torch.is_tensor(raw_step_ids):
        raw_step_ids = raw_step_ids.tolist()

    normalized_step_ids = []
    for row in raw_step_ids:
        normalized_step_ids.append([int(token_id) for token_id in row if int(token_id) >= 0])
    return normalized_step_ids
