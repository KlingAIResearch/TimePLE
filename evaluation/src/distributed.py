from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def _env_int(keys: list[str], default: int) -> int:
    for key in keys:
        raw = os.environ.get(key)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            LOGGER.warning("Ignoring non-integer env var %s=%r", key, raw)
    return default


@dataclass(slots=True)
class DistributedContext:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    shard_output: bool
    auto_assign_local_gpus: bool
    assigned_cuda_visible_devices: str | None = None

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    def owns_index(self, index: int) -> bool:
        return index % self.world_size == self.rank

    def predictions_path(self, output_dir: Path) -> Path:
        return output_dir / f"predictions.rank{self.rank:04d}.jsonl"

    def failed_path(self, output_dir: Path) -> Path:
        return output_dir / f"failed.rank{self.rank:04d}.jsonl"

    def metrics_path(self, output_dir: Path) -> Path:
        return output_dir / f"metrics.rank{self.rank:04d}.json"


def load_distributed_context(cfg: dict, *, tensor_parallel_size: int) -> DistributedContext:
    enabled = bool(cfg.get("enabled", False))
    auto_from_env = bool(cfg.get("auto_from_env", True))

    if auto_from_env:
        world_size = _env_int(["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"], 1)
        rank = _env_int(["RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"], 0)
        local_rank = _env_int(
            ["LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"],
            0,
        )
        local_world_size = _env_int(
            ["LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE"],
            1,
        )
    else:
        world_size = int(cfg.get("world_size", 1))
        rank = int(cfg.get("rank", 0))
        local_rank = int(cfg.get("local_rank", 0))
        local_world_size = int(cfg.get("local_world_size", 1))

    if world_size > 1:
        enabled = True

    context = DistributedContext(
        enabled=enabled,
        world_size=max(world_size, 1),
        rank=max(rank, 0),
        local_rank=max(local_rank, 0),
        local_world_size=max(local_world_size, 1),
        shard_output=bool(cfg.get("shard_output", True)),
        auto_assign_local_gpus=bool(cfg.get("auto_assign_local_gpus", True)),
    )

    maybe_assign_cuda_visible_devices(context, tensor_parallel_size=tensor_parallel_size)
    return context


def maybe_assign_cuda_visible_devices(
    context: DistributedContext,
    *,
    tensor_parallel_size: int,
) -> None:
    if not context.enabled or not context.auto_assign_local_gpus:
        return
    if context.local_world_size <= 1:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure is runtime only
        LOGGER.warning("Failed to import torch for CUDA assignment: %s", exc)
        return

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        LOGGER.warning("No visible CUDA devices detected while preparing distributed assignment.")
        return

    required_devices = context.local_world_size * max(tensor_parallel_size, 1)
    if device_count < required_devices:
        LOGGER.warning(
            "Visible CUDA devices (%s) are fewer than local_world_size * tensor_parallel_size (%s). "
            "Skip automatic CUDA_VISIBLE_DEVICES assignment.",
            device_count,
            required_devices,
        )
        return

    start = context.local_rank * max(tensor_parallel_size, 1)
    stop = start + max(tensor_parallel_size, 1)
    device_ids = list(range(start, stop))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in device_ids)
    context.assigned_cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    LOGGER.info(
        "Assigned CUDA_VISIBLE_DEVICES=%s for local_rank=%s",
        context.assigned_cuda_visible_devices,
        context.local_rank,
    )
