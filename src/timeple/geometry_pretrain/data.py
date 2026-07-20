"""Balanced synthetic data generation for span-only CIS geometry pretraining."""

from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from timeple.geometry_pretrain.config import SyntheticDatasetConfig
from timeple.models import CanonicalSpanIntervalTransform


class SyntheticSquareGeometryDataset(Dataset):
    """
    Synthetic interval dataset with balanced span coverage over the canonical square.

    Every sample is a span interval. The dataset stratifies over ``(u, v)`` cells
    and uses the same number of samples per cell, so coverage remains explicit
    and easy to audit.
    """

    def __init__(self, config: SyntheticDatasetConfig, transform: CanonicalSpanIntervalTransform) -> None:
        super().__init__()
        self.config = config
        self.config.validate()
        self.transform = transform

        self._span_cell_count = self.config.span_u_bins * self.config.span_v_bins
        self._plan = self._build_balanced_plan()

    def _build_balanced_plan(self) -> List[int]:
        plan: List[int] = []
        span_ids = list(range(self._span_cell_count))
        for _ in range(self.config.span_samples_per_cell):
            plan.extend(span_ids)

        rng = random.Random(self.config.seed)
        rng.shuffle(plan)
        return plan

    def __len__(self) -> int:
        return len(self._plan)

    def _make_generator(self, idx: int, encoded_cell_id: int) -> torch.Generator:
        seed = (
            int(self.config.seed) * 1_000_003
            + int(idx) * 97_409
            + int(encoded_cell_id) * 16_381
        ) % (2**63 - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _sample_unit(generator: torch.Generator) -> float:
        return float(torch.rand((), generator=generator).item())

    def _sample_in_bin(self, idx: int, num_bins: int, generator: torch.Generator) -> float:
        lo = float(idx) / float(num_bins)
        hi = float(idx + 1) / float(num_bins)
        alpha = self._sample_unit(generator)
        return lo + (hi - lo) * alpha

    def _sample_duration_sec(self, generator: torch.Generator) -> float:
        if self.config.duration_distribution == "uniform":
            alpha = self._sample_unit(generator)
            return self.config.duration_min_sec + (
                self.config.duration_max_sec - self.config.duration_min_sec
            ) * alpha
        if self.config.duration_distribution == "log_uniform":
            lo = math.log(self.config.duration_min_sec)
            hi = math.log(self.config.duration_max_sec)
            alpha = self._sample_unit(generator)
            return math.exp(lo + (hi - lo) * alpha)
        raise ValueError(f"Unsupported duration distribution: {self.config.duration_distribution}")

    def planned_coverage(self) -> Dict[str, np.ndarray]:
        span_counts = np.full(
            (self.config.span_v_bins, self.config.span_u_bins),
            self.config.span_samples_per_cell,
            dtype=np.float64,
        )
        return {"span_counts": span_counts}

    def _build_span_sample(
        self,
        encoded_cell_id: int,
        generator: torch.Generator,
    ) -> Dict[str, float | int]:
        u_cell_idx = int(encoded_cell_id % self.config.span_u_bins)
        v_cell_idx = int(encoded_cell_id // self.config.span_u_bins)
        square_u = self._sample_in_bin(u_cell_idx, self.config.span_u_bins, generator)
        square_v = self._sample_in_bin(v_cell_idx, self.config.span_v_bins, generator)
        square_u_t = torch.tensor([square_u], dtype=torch.float32)
        square_v_t = torch.tensor([square_v], dtype=torch.float32)
        start_rel_t, end_rel_t, _ = self.transform.square_to_interval(square_u_t, square_v_t)
        return {
            "square_u": square_u,
            "square_v": square_v,
            "start_rel": float(start_rel_t.item()),
            "end_rel": float(end_rel_t.item()),
            "u_cell_idx": u_cell_idx,
            "v_cell_idx": v_cell_idx,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded_cell_id = self._plan[idx]
        generator = self._make_generator(idx, encoded_cell_id)
        sample = self._build_span_sample(encoded_cell_id, generator)

        duration_sec = self._sample_duration_sec(generator)
        start_sec = float(sample["start_rel"]) * duration_sec
        end_sec = float(sample["end_rel"]) * duration_sec

        return {
            "start_sec": torch.tensor(start_sec, dtype=torch.float32),
            "end_sec": torch.tensor(end_sec, dtype=torch.float32),
            "duration_sec": torch.tensor(duration_sec, dtype=torch.float32),
            "square_u": torch.tensor(float(sample["square_u"]), dtype=torch.float32),
            "square_v": torch.tensor(float(sample["square_v"]), dtype=torch.float32),
            "u_cell_idx": torch.tensor(int(sample["u_cell_idx"]), dtype=torch.long),
            "v_cell_idx": torch.tensor(int(sample["v_cell_idx"]), dtype=torch.long),
            "is_point": torch.tensor(False, dtype=torch.bool),
        }


class StratifiedSquareDurationEvaluationDataset(Dataset):
    """
    Evaluation dataset with explicit coverage over square cells and duration buckets.

    Samples are stratified over ``(u, v, duration_bucket)``. This is intended for
    offline model evaluation so every reported span bucket is backed by broad
    position and video-duration coverage.
    """

    def __init__(
        self,
        config: SyntheticDatasetConfig,
        transform: CanonicalSpanIntervalTransform,
        *,
        duration_bucket_edges_sec: np.ndarray,
        span_samples_per_combo: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.config.validate()
        self.transform = transform
        self.duration_bucket_edges_sec = np.asarray(duration_bucket_edges_sec, dtype=np.float64)
        if self.duration_bucket_edges_sec.ndim != 1 or self.duration_bucket_edges_sec.shape[0] < 2:
            raise ValueError("duration_bucket_edges_sec must be a 1D array with at least 2 values.")
        if not np.all(np.diff(self.duration_bucket_edges_sec) > 0):
            raise ValueError("duration_bucket_edges_sec must be strictly increasing.")
        self.duration_bucket_count = int(self.duration_bucket_edges_sec.shape[0] - 1)
        self.span_samples_per_combo = int(span_samples_per_combo)
        if self.span_samples_per_combo <= 0:
            raise ValueError("span_samples_per_combo must be positive.")
        self.seed = int(self.config.seed if seed is None else seed)

        self._span_cell_count = self.config.span_u_bins * self.config.span_v_bins
        self._span_stratum_count = self._span_cell_count * self.duration_bucket_count
        self._plan = self._build_plan()

    def _build_plan(self) -> List[int]:
        plan: List[int] = []
        span_ids = list(range(self._span_stratum_count))
        for _ in range(self.span_samples_per_combo):
            plan.extend(span_ids)

        rng = random.Random(self.seed)
        rng.shuffle(plan)
        return plan

    def __len__(self) -> int:
        return len(self._plan)

    def _make_generator(self, idx: int, encoded_stratum_id: int) -> torch.Generator:
        seed = (
            int(self.seed) * 1_000_003
            + int(idx) * 97_409
            + int(encoded_stratum_id) * 16_381
        ) % (2**63 - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _sample_unit(generator: torch.Generator) -> float:
        return float(torch.rand((), generator=generator).item())

    def _sample_in_bin(self, idx: int, num_bins: int, generator: torch.Generator) -> float:
        lo = float(idx) / float(num_bins)
        hi = float(idx + 1) / float(num_bins)
        alpha = self._sample_unit(generator)
        return lo + (hi - lo) * alpha

    def _sample_duration_from_bucket(self, bucket_idx: int, generator: torch.Generator) -> float:
        lo = float(self.duration_bucket_edges_sec[bucket_idx])
        hi = float(self.duration_bucket_edges_sec[bucket_idx + 1])
        alpha = self._sample_unit(generator)
        if self.config.duration_distribution == "uniform":
            return lo + (hi - lo) * alpha
        if self.config.duration_distribution == "log_uniform":
            lo_log = math.log(max(lo, 1e-8))
            hi_log = math.log(max(hi, lo * (1.0 + 1e-8)))
            return math.exp(lo_log + (hi_log - lo_log) * alpha)
        raise ValueError(f"Unsupported duration_distribution: {self.config.duration_distribution}")

    def planned_coverage(self) -> Dict[str, np.ndarray]:
        span_counts = np.full(
            (self.duration_bucket_count, self.config.span_v_bins, self.config.span_u_bins),
            self.span_samples_per_combo,
            dtype=np.float64,
        )
        return {
            "span_counts": span_counts,
            "duration_bucket_edges_sec": self.duration_bucket_edges_sec.copy(),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded_stratum_id = self._plan[idx]
        generator = self._make_generator(idx, encoded_stratum_id)

        duration_bucket_idx = int(encoded_stratum_id // self._span_cell_count)
        encoded_cell_id = int(encoded_stratum_id % self._span_cell_count)
        u_cell_idx = int(encoded_cell_id % self.config.span_u_bins)
        v_cell_idx = int(encoded_cell_id // self.config.span_u_bins)
        square_u = self._sample_in_bin(u_cell_idx, self.config.span_u_bins, generator)
        square_v = self._sample_in_bin(v_cell_idx, self.config.span_v_bins, generator)
        square_u_t = torch.tensor([square_u], dtype=torch.float32)
        square_v_t = torch.tensor([square_v], dtype=torch.float32)
        start_rel_t, end_rel_t, _ = self.transform.square_to_interval(square_u_t, square_v_t)

        duration_sec = self._sample_duration_from_bucket(duration_bucket_idx, generator)
        start_sec = float(start_rel_t.item()) * duration_sec
        end_sec = float(end_rel_t.item()) * duration_sec

        return {
            "start_sec": torch.tensor(start_sec, dtype=torch.float32),
            "end_sec": torch.tensor(end_sec, dtype=torch.float32),
            "duration_sec": torch.tensor(duration_sec, dtype=torch.float32),
            "square_u": torch.tensor(square_u, dtype=torch.float32),
            "square_v": torch.tensor(square_v, dtype=torch.float32),
            "u_cell_idx": torch.tensor(u_cell_idx, dtype=torch.long),
            "v_cell_idx": torch.tensor(v_cell_idx, dtype=torch.long),
            "duration_bucket_idx": torch.tensor(duration_bucket_idx, dtype=torch.long),
            "is_point": torch.tensor(False, dtype=torch.bool),
        }
