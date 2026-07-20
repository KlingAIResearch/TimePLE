"""Duration-adaptive span-only Canonical Interval Square Codec.

- no pointness input
- no point-location head
- no point/span router
- only a 2D ``(u, v)`` span density on the canonical interval square
- tau=0 duration coordinate by default
- duration-adaptive decoder residual enabled by default
- absolute-duration-aware boundary loss enabled by default
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import interval_giou_loss, interval_iou
from .transforms import CanonicalSpanIntervalTransform


def _deep_merge_dict(base: Dict, override: Optional[Dict]) -> Dict:
    result = dict(base)
    if override is None:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "gelu",
    dropout: float = 0.0,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_make_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class TimePLEDecoderOutput:
    span_logits: torch.Tensor
    span_probs: torch.Tensor
    span_u: torch.Tensor
    span_v: torch.Tensor
    pred_start_rel: torch.Tensor
    pred_end_rel: torch.Tensor
    hard_start_rel: torch.Tensor
    hard_end_rel: torch.Tensor
    residual_delta_uv: Optional[torch.Tensor] = None
    residual_gate: Optional[torch.Tensor] = None


class SpanSquareEncoder(nn.Module):
    def __init__(
        self,
        num_u_bins: int,
        num_v_bins: int,
        token_dim: int,
        hidden_dims: Sequence[int],
        span_sigma_u: float,
        span_sigma_v: float,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_u_bins = int(num_u_bins)
        self.num_v_bins = int(num_v_bins)
        self.span_sigma_u = float(span_sigma_u)
        self.span_sigma_v = float(span_sigma_v)
        input_dim = self.num_u_bins * self.num_v_bins
        self.mlp = _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=token_dim,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.register_buffer("u_centers", torch.linspace(0.0, 1.0, self.num_u_bins))
        self.register_buffer("v_centers", torch.linspace(0.0, 1.0, self.num_v_bins))

    def _gaussian_target(self, centers: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
        sigma = max(float(sigma), 1e-4)
        target = target.unsqueeze(-1)
        dist_sq = (centers.unsqueeze(0) - target) ** 2
        weights = torch.exp(-dist_sq / (2.0 * sigma * sigma))
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def build_targets(self, u: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        span_u_target = self._gaussian_target(self.u_centers.to(u), u, self.span_sigma_u)
        span_v_target = self._gaussian_target(self.v_centers.to(v), v, self.span_sigma_v)
        span_target = span_u_target.unsqueeze(-1) * span_v_target.unsqueeze(-2)
        span_target = span_target / span_target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return {
            "span_target": span_target,
            "packed_target": span_target.reshape(u.shape[0], -1),
        }

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        targets = self.build_targets(u=u, v=v)
        packed_target = targets["packed_target"]
        param_dtype = next(self.mlp.parameters()).dtype
        if packed_target.dtype != param_dtype:
            packed_target = packed_target.to(param_dtype)
        return self.mlp(packed_target)


class SpanSquareDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_u_bins: int,
        num_v_bins: int,
        trunk_hidden_dims: Sequence[int],
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        duration_adaptive_residual: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.num_u_bins = int(num_u_bins)
        self.num_v_bins = int(num_v_bins)
        if trunk_hidden_dims:
            self.trunk = _build_mlp(
                input_dim=token_dim,
                hidden_dims=trunk_hidden_dims[:-1],
                output_dim=trunk_hidden_dims[-1],
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            head_dim = trunk_hidden_dims[-1]
        else:
            self.trunk = nn.Identity()
            head_dim = token_dim

        self.span_head = nn.Linear(head_dim, self.num_u_bins * self.num_v_bins)
        self.register_buffer("u_centers", torch.linspace(0.0, 1.0, self.num_u_bins))
        self.register_buffer("v_centers", torch.linspace(0.0, 1.0, self.num_v_bins))

        residual_cfg = duration_adaptive_residual or {}
        self.duration_adaptive_residual_enabled = bool(residual_cfg.get("enabled", False))
        self.residual_scale = float(residual_cfg.get("scale", 0.0))
        self.residual_weight_mode = str(residual_cfg.get("weight_mode", "relative_sqrt_inv"))
        self.residual_weight_eps = float(residual_cfg.get("weight_eps", 1e-4))
        self.residual_weight_max = float(residual_cfg.get("weight_max", 10.0))
        if self.duration_adaptive_residual_enabled:
            self.residual_head = _build_mlp(
                input_dim=head_dim + 3,
                hidden_dims=residual_cfg.get("hidden_dims", []),
                output_dim=2,
                activation=residual_cfg.get("activation", activation),
                dropout=float(residual_cfg.get("dropout", dropout)),
                use_layer_norm=bool(residual_cfg.get("use_layer_norm", use_layer_norm)),
            )
        else:
            self.residual_head = None

    def _relative_duration_weight(self, duration_rel: torch.Tensor) -> torch.Tensor:
        duration_rel = duration_rel.clamp(0.0, 1.0)
        if self.residual_weight_mode in {"none", "uniform", "constant"}:
            weight = torch.ones_like(duration_rel)
        elif self.residual_weight_mode == "relative_sqrt_inv":
            weight = torch.rsqrt(duration_rel.clamp_min(self.residual_weight_eps))
        elif self.residual_weight_mode == "relative_inv":
            weight = 1.0 / duration_rel.clamp_min(self.residual_weight_eps)
        else:
            raise ValueError(f"Unsupported duration adaptive residual weight mode: {self.residual_weight_mode!r}")
        if self.residual_weight_max > 0.0:
            weight = weight.clamp(max=self.residual_weight_max)
        return weight

    def _correct_uv_with_residual(
        self,
        *,
        features: torch.Tensor,
        span_u: torch.Tensor,
        span_v: torch.Tensor,
        transform: CanonicalSpanIntervalTransform,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual_delta_uv = None
        residual_gate = None
        if not (self.duration_adaptive_residual_enabled and self.residual_head is not None and self.residual_scale > 0.0):
            return span_u.clamp(0.0, 1.0), span_v.clamp(0.0, 1.0), residual_delta_uv, residual_gate

        if features.dim() != 2:
            raise ValueError(f"features must have shape [B, H], got {tuple(features.shape)}.")
        if span_u.shape != span_v.shape:
            raise ValueError(f"span_u/span_v shape mismatch: {tuple(span_u.shape)} vs {tuple(span_v.shape)}.")
        if span_u.dim() not in {1, 2}:
            raise ValueError(f"span_u/span_v must have shape [B] or [B, C], got {tuple(span_u.shape)}.")
        if features.size(0) != span_u.size(0):
            raise ValueError(f"features/uv batch mismatch: {features.size(0)} vs {span_u.size(0)}.")

        original_shape = span_u.shape
        if span_u.dim() == 1:
            features_flat = features
            span_u_flat = span_u
            span_v_flat = span_v
        else:
            cell_count = span_u.size(1)
            features_flat = features.unsqueeze(1).expand(-1, cell_count, -1).reshape(-1, features.size(-1))
            span_u_flat = span_u.reshape(-1)
            span_v_flat = span_v.reshape(-1)

        _, _, duration_rel = transform.square_to_interval(span_u_flat, span_v_flat)
        raw_weight = self._relative_duration_weight(duration_rel)
        if self.residual_weight_max > 0.0:
            residual_gate_flat = (raw_weight / self.residual_weight_max).clamp(0.0, 1.0)
        else:
            residual_gate_flat = raw_weight / raw_weight.detach().amax().clamp_min(1e-8)
        residual_input = torch.cat(
            [
                features_flat,
                span_u_flat.unsqueeze(-1).to(features_flat),
                span_v_flat.unsqueeze(-1).to(features_flat),
                residual_gate_flat.unsqueeze(-1).to(features_flat),
            ],
            dim=-1,
        )
        residual_delta_uv_flat = self.residual_scale * residual_gate_flat.unsqueeze(-1) * torch.tanh(
            self.residual_head(residual_input)
        )
        corrected_u = (span_u_flat + residual_delta_uv_flat[:, 0].to(span_u_flat)).clamp(0.0, 1.0)
        corrected_v = (span_v_flat + residual_delta_uv_flat[:, 1].to(span_v_flat)).clamp(0.0, 1.0)
        residual_delta_uv = residual_delta_uv_flat.reshape(*original_shape, 2)
        residual_gate = residual_gate_flat.reshape(original_shape)
        return corrected_u.reshape(original_shape), corrected_v.reshape(original_shape), residual_delta_uv, residual_gate

    def decode_uv_with_features(
        self,
        *,
        features: torch.Tensor,
        uv: torch.Tensor,
        transform: CanonicalSpanIntervalTransform,
        span_logits: Optional[torch.Tensor] = None,
        span_probs: Optional[torch.Tensor] = None,
    ) -> TimePLEDecoderOutput:
        if uv.dim() not in {2, 3} or uv.size(-1) != 2:
            raise ValueError(f"uv must have shape [B, 2] or [B, C, 2], got {tuple(uv.shape)}.")
        if features.dim() != 2 or features.size(0) != uv.size(0):
            raise ValueError(f"features must have shape [B, H] aligned with uv, got {tuple(features.shape)}.")

        decoder_dtype = next(self.span_head.parameters()).dtype
        if features.dtype != decoder_dtype:
            features = features.to(decoder_dtype)
        span_u = uv[..., 0].to(features)
        span_v = uv[..., 1].to(features)
        span_u, span_v, residual_delta_uv, residual_gate = self._correct_uv_with_residual(
            features=features,
            span_u=span_u,
            span_v=span_v,
            transform=transform,
        )
        span_start_rel, span_end_rel, _ = transform.square_to_interval(span_u, span_v)

        if span_logits is None:
            span_logits = features.new_empty(*span_u.shape, 0)
        if span_probs is None:
            span_probs = features.new_empty(*span_u.shape, 0)

        return TimePLEDecoderOutput(
            span_logits=span_logits,
            span_probs=span_probs,
            span_u=span_u,
            span_v=span_v,
            pred_start_rel=span_start_rel,
            pred_end_rel=span_end_rel,
            hard_start_rel=span_start_rel,
            hard_end_rel=span_end_rel,
            residual_delta_uv=residual_delta_uv,
            residual_gate=residual_gate,
        )

    def forward(self, token: torch.Tensor, transform: CanonicalSpanIntervalTransform) -> TimePLEDecoderOutput:
        trunk_dtype = next(self.span_head.parameters()).dtype
        if token.dtype != trunk_dtype:
            token = token.to(trunk_dtype)

        features = self.trunk(token)
        span_logits = self.span_head(features).view(-1, self.num_u_bins, self.num_v_bins)
        span_probs = F.softmax(span_logits.view(span_logits.shape[0], -1), dim=-1).view_as(span_logits)

        span_u_marginal = span_probs.sum(dim=-1)
        span_v_marginal = span_probs.sum(dim=-2)
        span_u = (span_u_marginal * self.u_centers.to(span_u_marginal)).sum(dim=-1)
        span_v = (span_v_marginal * self.v_centers.to(span_v_marginal)).sum(dim=-1)
        return self.decode_uv_with_features(
            features=features,
            uv=torch.stack([span_u, span_v], dim=-1),
            transform=transform,
            span_logits=span_logits,
            span_probs=span_probs,
        )


class TimePLECodec(nn.Module):
    """
    Duration-adaptive span-only Canonical Interval Square codec.

    External inputs and outputs are seconds. Internally the codec works on
    per-video relative intervals and canonical square coordinates.
    """

    default_config: Dict = {
        "token_dim": 4096,
        "transform": {
            "tau": 0.0,
            "eps": 1e-6,
        },
        "grid": {
            "num_u_bins": 128,
            "num_v_bins": 128,
            "span_sigma_u": 0.015,
            "span_sigma_v": 0.015,
        },
        "encoder": {
            "hidden_dims": [512, 2048],
            "activation": "gelu",
            "dropout": 0.0,
            "use_layer_norm": True,
        },
        "decoder": {
            "trunk_hidden_dims": [2048, 1024],
            "activation": "gelu",
            "dropout": 0.0,
            "use_layer_norm": True,
            "duration_adaptive_residual": {
                "enabled": True,
                "hidden_dims": [256],
                "scale": 0.02,
                "activation": "gelu",
                "dropout": 0.0,
                "use_layer_norm": True,
                "weight_mode": "relative_sqrt_inv",
                "weight_eps": 1e-4,
                "weight_max": 10.0,
            },
        },
        "loss": {
            "lambda_dfl": 1.0,
            "lambda_iou": 1.0,
            "lambda_boundary": 0.5,
            "boundary_weight_mode": "absolute_bucket",
            "boundary_weight_eps": 1e-4,
            "boundary_weight_max": 10.0,
            "boundary_weight_normalize": True,
            "absolute_short_threshold_sec": 10.0,
            "absolute_medium_threshold_sec": 30.0,
            "absolute_short_weight": 3.0,
            "absolute_medium_weight": 1.5,
            "absolute_long_weight": 1.0,
        },
    }

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__()
        self.config = _deep_merge_dict(self.default_config, config)
        transform_cfg = self.config["transform"]
        grid_cfg = self.config["grid"]
        encoder_cfg = self.config["encoder"]
        decoder_cfg = self.config["decoder"]
        self.loss_cfg = _deep_merge_dict(self.default_config["loss"], self.config.get("loss"))

        self.token_dim = int(self.config["token_dim"])
        self.transform = CanonicalSpanIntervalTransform(
            tau=transform_cfg.get("tau", 0.0),
            eps=transform_cfg.get("eps", 1e-6),
            point_threshold=transform_cfg.get("point_threshold", 1e-4),
        )
        self.encoder = SpanSquareEncoder(
            num_u_bins=grid_cfg["num_u_bins"],
            num_v_bins=grid_cfg["num_v_bins"],
            token_dim=self.token_dim,
            hidden_dims=encoder_cfg.get("hidden_dims", []),
            span_sigma_u=grid_cfg["span_sigma_u"],
            span_sigma_v=grid_cfg["span_sigma_v"],
            activation=encoder_cfg.get("activation", "gelu"),
            dropout=encoder_cfg.get("dropout", 0.0),
            use_layer_norm=encoder_cfg.get("use_layer_norm", True),
        )
        self.decoder = SpanSquareDecoder(
            token_dim=self.token_dim,
            num_u_bins=grid_cfg["num_u_bins"],
            num_v_bins=grid_cfg["num_v_bins"],
            trunk_hidden_dims=decoder_cfg.get("trunk_hidden_dims", []),
            activation=decoder_cfg.get("activation", "gelu"),
            dropout=decoder_cfg.get("dropout", 0.0),
            use_layer_norm=decoder_cfg.get("use_layer_norm", True),
            duration_adaptive_residual=decoder_cfg.get("duration_adaptive_residual"),
        )

    def get_output_dim(self) -> int:
        return self.token_dim

    def _get_param_dtype(self) -> torch.dtype:
        for param in self.parameters():
            return param.dtype
        return torch.float32

    def _prepare_durations(
        self,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        duration = torch.as_tensor(video_duration_sec, device=reference.device, dtype=reference.dtype)
        if duration.ndim == 0:
            duration = duration.expand_as(reference)
        elif duration.ndim == 1 and reference.ndim > 1 and duration.numel() == reference.shape[0]:
            duration = duration.view(-1, *([1] * (reference.ndim - 1))).expand_as(reference)
        elif duration.shape != reference.shape:
            duration = duration.reshape(-1)
            if duration.numel() == 1:
                duration = duration.expand_as(reference)
            elif duration.numel() != reference.numel():
                raise ValueError(
                    "Number of durations must match the number of intervals: "
                    f"{duration.numel()} vs {reference.numel()}."
                )
            else:
                duration = duration.reshape_as(reference)
        return duration.clamp_min(self.transform.eps)

    def _to_relative(
        self,
        start_sec: torch.Tensor,
        end_sec: torch.Tensor,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param_dtype = self._get_param_dtype()
        start_sec = start_sec.to(dtype=param_dtype)
        end_sec = end_sec.to(dtype=param_dtype)
        duration = self._prepare_durations(video_duration_sec, start_sec).to(dtype=param_dtype)
        start_rel, end_rel = self.transform.normalize_seconds(start_sec, end_sec, duration)
        return start_rel, end_rel, duration

    def encode_relative(self, start_rel: torch.Tensor, end_rel: torch.Tensor) -> torch.Tensor:
        u, v, _ = self.transform.interval_to_square(start_rel, end_rel)
        return self.encoder(u=u, v=v)

    def encode(
        self,
        start_sec: torch.Tensor,
        end_sec: torch.Tensor,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
    ) -> torch.Tensor:
        start_rel, end_rel, _ = self._to_relative(start_sec, end_sec, video_duration_sec)
        return self.encode_relative(start_rel, end_rel)

    def decode_relative(self, token: torch.Tensor) -> TimePLEDecoderOutput:
        return self.decoder(token=token, transform=self.transform)

    def decode_uv_with_features(
        self,
        *,
        features: torch.Tensor,
        uv: torch.Tensor,
    ) -> TimePLEDecoderOutput:
        return self.decoder.decode_uv_with_features(
            features=features,
            uv=uv,
            transform=self.transform,
        )

    def decode_uv_with_features_seconds(
        self,
        *,
        features: torch.Tensor,
        uv: torch.Tensor,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, TimePLEDecoderOutput]:
        decoded = self.decode_uv_with_features(features=features, uv=uv)
        start_sec, end_sec, _ = self._decode_seconds_from_decoded(
            decoded,
            video_duration_sec=video_duration_sec,
            hard=hard,
        )
        return start_sec, end_sec, decoded

    def decode(
        self,
        token: torch.Tensor,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        hard: bool = True,
        return_details: bool = False,
    ):
        decoded = self.decode_relative(token)
        start_sec, end_sec, _ = self._decode_seconds_from_decoded(
            decoded,
            video_duration_sec=video_duration_sec,
            hard=hard,
        )
        if return_details:
            return start_sec, end_sec, decoded
        return start_sec, end_sec

    def _decode_seconds_from_decoded(
        self,
        decoded: TimePLEDecoderOutput,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hard:
            start_rel = decoded.hard_start_rel
            end_rel = decoded.hard_end_rel
        else:
            start_rel = decoded.pred_start_rel
            end_rel = decoded.pred_end_rel

        durations = self._prepare_durations(video_duration_sec, start_rel).to(start_rel.dtype)
        start_sec, end_sec = self.transform.denormalize_seconds(start_rel, end_rel, durations)
        return start_sec, end_sec, durations

    def reencode_from_decoded(
        self,
        decoded: TimePLEDecoderOutput,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        hard: bool = False,
    ) -> torch.Tensor:
        pred_start_sec, pred_end_sec, _ = self._decode_seconds_from_decoded(
            decoded,
            video_duration_sec=video_duration_sec,
            hard=hard,
        )
        return self.encode(pred_start_sec, pred_end_sec, video_duration_sec=video_duration_sec)

    def reencode_from_token(
        self,
        token: torch.Tensor,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        hard: bool = False,
    ) -> torch.Tensor:
        decoded = self.decode_relative(token)
        return self.reencode_from_decoded(decoded, video_duration_sec=video_duration_sec, hard=hard)

    def compute_loss_from_decoded(
        self,
        decoded: TimePLEDecoderOutput,
        target_start_sec: torch.Tensor,
        target_end_sec: torch.Tensor,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}.")

        start_rel, end_rel, durations = self._to_relative(target_start_sec, target_end_sec, video_duration_sec)
        u, v, _ = self.transform.interval_to_square(start_rel, end_rel)
        targets = self.encoder.build_targets(u=u, v=v)

        span_logits = decoded.span_logits.reshape(decoded.span_logits.shape[0], -1)
        span_target = targets["span_target"].reshape(decoded.span_logits.shape[0], -1).to(span_logits.dtype)
        dfl_loss_per_token = -(span_target * F.log_softmax(span_logits, dim=-1)).sum(dim=-1)
        interval_l1_per_token = (
            (decoded.pred_start_rel - start_rel).abs()
            + (decoded.pred_end_rel - end_rel).abs()
        ) * 0.5
        boundary_weight_per_token = self._boundary_weight(
            start_rel=start_rel,
            end_rel=end_rel,
            target_start_sec=target_start_sec.to(start_rel.dtype),
            target_end_sec=target_end_sec.to(start_rel.dtype),
        )
        boundary_loss_per_token = boundary_weight_per_token * interval_l1_per_token
        span_giou_per_token = interval_giou_loss(
            decoded.pred_start_rel,
            decoded.pred_end_rel,
            start_rel,
            end_rel,
        )
        span_iou_per_token = interval_iou(
            decoded.hard_start_rel,
            decoded.hard_end_rel,
            start_rel,
            end_rel,
        )
        iou_loss_per_token = 1.0 - span_iou_per_token

        pred_start_sec, pred_end_sec, _ = self._decode_seconds_from_decoded(
            decoded,
            video_duration_sec=durations,
            hard=True,
        )
        mae_start_per_token = (pred_start_sec - target_start_sec.to(pred_start_sec.dtype)).abs()
        mae_end_per_token = (pred_end_sec - target_end_sec.to(pred_end_sec.dtype)).abs()

        if reduction == "none":
            total_loss_per_token = (
                float(self.loss_cfg.get("lambda_dfl", self.loss_cfg.get("lambda_span", 1.0))) * dfl_loss_per_token
                + float(self.loss_cfg.get("lambda_iou", 1.0)) * iou_loss_per_token
                + float(self.loss_cfg.get("lambda_boundary", 0.0)) * boundary_loss_per_token
            )
            return {
                "total_loss": total_loss_per_token,
                "dfl_loss": dfl_loss_per_token,
                "iou_loss": iou_loss_per_token,
                "span_loss": dfl_loss_per_token,
                "interval_l1": interval_l1_per_token,
                "boundary_loss": boundary_loss_per_token,
                "boundary_weight": boundary_weight_per_token,
                "span_giou_loss": span_giou_per_token,
                "span_iou": span_iou_per_token,
                "mae_start": mae_start_per_token,
                "mae_end": mae_end_per_token,
                "mae_total": 0.5 * (mae_start_per_token + mae_end_per_token),
                "pred_start_sec": pred_start_sec.detach(),
                "pred_end_sec": pred_end_sec.detach(),
            }

        dfl_loss = dfl_loss_per_token.mean()
        iou_loss = iou_loss_per_token.mean()
        interval_l1 = interval_l1_per_token.mean()
        boundary_loss = boundary_loss_per_token.mean()
        boundary_weight = boundary_weight_per_token.mean()
        span_giou = span_giou_per_token.mean()
        span_iou = span_iou_per_token.mean()

        total_loss = (
            float(self.loss_cfg.get("lambda_dfl", self.loss_cfg.get("lambda_span", 1.0))) * dfl_loss
            + float(self.loss_cfg.get("lambda_iou", 1.0)) * iou_loss
            + float(self.loss_cfg.get("lambda_boundary", 0.0)) * boundary_loss
        )

        mae_start = mae_start_per_token.mean()
        mae_end = mae_end_per_token.mean()

        return {
            "total_loss": total_loss,
            "dfl_loss": dfl_loss,
            "iou_loss": iou_loss,
            "span_loss": dfl_loss,
            "interval_l1": interval_l1,
            "boundary_loss": boundary_loss,
            "boundary_weight": boundary_weight,
            "span_giou_loss": span_giou,
            "span_iou": span_iou,
            "mae_start": mae_start,
            "mae_end": mae_end,
            "mae_total": 0.5 * (mae_start + mae_end),
            "pred_start_sec": pred_start_sec.detach(),
            "pred_end_sec": pred_end_sec.detach(),
        }

    def _boundary_weight(
        self,
        *,
        start_rel: torch.Tensor,
        end_rel: torch.Tensor,
        target_start_sec: torch.Tensor,
        target_end_sec: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(self.loss_cfg.get("boundary_weight_mode", "none"))
        if mode in {"none", "uniform", "constant"}:
            weight = torch.ones_like(start_rel)
        elif mode == "relative_sqrt_inv":
            duration_rel = (end_rel - start_rel).abs().clamp_min(float(self.loss_cfg.get("boundary_weight_eps", 1e-4)))
            weight = torch.rsqrt(duration_rel)
        elif mode == "relative_inv":
            duration_rel = (end_rel - start_rel).abs().clamp_min(float(self.loss_cfg.get("boundary_weight_eps", 1e-4)))
            weight = 1.0 / duration_rel
        elif mode == "absolute_bucket":
            span_duration_sec = (target_end_sec - target_start_sec).abs().to(start_rel).clamp_min(1e-8)
            short_threshold = float(self.loss_cfg.get("absolute_short_threshold_sec", 10.0))
            medium_threshold = float(self.loss_cfg.get("absolute_medium_threshold_sec", 30.0))
            short_weight = float(self.loss_cfg.get("absolute_short_weight", 3.0))
            medium_weight = float(self.loss_cfg.get("absolute_medium_weight", 1.5))
            long_weight = float(self.loss_cfg.get("absolute_long_weight", 1.0))
            weight = torch.where(
                span_duration_sec <= short_threshold,
                torch.full_like(span_duration_sec, short_weight),
                torch.where(
                    span_duration_sec <= medium_threshold,
                    torch.full_like(span_duration_sec, medium_weight),
                    torch.full_like(span_duration_sec, long_weight),
                ),
            )
        else:
            raise ValueError(f"Unsupported boundary_weight_mode: {mode!r}")

        max_weight = float(self.loss_cfg.get("boundary_weight_max", 0.0))
        if max_weight > 0.0:
            weight = weight.clamp(max=max_weight)
        if bool(self.loss_cfg.get("boundary_weight_normalize", True)):
            weight = weight / weight.detach().mean().clamp_min(1e-8)
        return weight

    def compute_loss(
        self,
        time_embeddings: torch.Tensor,
        target_start_sec: torch.Tensor,
        target_end_sec: torch.Tensor,
        video_duration_sec: torch.Tensor | Sequence[float] | float,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        decoded = self.decode_relative(time_embeddings)
        return self.compute_loss_from_decoded(
            decoded,
            target_start_sec=target_start_sec,
            target_end_sec=target_end_sec,
            video_duration_sec=video_duration_sec,
            reduction=reduction,
        )

__all__ = [
    "CanonicalSpanIntervalCodec",
    "TimePLEDecoderOutput",
    "TimePLECodec",
    "SpanSquareDecoder",
    "SpanSquareEncoder",
]


# Compatibility alias for geometry-pretrain utilities that use the original
# span-only codec interface name.
CanonicalSpanIntervalCodec = TimePLECodec
