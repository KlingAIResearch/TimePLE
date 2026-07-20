"""
MLP-only TimePLE interface adapter modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


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
    name = str(name).lower()
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


class _RMSNormNoAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms


class _OneLayerMLPProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
        dropout: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.norm = _RMSNormNoAffine(self.input_dim, eps=eps)
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected projector input dim {self.input_dim}, got {x.shape[-1]}.")
        return self.out_proj(self.dropout(self.act(self.in_proj(self.norm(x)))))


def _resolve_dim(cfg: Dict[str, object], key: str, default: int) -> int:
    value = cfg.get(key)
    if value is None:
        return int(default)
    dim = int(value)
    if dim <= 0:
        raise ValueError(f"{key} must be positive, got {dim}.")
    return dim


def _ratio_stats(delta: torch.Tensor, reference: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    reference_norm_sq = reference.float().pow(2).sum(dim=-1) + eps
    delta_norm_sq = delta.float().pow(2).sum(dim=-1)
    residual_penalty_per_token = delta_norm_sq / reference_norm_sq
    residual_ratio_per_token = torch.sqrt(residual_penalty_per_token)
    return {
        "residual_penalty_per_token": residual_penalty_per_token,
        "residual_ratio_per_token": residual_ratio_per_token,
        "residual_penalty": residual_penalty_per_token.mean(),
        "residual_ratio": residual_ratio_per_token.mean(),
    }


@dataclass
class TimePLEInputAdapterOutput:
    adapted: torch.Tensor
    base: torch.Tensor
    delta: torch.Tensor
    residual_ratio: torch.Tensor
    residual_penalty: torch.Tensor
    lambda_value: torch.Tensor
    anchor_alpha: torch.Tensor
    residual_penalty_per_token: Optional[torch.Tensor] = None
    residual_ratio_per_token: Optional[torch.Tensor] = None


@dataclass
class TimePLEOutputAdapterOutput:
    adapted: torch.Tensor
    base: torch.Tensor
    delta: torch.Tensor
    residual_ratio: torch.Tensor
    residual_penalty: torch.Tensor
    lambda_value: torch.Tensor
    bridge_norm: torch.Tensor
    residual_penalty_per_token: Optional[torch.Tensor] = None
    residual_ratio_per_token: Optional[torch.Tensor] = None


class TimePLEMLPProjectorInputAdapter(nn.Module):
    default_config: Dict[str, object] = {
        "enabled": True,
        "input_dim": None,
        "hidden_dim": None,
        "output_dim": None,
        "activation": "gelu",
        "dropout": 0.0,
        "anchor_alpha_init": 0.01,
        "anchor_alpha_max": 0.02,
        "eps": 1e-6,
    }

    def __init__(
        self,
        model_hidden_dim: int,
        codec_dim: Optional[int],
        config: Optional[Dict[str, object]],
    ) -> None:
        super().__init__()
        cfg = _deep_merge_dict(self.default_config, config)
        self.config = cfg
        self.model_hidden_dim = int(model_hidden_dim)
        self.codec_dim = int(codec_dim if codec_dim is not None else model_hidden_dim)
        self.input_dim = _resolve_dim(cfg, "input_dim", self.codec_dim)
        self.output_dim = _resolve_dim(cfg, "output_dim", self.model_hidden_dim)
        hidden_default = self.output_dim if self.input_dim != self.output_dim else self.input_dim
        self.hidden_dim = _resolve_dim(cfg, "hidden_dim", hidden_default)
        self.eps = float(cfg["eps"])
        self.anchor_alpha_max = float(cfg["anchor_alpha_max"])
        anchor_alpha_init = min(float(cfg["anchor_alpha_init"]), self.anchor_alpha_max)
        self.register_buffer("anchor_alpha", torch.tensor(anchor_alpha_init, dtype=torch.float32))
        self.projector = _OneLayerMLPProjector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation=str(cfg["activation"]),
            dropout=float(cfg["dropout"]),
            eps=self.eps,
        )

    def forward(
        self,
        codec_latent: torch.Tensor,
        anchor_embedding: Optional[torch.Tensor] = None,
    ) -> TimePLEInputAdapterOutput:
        x_base = self.projector(codec_latent)
        x_adapt = x_base
        if anchor_embedding is not None:
            if anchor_embedding.shape[-1] != self.output_dim:
                raise ValueError(f"Expected anchor embedding dim {self.output_dim}, got {anchor_embedding.shape[-1]}.")
            x_adapt = x_adapt + self.anchor_alpha.to(codec_latent.dtype) * anchor_embedding

        reference = anchor_embedding if anchor_embedding is not None else x_base.detach()
        x_delta = x_adapt - reference
        stats = _ratio_stats(x_delta, reference, self.eps)

        return TimePLEInputAdapterOutput(
            adapted=x_adapt,
            base=x_base,
            delta=x_delta,
            residual_ratio=stats["residual_ratio"],
            residual_penalty=stats["residual_penalty"],
            lambda_value=torch.zeros((), device=codec_latent.device, dtype=codec_latent.dtype),
            anchor_alpha=self.anchor_alpha.detach().to(codec_latent.dtype),
            residual_penalty_per_token=stats["residual_penalty_per_token"],
            residual_ratio_per_token=stats["residual_ratio_per_token"],
        )


class TimePLEMLPProjectorOutputAdapter(nn.Module):
    default_config: Dict[str, object] = {
        "enabled": True,
        "input_dim": None,
        "hidden_dim": None,
        "output_dim": None,
        "activation": "gelu",
        "dropout": 0.0,
        "eps": 1e-6,
    }

    def __init__(
        self,
        model_hidden_dim: int,
        codec_dim: Optional[int],
        config: Optional[Dict[str, object]],
    ) -> None:
        super().__init__()
        cfg = _deep_merge_dict(self.default_config, config)
        self.config = cfg
        self.model_hidden_dim = int(model_hidden_dim)
        self.codec_dim = int(codec_dim if codec_dim is not None else model_hidden_dim)
        self.input_dim = _resolve_dim(cfg, "input_dim", self.model_hidden_dim)
        self.output_dim = _resolve_dim(cfg, "output_dim", self.codec_dim)
        hidden_default = self.input_dim if self.input_dim != self.output_dim else self.output_dim
        self.hidden_dim = _resolve_dim(cfg, "hidden_dim", hidden_default)
        self.eps = float(cfg["eps"])
        self.projector = _OneLayerMLPProjector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation=str(cfg["activation"]),
            dropout=float(cfg["dropout"]),
            eps=self.eps,
        )

    def forward(self, llm_hidden: torch.Tensor, compute_diagnostics: bool = True) -> TimePLEOutputAdapterOutput:
        z_hat = self.projector(llm_hidden)
        if self.input_dim == self.output_dim:
            h_delta = z_hat - llm_hidden
        else:
            h_delta = torch.zeros_like(z_hat)

        if not compute_diagnostics:
            zero = torch.zeros((), device=llm_hidden.device, dtype=llm_hidden.dtype)
            return TimePLEOutputAdapterOutput(
                adapted=z_hat,
                base=z_hat,
                delta=h_delta,
                residual_ratio=zero,
                residual_penalty=zero,
                lambda_value=zero,
                bridge_norm=zero,
                residual_penalty_per_token=None,
                residual_ratio_per_token=None,
            )

        stats = _ratio_stats(h_delta, z_hat.detach(), self.eps)
        return TimePLEOutputAdapterOutput(
            adapted=z_hat,
            base=z_hat,
            delta=h_delta,
            residual_ratio=stats["residual_ratio"],
            residual_penalty=stats["residual_penalty"],
            lambda_value=torch.zeros((), device=llm_hidden.device, dtype=llm_hidden.dtype),
            bridge_norm=torch.zeros((), device=llm_hidden.device, dtype=llm_hidden.dtype),
            residual_penalty_per_token=stats["residual_penalty_per_token"],
            residual_ratio_per_token=stats["residual_ratio_per_token"],
        )


class TimePLEInterfaceAdapter(nn.Module):
    default_config: Dict[str, object] = {
        "adapter_type": "mlp_projector",
        "regularization": {
            "base_decode_aux_weight": 0.0,
            "input_residual_norm_weight": 0.0,
            "output_residual_norm_weight": 0.0,
            "output_bridge_weight": 0.0,
        },
        "projector": {
            "input": TimePLEMLPProjectorInputAdapter.default_config,
            "output": TimePLEMLPProjectorOutputAdapter.default_config,
            "regularization": {},
        },
    }

    def __init__(
        self,
        model_hidden_dim: int,
        config: Optional[Dict[str, object]] = None,
        *,
        codec_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_hidden_dim = int(model_hidden_dim)
        self.dim = self.model_hidden_dim
        self.codec_dim = int(codec_dim if codec_dim is not None else model_hidden_dim)
        self.config = _deep_merge_dict(self.default_config, config)
        self.adapter_type = self._resolve_adapter_type(self.config.get("adapter_type", "mlp_projector"))

        projector_cfg = dict(self.config.get("projector", {}) or {})
        projector_regularization = dict(projector_cfg.get("regularization", {}) or {})
        self.input_adapter = TimePLEMLPProjectorInputAdapter(
            self.model_hidden_dim,
            self.codec_dim,
            projector_cfg.get("input"),
        )
        self.output_adapter = TimePLEMLPProjectorOutputAdapter(
            self.model_hidden_dim,
            self.codec_dim,
            projector_cfg.get("output"),
        )
        self.regularization = _deep_merge_dict(
            dict(self.config.get("regularization", {}) or {}),
            projector_regularization,
        )

    @staticmethod
    def _resolve_adapter_type(value: object) -> str:
        if value is None:
            return "mlp_projector"
        name = str(value).lower().strip()
        if not name or name in {"mlp_projector", "mlp", "one_layer_mlp_projector"}:
            return "mlp_projector"
        raise ValueError(f"Unsupported TimePLE interface adapter type: {value!r}. Only mlp_projector is supported.")

    def forward_input(
        self,
        codec_latent: torch.Tensor,
        anchor_embedding: Optional[torch.Tensor] = None,
    ) -> TimePLEInputAdapterOutput:
        return self.input_adapter(codec_latent, anchor_embedding=anchor_embedding)

    def forward_output(self, llm_hidden: torch.Tensor, compute_diagnostics: bool = True) -> TimePLEOutputAdapterOutput:
        return self.output_adapter(llm_hidden, compute_diagnostics=compute_diagnostics)

    def load_stats(self, stats_dict: Dict[str, torch.Tensor]) -> None:
        if stats_dict:
            raise RuntimeError("TimePLE mlp_projector does not use legacy stats buffers.")

    def export_debug_state(self) -> Dict[str, torch.Tensor]:
        return {
            "alpha_ts": self.input_adapter.anchor_alpha.detach().clone(),
        }

    def get_regularization_weights(self) -> Dict[str, float]:
        return {
            "base_decode_aux_weight": float(self.regularization.get("base_decode_aux_weight", 0.0)),
            "input_residual_norm_weight": float(self.regularization.get("input_residual_norm_weight", 0.0)),
            "output_residual_norm_weight": float(self.regularization.get("output_residual_norm_weight", 0.0)),
            "output_bridge_weight": 0.0,
        }


TimePLEInputAdapter = TimePLEMLPProjectorInputAdapter
TimePLEOutputAdapter = TimePLEMLPProjectorOutputAdapter
TimePLEProjectorInputAdapter = TimePLEMLPProjectorInputAdapter
TimePLEProjectorOutputAdapter = TimePLEMLPProjectorOutputAdapter


__all__ = [
    "TimePLEInputAdapter",
    "TimePLEInputAdapterOutput",
    "TimePLEInterfaceAdapter",
    "TimePLEOutputAdapter",
    "TimePLEOutputAdapterOutput",
    "TimePLEMLPProjectorInputAdapter",
    "TimePLEMLPProjectorOutputAdapter",
    "TimePLEProjectorInputAdapter",
    "TimePLEProjectorOutputAdapter",
]
