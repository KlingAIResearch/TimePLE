from __future__ import annotations

import torch
import pytest

from timeple.models import TimePLECodec


def _small_codec() -> TimePLECodec:
    return TimePLECodec(
        {
            "token_dim": 16,
            "grid": {"num_u_bins": 8, "num_v_bins": 8},
            "encoder": {"hidden_dims": [16]},
            "decoder": {
                "trunk_hidden_dims": [16],
                "duration_adaptive_residual": {"enabled": True, "hidden_dims": [8]},
            },
        }
    )


def test_codec_forward_shapes_and_finite_values() -> None:
    codec = _small_codec()
    starts = torch.tensor([0.0, 20.0])
    ends = torch.tensor([10.0, 80.0])
    durations = torch.tensor([100.0, 100.0])

    tokens = codec.encode(starts, ends, durations)
    pred_starts, pred_ends, decoded = codec.decode(tokens, durations, return_details=True)

    assert tokens.shape == (2, 16)
    assert pred_starts.shape == pred_ends.shape == (2,)
    assert decoded.span_probs.shape == (2, 8, 8)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(pred_starts).all()
    assert torch.isfinite(pred_ends).all()


def test_codec_rejects_mismatched_duration_count() -> None:
    codec = _small_codec()
    starts = torch.tensor([0.0, 1.0])
    ends = torch.tensor([2.0, 3.0])
    with pytest.raises(ValueError, match="durations"):
        codec.encode(starts, ends, [10.0, 20.0, 30.0])
