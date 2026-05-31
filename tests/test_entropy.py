"""Entropy helper checks."""

from __future__ import annotations

import pytest
import torch as th

from sb3x.common.entropy import entropy_loss


def test_entropy_loss_can_weight_named_components() -> None:
    """Per-group entropy weights should replace the default summed entropy."""
    log_prob = th.zeros(2, dtype=th.float32)
    components = {
        "steer": th.tensor([1.0, 3.0]),
        "pitch": th.tensor([2.0, 4.0]),
    }

    loss, metrics = entropy_loss(
        log_prob=log_prob,
        entropy=components["steer"] + components["pitch"],
        entropy_components=components,
        entropy_group_weights={"pitch": 0.5},
    )

    th.testing.assert_close(loss, th.tensor(-1.5))
    assert metrics == {"pitch": 3.0, "steer": 2.0}


def test_entropy_loss_rejects_unmatched_group_weights() -> None:
    """Configured group weights must match at least one entropy component."""
    with pytest.raises(ValueError, match="do not match"):
        entropy_loss(
            log_prob=th.zeros(2, dtype=th.float32),
            entropy=None,
            entropy_components={"steer": th.ones(2)},
            entropy_group_weights={"pitch": 1.0},
        )
