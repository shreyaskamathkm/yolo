"""Tests for optim_utils: lerp, warmup policies, and WarmupBatchScheduler."""

import math

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from yolo.utils.optim_utils import (
    LinearWarmupPolicy,
    WarmupBatchScheduler,
    YOLOWarmupPolicy,
    lerp,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_optimizer(base_lr: float = 0.01, num_groups: int = 3) -> SGD:
    """Three-group optimizer mirroring create_optimizer (bias, conv, bn)."""
    params = [{"params": [nn.Parameter(torch.zeros(1))], "lr": base_lr} for _ in range(num_groups)]
    return SGD(params, lr=base_lr)


def _make_scheduler(
    optimizer,
    steps_per_epoch: int = 10,
    warmup_epochs: int = 3,
    epochs: int = 20,
    policy=None,
) -> WarmupBatchScheduler:
    inner = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    return WarmupBatchScheduler(
        optimizer=optimizer,
        scheduler=inner,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        warmup_policy=policy,
        start_momentum=0.8,
        end_momentum=0.937,
    )


# ── lerp ──────────────────────────────────────────────────────────────────────


def test_lerp_basic():
    assert math.isclose(lerp(0.0, 1.0, 1, 4), 0.25)
    assert math.isclose(lerp(0.0, 1.0, 4, 4), 1.0)
    assert math.isclose(lerp(2.0, 4.0, 1, 2), 3.0)


def test_lerp_zero_total_returns_end():
    assert lerp(0.0, 5.0, 0, 0) == 5.0


def test_lerp_start_equals_end():
    assert lerp(3.0, 3.0, 5, 10) == 3.0


# ── LinearWarmupPolicy ────────────────────────────────────────────────────────


def test_linear_policy_start_lr_always_zero():
    policy = LinearWarmupPolicy(warmup_epochs=3)
    for group_idx in range(3):
        assert policy.start_lr(group_idx, 0.01) == 0.0


def test_linear_policy_target_lr_ramps_uniformly():
    base_lr = 0.01
    warmup_epochs = 4
    policy = LinearWarmupPolicy(warmup_epochs=warmup_epochs)
    for epoch in range(warmup_epochs):
        for group_idx in range(3):
            expected = lerp(0.0, base_lr, epoch + 1, warmup_epochs)
            assert math.isclose(policy.target_lr(epoch, group_idx, base_lr), expected)


def test_linear_policy_converges_to_base_lr():
    base_lr = 0.01
    policy = LinearWarmupPolicy(warmup_epochs=3)
    assert math.isclose(policy.target_lr(2, 0, base_lr), base_lr)


# ── YOLOWarmupPolicy ──────────────────────────────────────────────────────────


def test_yolo_policy_start_lr_bias_is_10x():
    policy = YOLOWarmupPolicy(warmup_epochs=3)
    assert math.isclose(policy.start_lr(0, 0.01), 0.1)


def test_yolo_policy_start_lr_others_are_zero():
    policy = YOLOWarmupPolicy(warmup_epochs=3)
    assert policy.start_lr(1, 0.01) == 0.0
    assert policy.start_lr(2, 0.01) == 0.0


def test_yolo_policy_bias_drops_over_warmup():
    policy = YOLOWarmupPolicy(warmup_epochs=3)
    base_lr = 0.01
    targets = [policy.target_lr(e, 0, base_lr) for e in range(3)]
    assert targets[0] > targets[1] > targets[2], "Bias LR must strictly drop each epoch"


def test_yolo_policy_others_rise_over_warmup():
    policy = YOLOWarmupPolicy(warmup_epochs=3)
    base_lr = 0.01
    targets = [policy.target_lr(e, 1, base_lr) for e in range(3)]
    assert targets[0] < targets[1] < targets[2], "Conv/BN LR must strictly rise each epoch"


def test_yolo_policy_all_groups_converge_at_warmup_end():
    """Both lambda1 and lambda2 must equal 1.0 at the last warmup epoch."""
    base_lr = 0.01
    warmup_epochs = 3
    policy = YOLOWarmupPolicy(warmup_epochs=warmup_epochs)
    for group_idx in range(3):
        result = policy.target_lr(warmup_epochs - 1, group_idx, base_lr)
        assert math.isclose(
            result, base_lr, rel_tol=1e-6
        ), f"Group {group_idx} did not converge to base_lr at warmup end"


# ── WarmupBatchScheduler — LR behavior ───────────────────────────────────────


def test_scheduler_init_lr_with_yolo_policy():
    """At construction, LR is set to batch-0 interpolated value (not raw start_lr)."""
    base_lr = 0.01
    steps = 10
    optimizer = _make_optimizer(base_lr)
    policy = YOLOWarmupPolicy(warmup_epochs=3)
    scheduler = _make_scheduler(optimizer, steps_per_epoch=steps, policy=policy)

    # Bias (group 0): lerp(10*base_lr, lambda2(0)*base_lr, 1, steps)
    bias_start = policy.start_lr(0, base_lr)
    bias_end = policy.target_lr(0, 0, base_lr)
    expected_bias = lerp(bias_start, bias_end, 1, steps)
    assert math.isclose(optimizer.param_groups[0]["lr"], expected_bias, rel_tol=1e-5)

    # Conv (group 1): lerp(0, lambda1(0)*base_lr, 1, steps)
    conv_start = policy.start_lr(1, base_lr)
    conv_end = policy.target_lr(0, 1, base_lr)
    expected_conv = lerp(conv_start, conv_end, 1, steps)
    assert math.isclose(optimizer.param_groups[1]["lr"], expected_conv, rel_tol=1e-5)


def test_scheduler_bias_lr_drops_not_spikes():
    """Bias group LR should start high and decrease — never go below epoch target mid-warmup."""
    base_lr = 0.01
    steps = 20
    warmup_epochs = 3
    optimizer = _make_optimizer(base_lr)
    policy = YOLOWarmupPolicy(warmup_epochs=warmup_epochs)
    scheduler = _make_scheduler(optimizer, steps_per_epoch=steps, warmup_epochs=warmup_epochs, policy=policy)

    bias_lrs = []
    for _ in range(warmup_epochs * steps):
        bias_lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Overall: bias should start near 10x and end near base_lr
    assert bias_lrs[0] > bias_lrs[-1], "Bias LR must be higher at start than end of warmup"
    assert bias_lrs[0] > base_lr, "Bias LR must start above base_lr"
    assert math.isclose(bias_lrs[-1], base_lr, rel_tol=0.05), "Bias LR must end near base_lr"


def test_scheduler_conv_lr_rises():
    """Conv/BN groups should rise from near-0 to base_lr during warmup."""
    base_lr = 0.01
    steps = 20
    warmup_epochs = 3
    optimizer = _make_optimizer(base_lr)
    policy = YOLOWarmupPolicy(warmup_epochs=warmup_epochs)
    scheduler = _make_scheduler(optimizer, steps_per_epoch=steps, warmup_epochs=warmup_epochs, policy=policy)

    conv_lrs = []
    for _ in range(warmup_epochs * steps):
        conv_lrs.append(optimizer.param_groups[1]["lr"])
        optimizer.step()
        scheduler.step()

    assert conv_lrs[0] < conv_lrs[-1], "Conv LR must be lower at start than end of warmup"
    assert math.isclose(conv_lrs[-1], base_lr, rel_tol=0.05), "Conv LR must end near base_lr"


def test_scheduler_inner_not_stepped_during_warmup():
    """Inner scheduler must not advance while still inside warmup epochs."""
    base_lr = 0.01
    steps = 5
    warmup_epochs = 3
    optimizer = _make_optimizer(base_lr)
    inner = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0)
    inner_last_epoch_at_init = inner.last_epoch

    scheduler = WarmupBatchScheduler(
        optimizer=optimizer,
        scheduler=inner,
        steps_per_epoch=steps,
        warmup_epochs=warmup_epochs,
        warmup_policy=YOLOWarmupPolicy(warmup_epochs=warmup_epochs),
    )

    # Run up to but NOT including the warmup→post-warmup boundary
    for _ in range(warmup_epochs * steps - 1):
        optimizer.step()
        scheduler.step()

    assert inner.last_epoch == inner_last_epoch_at_init, "Inner scheduler must not be stepped before warmup ends"


def test_scheduler_inner_stepped_after_warmup():
    """Inner scheduler should advance once per epoch boundary after warmup ends.

    The transition into epoch `warmup_epochs` counts as the first post-warmup step,
    so after `post_warmup_epochs` additional full epochs the inner scheduler will
    have been stepped `post_warmup_epochs + 1` times in total.
    """
    base_lr = 0.01
    steps = 5
    warmup_epochs = 2
    post_warmup_epochs = 3
    optimizer = _make_optimizer(base_lr)
    inner = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0)
    inner_last_epoch_at_init = inner.last_epoch

    scheduler = WarmupBatchScheduler(
        optimizer=optimizer,
        scheduler=inner,
        steps_per_epoch=steps,
        warmup_epochs=warmup_epochs,
        warmup_policy=YOLOWarmupPolicy(warmup_epochs=warmup_epochs),
    )

    total_steps = (warmup_epochs + post_warmup_epochs) * steps
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()

    # +1 because the warmup→post-warmup transition itself steps the inner scheduler
    assert inner.last_epoch == inner_last_epoch_at_init + post_warmup_epochs + 1


# ── WarmupBatchScheduler — momentum ──────────────────────────────────────────


def test_momentum_rises_during_warmup():
    base_lr = 0.01
    steps = 10
    warmup_epochs = 3
    optimizer = _make_optimizer(base_lr)
    for g in optimizer.param_groups:
        g["momentum"] = 0.8

    scheduler = _make_scheduler(optimizer, steps_per_epoch=steps, warmup_epochs=warmup_epochs)
    momentums = []
    for _ in range(warmup_epochs * steps):
        momentums.append(optimizer.param_groups[0]["momentum"])
        optimizer.step()
        scheduler.step()

    assert momentums[0] < momentums[-1], "Momentum must rise during warmup"
    assert math.isclose(momentums[-1], 0.937, rel_tol=1e-3)


def test_momentum_flat_after_warmup():
    base_lr = 0.01
    steps = 5
    warmup_epochs = 2
    optimizer = _make_optimizer(base_lr)
    for g in optimizer.param_groups:
        g["momentum"] = 0.8

    scheduler = _make_scheduler(optimizer, steps_per_epoch=steps, warmup_epochs=warmup_epochs)

    # Run through warmup
    for _ in range(warmup_epochs * steps):
        optimizer.step()
        scheduler.step()

    # Collect post-warmup momentums
    post_momentums = []
    for _ in range(steps * 3):
        post_momentums.append(optimizer.param_groups[0]["momentum"])
        optimizer.step()
        scheduler.step()

    assert all(math.isclose(m, 0.937, rel_tol=1e-5) for m in post_momentums)


# ── WarmupBatchScheduler — checkpoint round-trip ─────────────────────────────


def test_checkpoint_roundtrip_mid_warmup():
    """Resume from checkpoint mid-warmup and produce identical LRs."""
    base_lr = 0.01
    steps = 8
    warmup_epochs = 3
    resume_at = warmup_epochs * steps // 2  # halfway through warmup

    # Run original scheduler to resume_at
    opt_a = _make_optimizer(base_lr)
    sched_a = _make_scheduler(
        opt_a, steps_per_epoch=steps, warmup_epochs=warmup_epochs, policy=YOLOWarmupPolicy(warmup_epochs=warmup_epochs)
    )
    for _ in range(resume_at):
        opt_a.step()
        sched_a.step()

    saved_state = sched_a.state_dict()

    # Clone optimizer and restore scheduler from checkpoint
    opt_b = _make_optimizer(base_lr)
    inner_b = CosineAnnealingLR(opt_b, T_max=20, eta_min=0.0)
    sched_b = WarmupBatchScheduler(
        optimizer=opt_b,
        scheduler=inner_b,
        steps_per_epoch=steps,
        warmup_epochs=warmup_epochs,
        warmup_policy=YOLOWarmupPolicy(warmup_epochs=warmup_epochs),
    )
    sched_b.load_state_dict(saved_state)

    # Both schedulers must produce identical LRs for the next N steps
    for _ in range(steps * 2):
        lrs_a = [g["lr"] for g in opt_a.param_groups]
        lrs_b = [g["lr"] for g in opt_b.param_groups]
        for a, b in zip(lrs_a, lrs_b):
            assert math.isclose(a, b, rel_tol=1e-6), f"LR mismatch after restore: {a} vs {b}"
        opt_a.step()
        sched_a.step()
        opt_b.step()
        sched_b.step()


def test_checkpoint_roundtrip_post_warmup():
    """Resume from checkpoint post-warmup and produce identical LRs."""
    base_lr = 0.01
    steps = 5
    warmup_epochs = 2
    resume_at = (warmup_epochs + 3) * steps  # 3 epochs into post-warmup

    opt_a = _make_optimizer(base_lr)
    sched_a = _make_scheduler(
        opt_a, steps_per_epoch=steps, warmup_epochs=warmup_epochs, policy=YOLOWarmupPolicy(warmup_epochs=warmup_epochs)
    )
    for _ in range(resume_at):
        opt_a.step()
        sched_a.step()

    saved_state = sched_a.state_dict()

    opt_b = _make_optimizer(base_lr)
    inner_b = CosineAnnealingLR(opt_b, T_max=20, eta_min=0.0)
    sched_b = WarmupBatchScheduler(
        optimizer=opt_b,
        scheduler=inner_b,
        steps_per_epoch=steps,
        warmup_epochs=warmup_epochs,
        warmup_policy=YOLOWarmupPolicy(warmup_epochs=warmup_epochs),
    )
    sched_b.load_state_dict(saved_state)

    for _ in range(steps * 3):
        lrs_a = [g["lr"] for g in opt_a.param_groups]
        lrs_b = [g["lr"] for g in opt_b.param_groups]
        for a, b in zip(lrs_a, lrs_b):
            assert math.isclose(a, b, rel_tol=1e-6), f"LR mismatch after restore: {a} vs {b}"
        opt_a.step()
        sched_a.step()
        opt_b.step()
        sched_b.step()
