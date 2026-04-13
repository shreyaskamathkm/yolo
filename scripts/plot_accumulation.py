"""Visualize GradientAccumulation callback behavior across different world sizes.

Drives the actual GradientAccumulation callback through its hooks using a minimal
mock trainer, so the plot reflects exactly what the callback computes in training.

Each world size gets its own pair of curves (accumulation + effective batch size).

Usage:
    python scripts/plot_accumulation.py
"""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from yolo.utils.model_utils import GradientAccumulation
from yolo.utils.optim_utils import WarmupBatchScheduler, YOLOWarmupPolicy

# ── Config ────────────────────────────────────────────────────────────────────
ACTUAL_BATCH_SIZE = 8
EQUIVALENT_BATCH = 64
EPOCHS = 20
STEPS_PER_EPOCH = 50  # optimizer steps per epoch
WARMUP_EPOCHS = 3
WORLD_SIZES = [1, 2, 4, 8]
# ─────────────────────────────────────────────────────────────────────────────


def _make_data_cfg():
    return SimpleNamespace(
        batch_size=ACTUAL_BATCH_SIZE,
        equivalent_batch_size=EQUIVALENT_BATCH,
    )


def _make_scheduler_cfg():
    warmup = SimpleNamespace(epochs=WARMUP_EPOCHS)
    return SimpleNamespace(warmup=warmup)


def _build_warmup_scheduler(steps_per_epoch: int) -> WarmupBatchScheduler:
    """Build a real WarmupBatchScheduler backed by CosineAnnealingLR."""
    model = nn.Linear(4, 4, bias=True)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.937)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.01 * 0.01)
    return WarmupBatchScheduler(
        optimizer=optimizer,
        scheduler=cosine,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_policy=YOLOWarmupPolicy(warmup_epochs=WARMUP_EPOCHS),
    )


def _make_trainer(world_size: int, steps_per_epoch: int):
    """Minimal mock trainer with the attributes the callback reads."""
    batch_sched = _build_warmup_scheduler(steps_per_epoch)
    sched_cfg = SimpleNamespace(scheduler=batch_sched)

    trainer = SimpleNamespace(
        world_size=world_size,
        lr_scheduler_configs=[sched_cfg],
        global_step=0,
        accumulate_grad_batches=1,
    )
    return trainer


def simulate_world(world_size: int):
    """Drive the callback hooks exactly as Lightning would and record results."""
    callback = GradientAccumulation(_make_data_cfg(), _make_scheduler_cfg())
    trainer = _make_trainer(world_size, STEPS_PER_EPOCH)

    # Lightning lifecycle: setup → on_train_start → per-step loop
    callback.setup(trainer, pl_module=None, stage="fit")
    callback.on_train_start(trainer, pl_module=None)

    accum_values = []
    eff_batch_values = []
    total_steps = EPOCHS * STEPS_PER_EPOCH

    for global_step in range(total_steps):
        trainer.global_step = global_step
        callback.on_train_batch_start(trainer, pl_module=None)

        accum = trainer.accumulate_grad_batches
        accum_values.append(accum)
        eff_batch_values.append(accum * ACTUAL_BATCH_SIZE * world_size)

    print(
        f"world={world_size:2d}  max_accum={callback.max_accumulation}  "
        f"warmup_batches={callback.warmup_batches}  "
        f"eff_batch_at_end={eff_batch_values[-1]}"
    )
    return accum_values, eff_batch_values, callback.max_accumulation, callback.warmup_batches


def plot(results: dict):
    total_steps = EPOCHS * STEPS_PER_EPOCH
    steps = list(range(total_steps))
    epoch_boundaries = [e * STEPS_PER_EPOCH for e in range(1, EPOCHS + 1)]
    warmup_end = WARMUP_EPOCHS * STEPS_PER_EPOCH

    colors = ["#4c8fbd", "#e07b54", "#6abf69", "#9b59b6"]

    fig, (ax_acc, ax_eff) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"GradientAccumulation callback  —  "
        f"batch={ACTUAL_BATCH_SIZE} per GPU  |  target={EQUIVALENT_BATCH}  |  "
        f"warmup={WARMUP_EPOCHS} epochs  |  {STEPS_PER_EPOCH} steps/epoch",
        fontsize=11,
    )

    for (world_size, (accum_vals, eff_vals, max_accum, _)), color in zip(results.items(), colors):
        label = f"world={world_size}  (max_accum={max_accum})"
        ax_acc.step(steps, accum_vals, where="post", color=color, linewidth=1.4, label=label)
        ax_eff.step(steps, eff_vals, where="post", color=color, linewidth=1.4, label=label)

    for ax in (ax_acc, ax_eff):
        for x in epoch_boundaries:
            ax.axvline(x, color="gray", linewidth=0.3, linestyle="--", alpha=0.4)
        ax.axvline(warmup_end, color="red", linewidth=1.0, linestyle=":", label="warmup end")

    ax_acc.axhline(1, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_eff.axhline(EQUIVALENT_BATCH, color="black", linewidth=0.8, linestyle="--", label=f"target={EQUIVALENT_BATCH}")

    ax_acc.set_ylabel("accumulate_grad_batches")
    ax_acc.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.legend(fontsize=8)

    ax_eff.set_ylabel("Effective Batch Size\n(accum × batch × world)")
    ax_eff.set_xlabel("Global Optimizer Step")
    ax_eff.legend(fontsize=8)

    plt.tight_layout()
    out = Path(__file__).parent / "accumulation_plot.png"
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")


if __name__ == "__main__":
    results = {}
    for ws in WORLD_SIZES:
        results[ws] = simulate_world(ws)
    plot(results)
