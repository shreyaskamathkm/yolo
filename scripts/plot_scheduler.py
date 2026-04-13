"""Visualize WarmupBatchScheduler LR and momentum curves.

Simulates a full training run and plots per-batch LR (all param groups) and
momentum across epochs.

Usage:
    python scripts/plot_scheduler.py
"""

import matplotlib
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from yolo.utils.optim_utils import WarmupBatchScheduler, YOLOWarmupPolicy

# ── Config ────────────────────────────────────────────────────────────────────
BASE_LR = 0.01
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.937
EPOCHS = 50
STEPS_PER_EPOCH = 100
WARMUP_EPOCHS = 3
START_MOMENTUM = 0.8
END_MOMENTUM = 0.937
# ─────────────────────────────────────────────────────────────────────────────


def build_optimizer_and_scheduler():
    # Three param groups mirroring create_optimizer: bias, conv, bn
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, bias=False),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, bias=True),
    )

    bias_params = [p for n, p in model.named_parameters() if "bias" in n]
    norm_params = [p for n, p in model.named_parameters() if "weight" in n and "bn" in n]
    conv_params = [p for n, p in model.named_parameters() if "weight" in n and "bn" not in n]

    param_groups = [
        {"params": bias_params, "lr": BASE_LR, "momentum": MOMENTUM, "weight_decay": 0},
        {"params": conv_params, "lr": BASE_LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY},
        {"params": norm_params, "lr": BASE_LR, "momentum": MOMENTUM, "weight_decay": 0},
    ]
    optimizer = SGD(param_groups, lr=BASE_LR)

    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=BASE_LR * 0.01)

    batch_sched = WarmupBatchScheduler(
        optimizer=optimizer,
        scheduler=cosine,
        steps_per_epoch=STEPS_PER_EPOCH,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_policy=YOLOWarmupPolicy(warmup_epochs=WARMUP_EPOCHS),
        start_momentum=START_MOMENTUM,
        end_momentum=END_MOMENTUM,
    )
    return optimizer, batch_sched


def simulate():
    optimizer, scheduler = build_optimizer_and_scheduler()

    total_steps = EPOCHS * STEPS_PER_EPOCH
    lrs = [[] for _ in range(3)]  # one list per param group
    momentums = []

    for _ in range(total_steps):
        for i, group in enumerate(optimizer.param_groups):
            lrs[i].append(group["lr"])
        momentums.append(optimizer.param_groups[0].get("momentum", END_MOMENTUM))

        # Simulate optimizer step then scheduler step (Lightning interval="step")
        optimizer.step()
        scheduler.step()

    return lrs, momentums


def plot(lrs, momentums):
    steps = list(range(len(lrs[0])))
    epoch_boundaries = [e * STEPS_PER_EPOCH for e in range(1, EPOCHS + 1)]

    fig, (ax_lr, ax_mom) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"WarmupBatchScheduler  —  {EPOCHS} epochs × {STEPS_PER_EPOCH} steps/epoch  |  "
        f"warmup={WARMUP_EPOCHS} epochs  |  policy=YOLOWarmupPolicy  |  inner=CosineAnnealingLR",
        fontsize=11,
    )

    labels = ["bias (group 0)", "conv (group 1)", "bn (group 2)"]
    colors = ["#e07b54", "#4c8fbd", "#6abf69"]
    for i, (lr, label, color) in enumerate(zip(lrs, labels, colors)):
        ax_lr.plot(steps, lr, label=label, color=color, linewidth=1.2)
    for x in epoch_boundaries:
        ax_lr.axvline(x, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    ax_lr.axvline(WARMUP_EPOCHS * STEPS_PER_EPOCH, color="red", linewidth=1, linestyle=":", label="warmup end")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.legend(fontsize=8)
    ax_lr.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    ax_mom.plot(steps, momentums, color="#9b59b6", linewidth=1.2)
    for x in epoch_boundaries:
        ax_mom.axvline(x, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    ax_mom.axvline(WARMUP_EPOCHS * STEPS_PER_EPOCH, color="red", linewidth=1, linestyle=":")
    ax_mom.set_ylabel("Momentum")
    ax_mom.set_xlabel("Global Optimizer Step")
    ax_mom.set_ylim(START_MOMENTUM - 0.01, END_MOMENTUM + 0.005)
    ax_mom.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    out = Path(__file__).parent / "scheduler_plot.png"
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")


if __name__ == "__main__":
    lrs, momentums = simulate()
    plot(lrs, momentums)
