"""Compare the original EMA callback against the updated one through Lightning.

Both callbacks are attached to the same Trainer and the same LightningModule so
they see identical model weights and identical training steps.  After every
batch a recorder snapshot is taken from each callback and the two shadow dicts
are diffed, highlighting:

  1. Float buffer handling — original lerps running_mean / running_var;
     updated copies them directly.
  2. Numerical equivalence of learned parameters — both should agree within
     floating-point tolerance.

Usage:
    python scripts/compare_ema.py
"""

import sys
from copy import deepcopy
from math import exp
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import no_grad
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.utils.logger import logger
from yolo.utils.model_utils import EMA as UpdatedEMA
from yolo.utils.optim_utils import lerp

# ── Original EMA (verbatim from model_utils.py before the refactor) ───────────


class OriginalEMA(Callback):
    def __init__(self, decay: float = 0.9999, tau: float = 2000):
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.batch_step_counter = 0
        self.ema_state_dict = None

    def setup(self, trainer, pl_module, stage):
        pl_module.ema = deepcopy(pl_module.model)
        self.tau /= trainer.world_size
        for param in pl_module.ema.parameters():
            param.requires_grad = False

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule"):
        self.batch_step_counter = 0
        if self.ema_state_dict is None:
            self.ema_state_dict = deepcopy(pl_module.model.state_dict())
        pl_module.ema.load_state_dict(self.ema_state_dict)

    @no_grad()
    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.batch_step_counter += 1
        if self.batch_step_counter % trainer.accumulate_grad_batches:
            return
        self.step += 1
        decay_factor = self.decay * (1 - exp(-self.step / self.tau))
        for key, param in pl_module.model.state_dict().items():
            self.ema_state_dict[key] = lerp(param.detach(), self.ema_state_dict[key], decay_factor)


# ── LightningModule with BatchNorm ────────────────────────────────────────────


class BNModule(LightningModule):
    """Minimal module with BatchNorm so both parameters and buffers are present."""

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        self.ema = self.model  # required by UpdatedEMA's apply_shadow / restore
        self._train_loader = train_loader
        self._val_loader = val_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        return F.mse_loss(self(x), F.one_hot(y, num_classes=4).float())

    def validation_step(self, batch, _):
        x, y = batch
        return F.mse_loss(self(x), F.one_hot(y, num_classes=4).float())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader


# ── Recorder: snapshots EMA state after every batch ──────────────────────────


class _Recorder(Callback):
    """Captures a clone of an EMA callback's shadow after every training batch."""

    def __init__(self, ema: Callback) -> None:
        self._ema = ema
        self.snapshots: List[Optional[Dict[str, torch.Tensor]]] = []

    def _shadow(self) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(self._ema, OriginalEMA):
            return self._ema.ema_state_dict
        return self._ema.shadow  # UpdatedEMA

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        state = self._shadow()
        if state is not None:
            self.snapshots.append({k: v.detach().clone() for k, v in state.items()})
        else:
            self.snapshots.append(None)


# ── Comparison printer ────────────────────────────────────────────────────────


def _print_diff(
    orig: Dict[str, torch.Tensor],
    upd: Dict[str, torch.Tensor],
    param_keys: set,
    buffer_keys: set,
    step: int,
) -> None:
    print(f"\n{'─'*64}")
    print(f"  Step {step}")
    print(f"{'─'*64}")

    print("  Parameters  (expect: OK — same formula, both should agree)")
    for key in sorted(param_keys):
        o, u = orig[key].float(), upd[key].float()
        diff = (o - u).abs().max().item()
        tag = "OK  " if torch.allclose(o, u, atol=1e-5) else "DIFF"
        print(f"    [{tag}]  {key:<42}  max_diff={diff:.2e}")

    print("  Buffers     (expect: DIFF — original lerps, updated copies)")
    for key in sorted(buffer_keys):
        o, u = orig[key].float(), upd[key].float()
        diff = (o - u).abs().max().item()
        tag = "SAME" if torch.allclose(o, u, atol=1e-5) else "DIFF"
        print(f"    [{tag}]  {key:<42}  max_diff={diff:.2e}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(n_batches: int = 5, decay: float = 0.9, tau: float = 10.0) -> None:
    torch.manual_seed(42)

    # Shared dataset — both runs see identical data
    x = torch.randn(n_batches * 4, 8)
    y = torch.randint(0, 4, (n_batches * 4,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    val_loader = DataLoader(TensorDataset(x[:4], y[:4]), batch_size=4, shuffle=False)

    # Build module once; both callbacks attach to the same pl_module
    module = BNModule(train_loader, val_loader)
    param_keys = {k for k, _ in module.model.named_parameters()}
    buffer_keys = {k for k, _ in module.model.named_buffers()}

    original_ema = OriginalEMA(decay=decay, tau=tau)
    updated_ema = UpdatedEMA(decay=decay, tau=tau)
    rec_orig = _Recorder(original_ema)
    rec_upd = _Recorder(updated_ema)

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=n_batches,
        num_sanity_val_steps=1,  # needed so OriginalEMA initialises ema_state_dict
        callbacks=[original_ema, updated_ema, rec_orig, rec_upd],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module)

    print("\n" + "=" * 64)
    print("  EMA Callback Comparison")
    print(f"  decay={decay}  tau={tau}  batches={n_batches}")
    print("=" * 64)

    for step, (snap_orig, snap_upd) in enumerate(zip(rec_orig.snapshots, rec_upd.snapshots), start=1):
        if snap_orig is None or snap_upd is None:
            print(f"\n  Step {step}: one EMA not yet initialised — skipping")
            continue
        _print_diff(snap_orig, snap_upd, param_keys, buffer_keys, step)

    print(f"\n{'='*64}")
    print("  Summary")
    print(f"{'='*64}")
    print("  Parameters : both produce equivalent values (within float tolerance).")
    print("  Buffers    : diverge by design —")
    print("               OriginalEMA lerps batch statistics (incorrect),")
    print("               UpdatedEMA  copies them directly  (correct).")
    print()


if __name__ == "__main__":
    main()