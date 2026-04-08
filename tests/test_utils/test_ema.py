"""Tests for the EMA (Exponential Moving Average) Lightning callback."""

from math import exp

import pytest
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from tests.conftest import DummyModule, DummyModuleWithVal, TinyDataset
from yolo.utils.model_utils import EMA

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_loader(num_samples: int = 8, batch_size: int = 2) -> DataLoader:
    return DataLoader(TinyDataset(num_samples=num_samples), batch_size=batch_size, shuffle=False)


def _make_module(num_samples: int = 8, batch_size: int = 2) -> DummyModule:
    return DummyModule(_make_loader(num_samples, batch_size))


def _fit(module, callbacks, max_epochs=1, limit_train_batches=4, accumulate_grad_batches=1, **kwargs):
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        **kwargs,
    )
    trainer.fit(module)
    return trainer


# ── Init ─────────────────────────────────────────────────────────────────────


def test_init_defaults():
    ema = EMA()
    assert ema.decay == 0.9999
    assert ema.tau == 2000.0
    assert ema.step == 0
    assert ema.batch_count == 0
    assert ema.shadow is None
    assert ema._training_weights is None


def test_init_custom_params():
    ema = EMA(decay=0.99, tau=500.0)
    assert ema.decay == 0.99
    assert ema.tau == 500.0


# ── _beta() ───────────────────────────────────────────────────────────────────


def test_beta_increases_with_step():
    """Effective smoothing coefficient grows toward decay as step increases."""
    ema = EMA(decay=0.9, tau=100.0)
    betas = []
    for s in [1, 10, 100, 1000]:
        ema.step = s
        betas.append(ema._beta())
    assert betas == sorted(betas), "beta should be monotonically increasing"
    assert betas[-1] < ema.decay, "beta should never exceed decay"


# ── First update ──────────────────────────────────────────────────────────────


def test_setup_clones_model_before_training():
    """setup() clones model weights into shadow before any training batch runs."""
    torch.manual_seed(0)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()

    # Simulate what Lightning calls before fit
    stub_trainer = type("T", (), {"world_size": 1})()
    ema.setup(stub_trainer, module, stage="fit")

    assert ema.shadow is not None
    assert ema.step == 0
    for key, param in module.model.state_dict().items():
        assert torch.allclose(
            ema.shadow[key], param, atol=1e-8
        ), f"{key}: setup() should clone model weights exactly — no decay applied"


# ── Update formula ────────────────────────────────────────────────────────────


def test_update_blends_in_place():
    """shadow = beta * shadow + (1 - beta) * model, verified step by step."""
    torch.manual_seed(1)
    ema = EMA(decay=0.9, tau=100.0)
    module = _make_module()

    _fit(module, [ema], limit_train_batches=1)
    shadow_after_step1 = {k: v.clone() for k, v in ema.shadow.items()}

    with torch.no_grad():
        for p in module.model.parameters():
            p.add_(0.05)
    model_snapshot = {k: v.clone() for k, v in module.model.state_dict().items()}

    ema.step = 2
    ema.update(module)

    assert ema.shadow is not None
    beta = ema._beta()
    for key in ema.shadow:
        expected = beta * shadow_after_step1[key] + (1.0 - beta) * model_snapshot[key].detach()
        assert torch.allclose(
            ema.shadow[key], expected, atol=1e-5
        ), f"{key}: shadow does not match beta*shadow + (1-beta)*model"


def test_step_counter_increments_across_epochs():
    """step accumulates continuously across multiple epochs."""
    torch.manual_seed(2)
    ema = EMA(decay=0.999, tau=2000.0)
    train_loader = _make_loader(num_samples=12)
    val_loader = _make_loader(num_samples=4)
    module = DummyModuleWithVal(train_loader, val_loader)

    steps_at_epoch_end = []

    class StepRecorder(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            steps_at_epoch_end.append(ema.step)

    _fit(module, [ema, StepRecorder()], max_epochs=3, limit_train_batches=3, num_sanity_val_steps=0)

    assert steps_at_epoch_end == [3, 6, 9]


# ── apply_shadow / restore ────────────────────────────────────────────────────


def test_apply_shadow_loads_shadow_into_model():
    """apply_shadow replaces live model weights with the shadow copy."""
    torch.manual_seed(3)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()
    _fit(module, [ema], limit_train_batches=4)

    ema.apply_shadow(module)
    for key in ema.shadow:
        assert torch.allclose(module.model.state_dict()[key], ema.shadow[key], atol=1e-6)


def test_restore_returns_to_training_weights():
    """restore() recovers exactly the weights present before apply_shadow."""
    torch.manual_seed(4)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()
    _fit(module, [ema], limit_train_batches=4)

    live_weights = {k: v.clone() for k, v in module.model.state_dict().items()}
    ema.apply_shadow(module)
    ema.restore(module)

    for key in live_weights:
        assert torch.allclose(module.model.state_dict()[key], live_weights[key], atol=1e-8)


def test_apply_shadow_noop_before_training():
    """apply_shadow is a no-op when shadow is still None."""
    torch.manual_seed(5)
    ema = EMA()
    module = _make_module()
    original = {k: v.clone() for k, v in module.model.state_dict().items()}

    ema.apply_shadow(module)

    for key in original:
        assert torch.allclose(module.model.state_dict()[key], original[key], atol=1e-8)


def test_restore_noop_without_snapshot():
    """restore() is a no-op when _training_weights is None."""
    torch.manual_seed(6)
    ema = EMA()
    module = _make_module()
    original = {k: v.clone() for k, v in module.model.state_dict().items()}

    ema.restore(module)

    for key in original:
        assert torch.allclose(module.model.state_dict()[key], original[key], atol=1e-8)


# ── Validation hooks ──────────────────────────────────────────────────────────


def test_validation_runs_on_shadow_weights_then_restores():
    """on_validation_start swaps in shadow; on_validation_end restores training weights."""
    torch.manual_seed(7)
    ema = EMA(decay=0.999, tau=2000.0)
    train_loader = _make_loader(num_samples=8)
    val_loader = _make_loader(num_samples=4)
    module = DummyModuleWithVal(train_loader, val_loader)

    captured = {}

    class StateCapture(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            captured["after_train"] = {k: v.clone() for k, v in pl_module.model.state_dict().items()}

        def on_validation_start(self, trainer, pl_module):
            captured["val_start"] = {k: v.clone() for k, v in pl_module.model.state_dict().items()}

        def on_validation_end(self, trainer, pl_module):
            captured["val_end"] = {k: v.clone() for k, v in pl_module.model.state_dict().items()}

    _fit(module, [ema, StateCapture()], limit_train_batches=3, num_sanity_val_steps=0)

    for key in ema.shadow:
        assert torch.allclose(
            captured["val_start"][key], ema.shadow[key], atol=1e-6
        ), "During validation model should hold shadow weights"
    for key in captured["after_train"]:
        assert torch.allclose(
            captured["after_train"][key], captured["val_end"][key], atol=1e-8
        ), "After validation model should be back to training weights"


def test_batch_count_resets_on_validation_start():
    """on_validation_start always resets batch_count to 0."""
    torch.manual_seed(8)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()
    trainer = _fit(module, [ema], limit_train_batches=3)

    assert ema.batch_count == 3
    ema.on_validation_start(trainer, module)
    assert ema.batch_count == 0


def test_shadow_evolves_across_epochs():
    """Shadow weights change between consecutive epochs as training progresses."""
    torch.manual_seed(9)
    ema = EMA(decay=0.999, tau=2000.0)
    train_loader = _make_loader(num_samples=8)
    val_loader = _make_loader(num_samples=4)
    module = DummyModuleWithVal(train_loader, val_loader)

    snapshots = []

    class ShadowSnapshot(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if ema.shadow is not None:
                snapshots.append({k: v.clone() for k, v in ema.shadow.items()})

    _fit(module, [ema, ShadowSnapshot()], max_epochs=3, limit_train_batches=2, num_sanity_val_steps=0)

    assert len(snapshots) == 3
    for i in range(2):
        for key in snapshots[i]:
            assert not torch.allclose(
                snapshots[i][key], snapshots[i + 1][key], atol=1e-6
            ), f"{key}: shadow should differ between epoch {i} and {i + 1}"


# ── Gradient accumulation ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "accumulate, limit_batches, expected_steps",
    [
        (1, 4, 4),
        (2, 4, 2),
        (4, 8, 2),
    ],
)
def test_update_frequency_matches_accumulation(accumulate, limit_batches, expected_steps):
    """EMA step fires only after a full accumulation cycle."""
    torch.manual_seed(10)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module(num_samples=limit_batches * 2)
    _fit(module, [ema], limit_train_batches=limit_batches, accumulate_grad_batches=accumulate)

    assert ema.step == expected_steps


def test_buffers_copied_directly_not_lerped():
    """BatchNorm buffers (running_mean, running_var) must be copied, not blended.

    After each update the shadow buffer should equal the model buffer exactly.
    If buffers were lerp'd instead they would lag behind the model statistics,
    producing a blended distribution that is incorrect for both train and EMA.
    """
    torch.manual_seed(16)

    class BNModule(LightningModule):
        def __init__(self, loader):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
            self._loader = loader

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, _):
            x, _ = batch
            return self(x).mean()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

        def train_dataloader(self):
            return self._loader

    ema = EMA(decay=0.9, tau=10.0)
    module = BNModule(_make_loader(num_samples=8))
    _fit(module, [ema], limit_train_batches=4)

    # After each update, shadow buffers must exactly match current model buffers.
    # A lerp would leave them lagging; a copy makes them equal.
    for key, buf in module.model.named_buffers():
        assert torch.allclose(
            ema.shadow[key], buf.detach(), atol=1e-8
        ), f"Buffer '{key}' should be copied directly into shadow, not lerp'd"


def test_no_update_before_accumulation_completes():
    """With accumulate=4 and only 3 batches, no EMA step fires (step stays 0).

    setup() still initialises shadow before training, but no lerp update
    should have been applied since the accumulation cycle never completed.
    """
    torch.manual_seed(11)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module(num_samples=8)
    initial_shadow = {k: v.clone() for k, v in module.model.state_dict().items()}
    _fit(module, [ema], limit_train_batches=3, accumulate_grad_batches=4)

    assert ema.step == 0
    assert ema.shadow is not None  # setup() initialised it
    for key in ema.shadow:
        assert torch.allclose(
            ema.shadow[key], initial_shadow[key], atol=1e-8
        ), f"{key}: shadow should be unchanged — no accumulation cycle completed"


# ── Checkpoint ────────────────────────────────────────────────────────────────


def test_checkpoint_round_trip():
    """Save and load preserves step, batch_count, and all shadow weights."""
    torch.manual_seed(12)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()
    trainer = _fit(module, [ema], limit_train_batches=3)

    ckpt = {}
    ema.on_save_checkpoint(trainer, module, ckpt)

    assert EMA._CHECKPOINT_KEY in ckpt
    assert ckpt["ema_step"] == ema.step
    assert ckpt["ema_batch_count"] == ema.batch_count

    loaded = EMA(decay=0.999, tau=2000.0)
    loaded.on_load_checkpoint(trainer, module, ckpt)

    assert loaded.step == ema.step
    assert loaded.batch_count == ema.batch_count
    for key in ema.shadow:
        assert torch.allclose(loaded.shadow[key], ema.shadow[key], atol=1e-8)


def test_checkpoint_tensors_saved_as_cpu():
    """Shadow tensors in checkpoint are always on CPU for portability."""
    torch.manual_seed(13)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()
    trainer = _fit(module, [ema], limit_train_batches=2)

    ckpt = {}
    ema.on_save_checkpoint(trainer, module, ckpt)

    for key, val in ckpt[EMA._CHECKPOINT_KEY].items():
        assert val.device.type == "cpu", f"{key} should be on CPU in the checkpoint"


def test_save_checkpoint_skipped_before_training():
    """on_save_checkpoint does nothing when shadow is still None."""
    ema = EMA()
    module = _make_module()
    trainer = Trainer(accelerator="cpu", devices=1, logger=False)

    ckpt = {}
    ema.on_save_checkpoint(trainer, module, ckpt)

    assert EMA._CHECKPOINT_KEY not in ckpt


def test_loaded_shadow_placed_on_model_device():
    """After on_load_checkpoint all shadow tensors live on the model's device."""
    torch.manual_seed(14)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()
    trainer = _fit(module, [ema], limit_train_batches=2)

    ckpt = {}
    ema.on_save_checkpoint(trainer, module, ckpt)

    loaded = EMA(decay=0.999, tau=2000.0)
    loaded.on_load_checkpoint(trainer, module, ckpt)

    target = next(module.model.parameters()).device
    for key, val in loaded.shadow.items():
        assert val.device == target, f"{key} is on {val.device}, expected {target}"


def test_training_continues_after_load():
    """step increments and shadow updates correctly after resuming from checkpoint."""
    torch.manual_seed(15)
    ema = EMA(decay=0.999, tau=2000.0)
    module = _make_module()
    trainer = _fit(module, [ema], limit_train_batches=2)

    ckpt = {}
    ema.on_save_checkpoint(trainer, module, ckpt)
    step_before = ema.step
    shadow_before = {k: v.clone() for k, v in ema.shadow.items()}

    loaded = EMA(decay=0.999, tau=2000.0)
    loaded.on_load_checkpoint(trainer, module, ckpt)

    with torch.no_grad():
        for p in module.model.parameters():
            p.add_(0.1)

    stub = type("T", (), {"accumulate_grad_batches": 1})()
    loaded.on_train_batch_end(stub, module)

    assert loaded.step == step_before + 1
    for key in shadow_before:
        assert not torch.allclose(
            loaded.shadow[key], shadow_before[key], atol=1e-8
        ), f"{key}: shadow should have updated after continuation step"