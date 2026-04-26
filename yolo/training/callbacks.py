# TODO Phase 2: update imports — schemas from yolo.config.schemas.training,
#               lerp from yolo.training.optim
from math import exp
from typing import Optional

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from yolo.config.config import DataConfig, SchedulerConfig
from yolo.training.optim import lerp
from yolo.utils.logger import logger
from yolo.utils.module_utils import unwrap_model


class EMA(Callback):
    """Exponential Moving Average of model weights as a Lightning Callback.

    Keeps a shadow copy of model parameters smoothed over training steps:
        beta = decay * (1 - exp(-step / tau))
        shadow = beta * shadow + (1 - beta) * model

    The tau warmup ramps beta up from ~0 at step 0, so early noisy updates
    don't dominate the shadow. Validation always runs on shadow weights;
    training weights are swapped back immediately after.
    """

    _CHECKPOINT_KEY = "ema_shadow"

    def __init__(self, decay: float = 0.9999, tau: float = 2000.0) -> None:
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step: int = 0
        self.batch_count: int = 0
        self.shadow: Optional[dict] = None
        self._training_weights: Optional[dict] = None

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        """Initialise shadow from the model before training begins."""
        if self.shadow is None:
            model = unwrap_model(pl_module.model)
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _beta(self) -> float:
        """Effective smoothing coefficient at the current step."""
        return self.decay * (1 - exp(-self.step / self.tau))

    @torch.no_grad()
    def update(self, pl_module: "LightningModule") -> None:
        """Blend model parameters into shadow; copy buffers directly."""
        model = unwrap_model(pl_module.model)
        current = model.state_dict()
        if self.shadow is None:
            self.shadow = {k: v.detach().clone() for k, v in current.items()}
            return

        beta = self._beta()

        param_keys = [k for k, _ in model.named_parameters()]
        shadow_params = [self.shadow[k] for k in param_keys]
        model_params = [current[k].detach().to(self.shadow[k].device) for k in param_keys]
        if hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(shadow_params, model_params, 1.0 - beta)
        elif hasattr(torch, "_foreach_mul_"):
            torch._foreach_mul_(shadow_params, beta)
            torch._foreach_add_(shadow_params, model_params, alpha=1.0 - beta)
        else:
            for s, m in zip(shadow_params, model_params):
                s.mul_(beta).add_(m, alpha=1.0 - beta)

        for key, buf in model.named_buffers():
            self.shadow[key].copy_(buf.detach().to(self.shadow[key].device))

    @torch.no_grad()
    def apply_shadow(self, pl_module: "LightningModule") -> None:
        """Snapshot training weights then load shadow weights into the model."""
        if self.shadow is None:
            return
        model = unwrap_model(pl_module.model)
        self._training_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, pl_module: "LightningModule") -> None:
        """Reload the training-weight snapshot, discarding the shadow swap."""
        if self._training_weights is None:
            return
        model = unwrap_model(pl_module.model)
        model.load_state_dict(self._training_weights, strict=True)
        self._training_weights = None

    @torch.no_grad()
    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.batch_count += 1
        if self.batch_count % trainer.accumulate_grad_batches != 0:
            return
        self.step += 1
        self.update(pl_module)

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.batch_count = 0
        self.apply_shadow(pl_module)

    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.restore(pl_module)

    def on_save_checkpoint(self, trainer: "Trainer", pl_module: "LightningModule", checkpoint: dict) -> None:
        if self.shadow is None:
            return
        checkpoint[self._CHECKPOINT_KEY] = {k: v.detach().cpu() for k, v in self.shadow.items()}
        checkpoint["ema_step"] = self.step
        checkpoint["ema_batch_count"] = self.batch_count

    def on_load_checkpoint(self, trainer: "Trainer", pl_module: "LightningModule", checkpoint: dict) -> None:
        self.step = checkpoint.get("ema_step", 0)
        self.batch_count = checkpoint.get("ema_batch_count", 0)
        if self._CHECKPOINT_KEY not in checkpoint:
            return
        model = unwrap_model(pl_module.model)
        target_device = next(model.parameters()).device
        self.shadow = {k: v.detach().clone().to(target_device) for k, v in checkpoint[self._CHECKPOINT_KEY].items()}


class GradientAccumulation(Callback):
    """Dynamic Gradient Accumulation callback.

    Adjusts the number of accumulation steps during the warmup phase to
    simulate a larger 'equivalent' batch size, which improves stability.
    """

    def __init__(self, data_cfg: DataConfig, scheduler_cfg: SchedulerConfig):
        """Initializes the GradientAccumulation callback.

        Args:
            data_cfg (DataConfig): Dataset configuration (batch sizes).
            scheduler_cfg (SchedulerConfig): Scheduler configuration (warmup epochs).
        """
        super().__init__()
        self.equivalent_batch_size = data_cfg.equivalent_batch_size
        self.actual_batch_size = data_cfg.batch_size
        self.warmup_epochs = getattr(scheduler_cfg.warmup, "epochs", 0)
        self.max_accumulation = 1
        self.warmup_batches = 0
        self.steps_per_epoch = 1
        logger.info(":arrows_counterclockwise: Enable Gradient Accumulation")

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        scaled_batch = self.actual_batch_size * trainer.world_size
        self.max_accumulation = max(1, round(self.equivalent_batch_size / scaled_batch))

    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for sched_cfg in trainer.lr_scheduler_configs:
            if hasattr(sched_cfg.scheduler, "steps_per_epoch"):
                self.steps_per_epoch = sched_cfg.scheduler.steps_per_epoch
                break
        self.warmup_batches = int(self.warmup_epochs * self.steps_per_epoch)

    def on_train_batch_start(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        step = trainer.global_step
        if step < self.warmup_batches:
            current_accumulation = round(lerp(1, self.max_accumulation, step, self.warmup_batches))
        else:
            current_accumulation = self.max_accumulation
        trainer.accumulate_grad_batches = current_accumulation
