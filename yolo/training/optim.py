from typing import List, Optional, Type, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from yolo.config.config import OptimizerConfig, SchedulerConfig
from yolo.model.builder import YOLO


def lerp(start: float, end: float, step: Union[int, float], total: int = 1) -> float:
    """
    Linearly interpolates between start and end values.

    start * (1 - step) + end * step

    Parameters:
        start (float): The starting value.
        end (float): The ending value.
        step (int): The current step in the interpolation process.
        total (int): The total number of steps.

    Returns:
        float: The interpolated value.
    """
    if total <= 0:
        return end
    return start + (end - start) * step / total


class WarmupLRPolicy:
    """Base strategy for per-group LR shape during warmup.

    Subclass this to define custom warmup curves without touching
    ``WarmupBatchScheduler`` (Open/Closed Principle).
    """

    def start_lr(self, group_idx: int, initial_lr: float) -> float:
        """LR at virtual epoch -1 — where epoch 0 batch interpolation begins."""
        raise NotImplementedError

    def target_lr(self, epoch: int, group_idx: int, initial_lr: float) -> float:
        """LR target at the end of warmup ``epoch``."""
        raise NotImplementedError


class LinearWarmupPolicy(WarmupLRPolicy):
    """Uniform ramp: all param groups rise from 0 → initial_lr over warmup."""

    def __init__(self, warmup_epochs: int):
        self.warmup_epochs = int(warmup_epochs)

    def start_lr(self, group_idx: int, initial_lr: float) -> float:
        return 0.0

    def target_lr(self, epoch: int, group_idx: int, initial_lr: float) -> float:
        return lerp(0.0, initial_lr, epoch + 1, self.warmup_epochs)


class YOLOWarmupPolicy(WarmupLRPolicy):
    """YOLO-style warmup: bias group drops, all other groups rise.

    Group 0 (bias): starts at 10× initial_lr and ramps down to 1×.
    Groups 1+ (conv, bn): start at 0 and ramp up to 1×.

    This mirrors the original lambda2/lambda1 scheme from YOLO.
    """

    def __init__(self, warmup_epochs: int):
        self.warmup_epochs = int(warmup_epochs)

    def _lambda2(self, epoch: int) -> float:
        """Scale factor for bias group: 10 → 1 over warmup."""
        return 10 - 9 * ((epoch + 1) / self.warmup_epochs) if epoch < self.warmup_epochs else 1.0

    def _lambda1(self, epoch: int) -> float:
        """Scale factor for other groups: 0 → 1 over warmup."""
        return (epoch + 1) / self.warmup_epochs if epoch < self.warmup_epochs else 1.0

    def start_lr(self, group_idx: int, initial_lr: float) -> float:
        # Virtual epoch -1: lambda2(-1)=10, lambda1(-1)=0
        return (10.0 if group_idx == 0 else 0.0) * initial_lr

    def target_lr(self, epoch: int, group_idx: int, initial_lr: float) -> float:
        factor = self._lambda2(epoch) if group_idx == 0 else self._lambda1(epoch)
        return factor * initial_lr


class WarmupBatchScheduler(_LRScheduler):
    """Batch-level LR and momentum scheduler with epoch-aware warmup.

    Wraps an epoch-level scheduler and linearly interpolates the learning rate
    across batches within each epoch. It also handles the initial momentum
    warmup phase.

    Note:
        In Lightning, this should be used with `interval="step"`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        steps_per_epoch: int,
        warmup_epochs: int = 3,
        warmup_policy: Optional[WarmupLRPolicy] = None,
        start_momentum: float = 0.8,
        end_momentum: float = 0.937,
        last_epoch: int = -1,
    ):
        """Initializes the WarmupBatchScheduler.

        Args:
            optimizer (Optimizer): The wrapped optimizer.
            scheduler (_LRScheduler): The epoch-level scheduler (e.g., CosineAnnealingLR).
            steps_per_epoch (int): Total training batches in one epoch.
            warmup_epochs (int, optional): Number of warmup epochs. Defaults to 3.
            warmup_policy (Optional[WarmupLRPolicy], optional): Warmup curve strategy.
            start_momentum (float, optional): Initial momentum. Defaults to 0.8.
            end_momentum (float, optional): Target momentum after warmup. Defaults to 0.937.
            last_epoch (int, optional): The index of the last epoch. Defaults to -1.
        """

        self.scheduler = scheduler
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_policy = warmup_policy
        self.start_momentum = float(start_momentum)
        self.end_momentum = float(end_momentum)

        # Capture base LRs before any scheduler modifies them
        self._initial_lr: List[float] = [group["lr"] for group in optimizer.param_groups]

        # Epoch 0 interpolation endpoints — derived from policy or defaults
        if warmup_policy is not None:
            self._start_lr: List[float] = [warmup_policy.start_lr(i, lr) for i, lr in enumerate(self._initial_lr)]
            self._end_lr: List[float] = [warmup_policy.target_lr(0, i, lr) for i, lr in enumerate(self._initial_lr)]
        else:
            self._start_lr = [0.0] * len(self._initial_lr) if warmup_epochs > 0 else list(self._initial_lr)
            self._end_lr = list(self._initial_lr)

        self._epoch: int = 0
        super().__init__(optimizer, last_epoch)

    def _position(self) -> tuple:
        """Return (epoch, batch) for the current last_epoch value."""
        step = max(self.last_epoch, 0)
        return step // self.steps_per_epoch, step % self.steps_per_epoch

    def get_lr(self) -> List[float]:
        _, batch = self._position()
        return [lerp(s, e, batch + 1, self.steps_per_epoch) for s, e in zip(self._start_lr, self._end_lr)]

    def _set_lr_momentum(self, epoch: int, batch: int) -> None:
        # LR: interpolate within epoch
        for group, s, e in zip(self.optimizer.param_groups, self._start_lr, self._end_lr):
            group["lr"] = lerp(s, e, batch + 1, self.steps_per_epoch)

        # Momentum: linear interpolation over total warmup steps (global-step progress)
        if self.warmup_epochs > 0:
            warmup_steps = self.warmup_epochs * self.steps_per_epoch
            global_step = epoch * self.steps_per_epoch + batch
            progress = min(global_step + 1, warmup_steps) / warmup_steps
            momentum = self.start_momentum + (self.end_momentum - self.start_momentum) * progress
        else:
            momentum = self.end_momentum
        for group in self.optimizer.param_groups:
            if "momentum" in group:
                group["momentum"] = float(momentum)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _advance_epoch(self) -> None:
        """Roll the interpolation window forward by one epoch."""
        self._start_lr = list(self._end_lr)
        self._epoch += 1
        if self.warmup_policy is not None and self._epoch < self.warmup_epochs:
            self._end_lr = [self.warmup_policy.target_lr(self._epoch, i, lr) for i, lr in enumerate(self._initial_lr)]
        else:
            self.scheduler.step()
            self._end_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None) -> None:
        self.last_epoch = self.last_epoch + 1 if epoch is None else int(epoch)
        current_epoch, batch = self._position()

        while self._epoch < current_epoch:
            self._advance_epoch()

        self._set_lr_momentum(epoch=current_epoch, batch=batch)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["initial_lr"] = list(self._initial_lr)
        state["start_lr"] = list(self._start_lr)
        state["end_lr"] = list(self._end_lr)
        state["epoch"] = int(self._epoch)
        state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        self._initial_lr = state_dict.pop("initial_lr", self._initial_lr)
        self._start_lr = state_dict.pop("start_lr", self._start_lr)
        self._end_lr = state_dict.pop("end_lr", self._end_lr)
        self._epoch = state_dict.pop("epoch", self._epoch)
        inner = state_dict.pop("scheduler", None)
        if inner is not None:
            self.scheduler.load_state_dict(inner)
        super().load_state_dict(state_dict)
        epoch, batch = self._position()
        self._set_lr_momentum(epoch=epoch, batch=batch)


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Factory function to build the optimizer.

    Separates model parameters into groups (bias, normalization, convolution)
    to apply specific optimization settings (e.g., no weight decay on bias).

    Args:
        model (YOLO): The model to optimize.
        optim_cfg (OptimizerConfig): Optimizer configuration.

    Returns:
        Optimizer: The initialized PyTorch optimizer.
    """

    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "momentum": optim_cfg.args.momentum, "weight_decay": 0},
        {"params": conv_params, "momentum": optim_cfg.args.momentum},
        {"params": norm_params, "momentum": optim_cfg.args.momentum, "weight_decay": 0},
    ]

    optimizer = optimizer_class(model_parameters, **optim_cfg.args)
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    schedule_cfg: SchedulerConfig,
    steps_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> _LRScheduler:
    """Factory function to build the learning rate scheduler.

    Args:
        optimizer (Optimizer): The optimizer to wrap.
        schedule_cfg (SchedulerConfig): Scheduler and warmup configuration.
        steps_per_epoch (Optional[int], optional): Steps per epoch for batch-level scheduling.
        epochs (Optional[int], optional): Total training epochs.

    Returns:
        _LRScheduler: The initialized scheduler (wrapped in WarmupBatchScheduler).
    """

    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    epoch_sched = scheduler_class(optimizer, **schedule_cfg.args)

    warmup_policy = None
    warmup_epochs = 0
    if hasattr(schedule_cfg, "warmup"):
        warmup_epochs = int(schedule_cfg.warmup.epochs)
        warmup_policy = YOLOWarmupPolicy(warmup_epochs=warmup_epochs)

    start_momentum = getattr(schedule_cfg.warmup, "start_momentum", 0.8) if hasattr(schedule_cfg, "warmup") else 0.8
    end_momentum = getattr(schedule_cfg.warmup, "end_momentum", 0.937) if hasattr(schedule_cfg, "warmup") else 0.937

    return WarmupBatchScheduler(
        optimizer=optimizer,
        scheduler=epoch_sched,
        steps_per_epoch=steps_per_epoch or 1,
        warmup_epochs=warmup_epochs,
        warmup_policy=warmup_policy,
        start_momentum=start_momentum,
        end_momentum=end_momentum,
    )
