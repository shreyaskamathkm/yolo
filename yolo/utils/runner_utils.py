import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from yolo.config.config import Config
from yolo.training.callbacks import EMA, GradientAccumulation
from yolo.utils.logging_utils import (
    ImageLogger,
    YOLORichModelSummary,
    YOLORichProgressBar,
)


def set_seed(seed: int):
    """
    Ensure reproducible training by setting random seeds.
    """
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_callbacks(cfg: Config):
    """
    Construct all PyTorch Lightning Callbacks for training and progress monitoring.
    """
    quiet = getattr(cfg, "quiet", False)
    callbacks = []

    # Training logic callbacks
    if cfg.task.task == "train" and hasattr(cfg.task.data, "equivalent_batch_size"):
        callbacks.append(GradientAccumulation(data_cfg=cfg.task.data, scheduler_cfg=cfg.task.scheduler))

    if hasattr(cfg.task, "ema") and getattr(cfg.task.ema, "enable", False):
        callbacks.append(EMA(decay=cfg.task.ema.decay))

    if quiet:
        return callbacks

    # Progress and visual logging callbacks
    callbacks.append(YOLORichProgressBar())
    callbacks.append(YOLORichModelSummary())
    callbacks.append(ImageLogger())

    if (getattr(cfg, "use_tensorboard", False) or getattr(cfg, "use_wandb", False)) and cfg.task.task == "train":
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if cfg.task.task == "train":
        save_top_k = -1 if getattr(cfg.task, "save_all_checkpoints", False) else 2
        callbacks.append(
            ModelCheckpoint(
                save_last=True,
                save_top_k=save_top_k,
                monitor="map",
                mode="max",
                filename="epoch_{epoch:02d}_map_{map:.2f}",
                auto_insert_metric_name=False,
                save_weights_only=False,
            )
        )

    return callbacks
