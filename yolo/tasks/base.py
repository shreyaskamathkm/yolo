import logging
from math import ceil

from lightning import LightningModule
from omegaconf import OmegaConf

from yolo.config.config import Config
from yolo.model.builder import create_model
from yolo.training.optim import create_optimizer, create_scheduler
from yolo.utils.module_utils import clean_state_dict, restore_compile_prefix

logger = logging.getLogger(__name__)


class BaseModule(LightningModule):
    """Base LightningModule for YOLO tasks.

    Handles model initialization, state_dict cleaning for torch.compile,
    and shared optimizer/scheduler configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
        logger.debug(f"Initialized {self.__class__.__name__} with model: {getattr(cfg.model, 'name', 'unknown')}")
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(self, x):
        return self.model(x)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Strip torch.compile prefixes from state_dict when saving."""
        checkpoint["state_dict"] = clean_state_dict(checkpoint["state_dict"])

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Add _orig_mod prefix to state_dict when loading if model is compiled."""
        if hasattr(self.model, "_orig_mod"):
            checkpoint["state_dict"] = restore_compile_prefix(checkpoint["state_dict"])
        else:
            checkpoint["state_dict"] = clean_state_dict(checkpoint["state_dict"])

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        logger.info(f"Initialized optimizer: {optimizer.__class__.__name__}")

        batch_size = self.cfg.task.data.batch_size
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        equivalent_batch_size = getattr(self.cfg.task.data, "equivalent_batch_size", None)
        if equivalent_batch_size is not None:
            max_accum = max(1, round(equivalent_batch_size / (batch_size * world_size)))
        else:
            max_accum = 1

        train_loader = self.train_dataloader()

        # Use dataset length — invariant to loader sharding (e.g. Ray Train or Distributed Sampler
        # wraps the loader per rank, so len(train_loader) would be the per-rank count).
        if hasattr(train_loader, "dataset"):
            n_samples = len(train_loader.dataset)
            global_batch = batch_size * world_size * max_accum
            drop_last = getattr(self.cfg.task.data, "drop_last", False)
            if drop_last:
                steps_per_epoch = max(1, n_samples // global_batch)
            else:
                steps_per_epoch = max(1, ceil(n_samples / global_batch))
        else:
            steps_per_epoch = max(1, ceil(len(train_loader) / max_accum))

        # Fix: ensure steps_per_epoch is at least 1
        steps_per_epoch = max(1, steps_per_epoch)
        logger.info(f"Calculated steps_per_epoch: {steps_per_epoch} (max_accum: {max_accum}, world_size: {world_size})")

        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler, steps_per_epoch, self.cfg.task.epoch)
        logger.info(f"Initialized scheduler: {scheduler.__class__.__name__}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
