import logging

import hydra
from lightning import Trainer
from omegaconf import OmegaConf

import yolo.tasks.detection.solver
from yolo.config.config import Config, resolve_config
from yolo.deploy import ModelExporter
from yolo.registry import SOLVERS, TRAINER_METHODS
from yolo.schema import TaskMode, TrainerTaskType
from yolo.utils.logging_utils import build_loggers
from yolo.utils.runner_utils import build_callbacks, set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    cfg = resolve_config(cfg)

    if hasattr(cfg, "seed"):
        set_seed(cfg.seed)

    if cfg.task.task == TaskMode.EXPORT:
        ModelExporter(cfg)()
        return

    loggers, save_path = build_loggers(cfg)
    callbacks = build_callbacks(cfg)

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.device,
        max_epochs=getattr(cfg.task, "epoch", None),
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        logger=loggers,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        deterministic=cfg.trainer.deterministic,
        fast_dev_run=cfg.trainer.fast_dev_run,
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    key = (cfg.task_type, cfg.task.task)
    if key not in SOLVERS:
        raise ValueError(f"No solver registered for task_type={cfg.task_type!r}, mode={cfg.task.task!r}")

    model = SOLVERS[key](cfg)
    task = getattr(trainer, TRAINER_METHODS[cfg.task.task])
    # Handle checkpoint resuming for training
    if cfg.task.task == TaskMode.TRAIN and hasattr(cfg.task, "resume") and cfg.task.resume:
        task(model, ckpt_path=cfg.task.resume)
    else:
        task(model)


if __name__ == "__main__":
    main()
