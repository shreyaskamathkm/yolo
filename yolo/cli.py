import hydra
from lightning import Trainer

import yolo.tasks.detection.solver  # register detection solvers
from yolo.config.config import Config
from yolo.tasks.registry import SOLVERS, TRAINER_METHODS
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    trainer_cfg = cfg.trainer
    trainer = Trainer(
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.device,
        max_epochs=getattr(cfg.task, "epoch", None),
        precision=trainer_cfg.precision,
        callbacks=callbacks,
        sync_batchnorm=trainer_cfg.sync_batchnorm,
        logger=loggers,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        gradient_clip_algorithm=trainer_cfg.gradient_clip_algorithm,
        deterministic=trainer_cfg.deterministic,
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    key = (cfg.task_type, cfg.task.task)
    if key not in SOLVERS:
        raise ValueError(f"No solver registered for task_type={cfg.task_type!r}, mode={cfg.task.task!r}")

    model = SOLVERS[key](cfg)
    getattr(trainer, TRAINER_METHODS[cfg.task.task])(model)


if __name__ == "__main__":
    main()
