import hydra
from lightning import Trainer

import yolo.tasks.detection.solver  # register detection solvers
from yolo.config.config import Config
from yolo.deploy import ModelExporter
from yolo.tasks.registry import SOLVERS, TRAINER_METHODS
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.task.task == "export":
        ModelExporter(cfg)()
        return

    callbacks, loggers, save_path = setup(cfg)

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
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    key = (cfg.task_type, cfg.task.task)
    if key not in SOLVERS:
        raise ValueError(f"No solver registered for task_type={cfg.task_type!r}, mode={cfg.task.task!r}")

    model = SOLVERS[key](cfg)
    task = getattr(trainer, TRAINER_METHODS[cfg.task.task])
    task(model)


if __name__ == "__main__":
    main()
