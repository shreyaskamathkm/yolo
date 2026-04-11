# Setup Config

To set up your configuration, generate a config class based on `Config` using [Hydra](https://hydra.cc/). The configuration includes general settings, `dataset` information, and task-specific info (`train`, `inference`, `validation`).

Next, create a `ProgressLogger` to handle output and the progress bar. This class is based on [rich](https://github.com/Textualize/rich)'s progress bar and customizes the logger via [loguru](https://loguru.readthedocs.io/).

=== "decorator"
    ```python
    import hydra
    from yolo import ProgressLogger
    from yolo.config.config import Config

    @hydra.main(config_path="config", config_name="config", version_base=None)
    def main(cfg: Config):
        progress = ProgressLogger(cfg, exp_name=cfg.name)
        pass
    ```

=== "initialize & compose"
    ```python
    from hydra import compose, initialize
    from yolo import ProgressLogger
    from yolo.config.config import Config

    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="config", overrides=["task=train", "model=v9-c"])

    progress = ProgressLogger(cfg, exp_name=cfg.name)
    ```
