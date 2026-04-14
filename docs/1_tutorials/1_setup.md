# Setup Config

To set up your configuration, generate a config class based on `Config` using [Hydra](https://hydra.cc/). The configuration includes general settings, `dataset` information, and task-specific info (`train`, `inference`, `validation`).

`YOLORichProgressBar` provides a [rich](https://github.com/Textualize/rich)-based progress bar and logging callback for PyTorch Lightning. It is the standard way to display training progress. Alongside it, `setup()` from `yolo.utils.logging_utils` returns the full list of callbacks, loggers, and the output save path derived from your config.

=== "decorator"
    ```python
    import hydra
    from yolo import YOLORichProgressBar
    from yolo.config.config import Config
    from yolo.utils.logging_utils import setup

    @hydra.main(config_path="config", config_name="config", version_base=None)
    def main(cfg: Config):
        callbacks, loggers, save_path = setup(cfg, exp_name=cfg.name)
    ```

=== "initialize & compose"
    ```python
    from hydra import compose, initialize
    from yolo import YOLORichProgressBar
    from yolo.config.config import Config
    from yolo.utils.logging_utils import setup

    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="config", overrides=["task=train", "model=v9-c"])

    callbacks, loggers, save_path = setup(cfg, exp_name=cfg.name)
    ```
