import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize
from lightning import LightningModule, Trainer
from torch.utils.data import Dataset

from yolo import Anc2Box, Config, Vec2Box, create_converter, create_model
from yolo.data.loader import StreamDataLoader, create_dataloader
from yolo.data.preparation import prepare_dataset
from yolo.model.builder import YOLO
from yolo.utils.logging_utils import set_seed, setup


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: mark test to run only if CUDA is available")


def get_cfg(overrides=[]) -> Config:
    config_path = "."
    with initialize(config_path=config_path, version_base=None):
        cfg: Config = compose(config_name="test", overrides=overrides)
        set_seed(cfg.lucky_number)
        return cfg


@pytest.fixture(scope="session")
def train_cfg() -> Config:
    return get_cfg(overrides=["task=train", "dataset=mock"])


@pytest.fixture(scope="session")
def validation_cfg():
    return get_cfg(overrides=["task=validation", "dataset=mock"])


@pytest.fixture(scope="session")
def inference_cfg():
    return get_cfg(overrides=["task=inference"])


@pytest.fixture(scope="session")
def inference_v7_cfg():
    return get_cfg(overrides=["task=inference", "model=v7"])


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model(train_cfg: Config, device) -> YOLO:
    model = create_model(train_cfg.model)
    return model.to(device)


@pytest.fixture(scope="session")
def model_v7(inference_v7_cfg: Config, device) -> YOLO:
    model = create_model(inference_v7_cfg.model)
    return model.to(device)


@pytest.fixture(scope="session")
def solver(train_cfg: Config) -> Trainer:
    train_cfg.use_wandb = False
    del train_cfg.task.data.equivalent_batch_size
    callbacks, loggers, save_path = setup(train_cfg)
    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(train_cfg.task, "epoch", None),
        precision="32-true",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        deterministic=True,
        default_root_dir=save_path,
    )
    return trainer


@pytest.fixture(scope="session")
def vec2box(train_cfg: Config, model: YOLO, device) -> Vec2Box:
    vec2box = create_converter(
        train_cfg.model.name,
        model,
        train_cfg.model.anchor,
        train_cfg.image_size,
        device,
        class_num=train_cfg.dataset.class_num,
    )
    return vec2box


@pytest.fixture(scope="session")
def anc2box(inference_v7_cfg: Config, model: YOLO, device) -> Anc2Box:
    anc2box = create_converter(
        inference_v7_cfg.model.name,
        model,
        inference_v7_cfg.model.anchor,
        inference_v7_cfg.image_size,
        device,
        class_num=inference_v7_cfg.dataset.class_num,
    )
    return anc2box


@pytest.fixture(scope="session")
def train_dataloader(train_cfg: Config):
    prepare_dataset(train_cfg.dataset, task="train")
    return create_dataloader(train_cfg.task.data, train_cfg.dataset, train_cfg.task.task)


@pytest.fixture(scope="session")
def validation_dataloader(validation_cfg: Config):
    prepare_dataset(validation_cfg.dataset, task="val")
    return create_dataloader(validation_cfg.task.data, validation_cfg.dataset, validation_cfg.task.task)


@pytest.fixture(scope="session")
def file_stream_data_loader(inference_cfg: Config):
    inference_cfg.task.data.source = "tests/data/images/train/000000050725.jpg"
    return StreamDataLoader(inference_cfg.task.data)


@pytest.fixture(scope="session")
def file_stream_data_loader_v7(inference_v7_cfg: Config):
    inference_v7_cfg.task.data.source = "tests/data/images/train/000000050725.jpg"
    return StreamDataLoader(inference_v7_cfg.task.data)


@pytest.fixture(scope="session")
def directory_stream_data_loader(inference_cfg: Config):
    inference_cfg.task.data.source = "tests/data/images/train"
    return StreamDataLoader(inference_cfg.task.data)


# ── EMA test helpers ──────────────────────────────────────────────────────────


class TinyDataset(Dataset):
    """Minimal dataset returning (features, label) pairs for unit tests.

    When image_size is None the data is a 1-D feature vector of length
    feature_dim.  When image_size is a (H, W) tuple the data is a 3-channel
    image tensor so tests can exercise image-shaped inputs as well.
    """

    def __init__(self, num_samples: int = 8, image_size=None, feature_dim: int = 4, out_dim: int = 2):
        if image_size is not None:
            self.data = torch.randn(num_samples, 3, *image_size)
        else:
            self.data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, out_dim, (num_samples,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DummyModule(LightningModule):
    """Single-layer linear LightningModule used for testing callbacks."""

    def __init__(self, train_loader, input_size: int = 4, output_size: int = 2):
        super().__init__()
        self.model = nn.Linear(input_size, output_size)
        self._train_loader = train_loader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        target = F.one_hot(y, num_classes=pred.shape[-1]).float()
        return F.mse_loss(pred, target)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return self._train_loader


class DummyModuleWithVal(DummyModule):
    """DummyModule that also exposes a validation dataloader."""

    def __init__(self, train_loader, val_loader, input_size: int = 4, output_size: int = 2):
        super().__init__(train_loader, input_size=input_size, output_size=output_size)
        self._val_loader = val_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        target = F.one_hot(y, num_classes=pred.shape[-1]).float()
        return F.mse_loss(pred, target)

    def val_dataloader(self):
        return self._val_loader
