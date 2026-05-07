import os
import shutil
from pathlib import Path

from yolo.config.config import Config
from yolo.data.preparation import prepare_dataset, prepare_weight


def test_prepare_dataset(train_cfg: Config):
    dataset_path = Path("tests/data")
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    prepare_dataset(train_cfg.dataset, task="train")
    prepare_dataset(train_cfg.dataset, task="val")

    images_path = Path("tests/data/images")
    for data_type in images_path.iterdir():
        assert len(os.listdir(data_type)) == 5

    annotations_path = Path("tests/data/annotations")
    assert "instances_val.json" in os.listdir(annotations_path)
    assert "instances_train.json" in os.listdir(annotations_path)


def test_prepare_weight(tmp_path):
    # Use a real model name to avoid 404
    weight_path = tmp_path / "v9-t.pt"
    prepare_weight(weight_path=weight_path)
    assert weight_path.exists()
