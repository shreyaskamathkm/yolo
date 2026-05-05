from pathlib import Path

from torch.utils.data import DataLoader

from yolo.config.config import Config
from yolo.data.loader import StreamDataLoader, create_dataloader
from yolo.data.schema import DataSplitType, TrainerTaskType


def test_create_dataloader_cache(train_cfg: Config):
    train_cfg.task.data.shuffle = False
    train_cfg.task.data.batch_size = 2

    cache_file = Path("tests/data/train.cache")
    cache_file.unlink(missing_ok=True)

    make_cache_loader = create_dataloader(train_cfg.task.data, train_cfg.dataset, split=DataSplitType.TRAIN)
    load_cache_loader = create_dataloader(train_cfg.task.data, train_cfg.dataset, split=DataSplitType.TRAIN)
    m_batch = next(iter(make_cache_loader))
    l_batch = next(iter(load_cache_loader))
    assert m_batch.batch_size == l_batch.batch_size
    assert m_batch.images.shape == l_batch.images.shape
    assert m_batch.reverse_transforms.shape == l_batch.reverse_transforms.shape
    assert m_batch.paths == l_batch.paths


def test_training_data_loader_correctness(train_dataloader: DataLoader):
    """Test that the training data loader produces correctly shaped data and metadata."""
    batch = next(iter(train_dataloader))
    assert batch.batch_size == 2
    assert batch.images.shape == (2, 3, 640, 640)
    assert batch.reverse_transforms.shape == (2, 5)
    all_train_images = {p.resolve() for p in Path("tests/data/images/train").glob("*.jpg")}
    batch_paths = {Path(p).resolve() for p in batch.paths}
    assert batch_paths.issubset(all_train_images), f"Batch paths {batch_paths} not in {all_train_images}"


def test_validation_data_loader_correctness(validation_dataloader: DataLoader):
    batch = next(iter(validation_dataloader))
    assert batch.batch_size == 5
    assert batch.images.shape == (5, 3, 640, 640)
    assert batch.targets.shape == (5, 18, 5)
    assert batch.reverse_transforms.shape == (5, 5)
    expected_paths = [
        Path("tests/data/images/val/000000151480.jpg").resolve(),
        Path("tests/data/images/val/000000284106.jpg").resolve(),
        Path("tests/data/images/val/000000323571.jpg").resolve(),
        Path("tests/data/images/val/000000556498.jpg").resolve(),
        Path("tests/data/images/val/000000570456.jpg").resolve(),
    ]
    assert sorted([Path(p).resolve() for p in batch.paths]) == sorted(expected_paths)


def test_file_stream_data_loader_frame(file_stream_data_loader: StreamDataLoader):
    """Test the frame output from the file stream data loader."""
    frame, rev_tensor, origin_frame, path = next(iter(file_stream_data_loader))
    assert frame.shape == (1, 3, 640, 640)
    assert rev_tensor.shape == (1, 5)
    assert origin_frame.size == (480, 640)


def test_directory_stream_data_loader_frame(directory_stream_data_loader: StreamDataLoader):
    """Test the frame output from the directory stream data loader."""
    frame, rev_tensor, origin_frame, path = next(iter(directory_stream_data_loader))
    assert frame.shape == (1, 3, 640, 640)
    assert rev_tensor.shape == (1, 5)
    assert origin_frame.size != (640, 640)
