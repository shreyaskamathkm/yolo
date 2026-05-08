import logging

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.logging import RichHandler

# Configure the base logger for the package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

if rank_zero_only.rank == 0 and not logger.hasHandlers():
    logger.addHandler(RichHandler(show_path=True, show_time=False, markup=True))

from yolo.config.config import Config, NMSConfig
from yolo.data.loader import Compose, create_dataloader
from yolo.deploy import ModelExporter, create_inference_backend
from yolo.model.builder import create_model
from yolo.tasks.detection.postprocess import (
    Anc2Box,
    Vec2Box,
    bbox_nms,
    create_converter,
)
from yolo.tasks.detection.solver import DetectionTrainModel as TrainModel
from yolo.utils.drawer import draw_bboxes
from yolo.utils.logging_utils import (
    ImageLogger,
    YOLORichModelSummary,
    YOLORichProgressBar,
)
from yolo.utils.model_utils import PostProcess

__all__ = [
    "logger",
    "create_model",
    "Config",
    "YOLORichProgressBar",
    "NMSConfig",
    "YOLORichModelSummary",
    "draw_bboxes",
    "Vec2Box",
    "Anc2Box",
    "bbox_nms",
    "create_converter",
    "Compose",
    "ImageLogger",
    "create_dataloader",
    "create_inference_backend",
    "ModelExporter",
    "TrainModel",
    "PostProcess",
]
