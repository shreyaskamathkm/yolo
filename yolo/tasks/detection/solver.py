import logging
import time
from pathlib import Path

import cv2
import filetype
import numpy as np
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.data.loader import create_dataloader
from yolo.deploy import create_inference_backend
from yolo.registry import SOLVERS
from yolo.schema import DataSplitType, TaskMode, TrainerTaskType
from yolo.tasks.base import BaseModule
from yolo.tasks.detection.loss import create_loss_function
from yolo.tasks.detection.postprocess import create_converter, to_metrics_format
from yolo.utils.drawer import draw_bboxes
from yolo.utils.model_utils import PostProcess

logger = logging.getLogger(__name__)


@SOLVERS.register_module(name=(TrainerTaskType.DETECTION, TaskMode.VAL))
class DetectionValidateModel(BaseModule):
    """LightningModule for YOLO detection validation.

    Handles metric calculation (mAP), data loading for validation,
    and post-processing of model predictions.
    """

    def __init__(self, cfg: Config):
        """Initializes the validation solver.

        Args:
            cfg (Config): System configuration.
        """

        super().__init__(cfg)
        self.cfg = cfg
        self.validation_cfg = getattr(cfg.task, "validation", cfg.task)
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(
            cfg.task.data, cfg.dataset, task=TrainerTaskType.DETECTION, split=DataSplitType.VAL
        )

    def setup(self, stage):
        logger.debug(f"Setting up Validation Model for stage: {stage}")
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        images, targets = batch.images, batch.targets
        H, W = images.shape[2:]
        raw_predicts = self.model(images)
        predicts = self.post_process(raw_predicts, image_size=[W, H])

        if hasattr(self, "loss_fn"):
            main_predicts = self.vec2box(raw_predicts["Main"])
            if "AUX" in raw_predicts and hasattr(self.loss_fn, "aux_rate"):
                aux_predicts = self.vec2box(raw_predicts["AUX"])
                val_loss, val_loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
            else:
                val_loss, val_loss_item = self.loss_fn(main_predicts, targets)

            self.log_dict(
                {f"Val_{k}": v for k, v in val_loss_item.items()},
                on_epoch=True,
                batch_size=batch.batch_size,
                sync_dist=True,
                rank_zero_only=True,
            )

        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        epoch_metrics.pop("classes", None)
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True, logger=True)
        self.metric.reset()


@SOLVERS.register_module(name=(TrainerTaskType.DETECTION, TaskMode.TRAIN))
class DetectionTrainModel(DetectionValidateModel):
    """LightningModule for YOLO detection training.

    Extends the validation model to include training loops, loss calculation,
    and optimizer/scheduler configuration.
    """

    def __init__(self, cfg: Config):
        """Initializes the training solver.

        Args:
            cfg (Config): System configuration.
        """

        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(
            self.cfg.task.data, self.cfg.dataset, task=TrainerTaskType.DETECTION, split=DataSplitType.TRAIN
        )

    def setup(self, stage):
        super().setup(stage)
        logger.debug("Initializing loss function")
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        images, targets = batch.images, batch.targets
        predicts = self(images)
        main_predicts = self.vec2box(predicts["Main"])
        if "AUX" in predicts and hasattr(self.loss_fn, "aux_rate"):
            aux_predicts = self.vec2box(predicts["AUX"])
            loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        else:
            loss, loss_item = self.loss_fn(main_predicts, targets)

        self.log_dict(
            loss_item,
            logger=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch.batch_size,
            rank_zero_only=True,
        )
        return loss


@SOLVERS.register_module(name=(TrainerTaskType.DETECTION, TaskMode.INFERENCE))
class DetectionInferenceModel(BaseModule):
    """LightningModule for YOLO detection inference.

    Handles high-performance inference using various backends, real-time
    preview of results, and saving of visualized outputs (images/videos).
    """

    def __init__(self, cfg: Config):
        """Initializes the inference solver.

        Args:
            cfg (Config): System configuration.
        """

        super().__init__(cfg)
        self.cfg = cfg
        self.model = create_inference_backend(cfg.task.backend, self.cfg.weight, str(self.device), self.cfg)
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, task=TaskMode.INFERENCE)
        self.last_time = time.time()
        self.video_writer = None
        self.current_video_path = None

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name,
            self.model,
            self.cfg.model.anchor,
            self.cfg.image_size,
            self.device,
            class_num=self.cfg.dataset.class_num,
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame, path = batch
        results = self(images)
        predicts = self.post_process(results, rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts[0], idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_result(img, batch_idx, path)
        return img, fps

    def on_predict_epoch_end(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            logger.info("🎥 Video saved successfully.")

    def _display_stream(self, img):
        curr_time = time.time()
        fps = 1 / (curr_time - self.last_time) if curr_time > self.last_time else 0.0
        self.last_time = curr_time
        return fps

    def _save_result(self, img, batch_idx, path=None):
        if isinstance(path, Path) and path.is_file():
            if filetype.is_image(path):
                # Save as individual image
                save_name = f"{path.name}"
                save_path = Path(self.trainer.default_root_dir) / save_name
                img.save(save_path)
                logger.info(f"💾 Saved visualize image at {save_path}")
            elif filetype.is_video(path):
                # Process as a video frame
                self._write_video_frame(img, path)
        else:
            # Fallback for live streams or unknown sources
            if getattr(self.predict_loader, "is_stream", None) or (path is not None and "stream" in str(path).lower()):
                self._write_video_frame(img, path)
            else:
                save_name = f"frame{batch_idx:03d}.png"
                save_path = Path(self.trainer.default_root_dir) / save_name
                img.save(save_path)
                logger.info(f"💾 Saved visualize image at {save_path}")

    def _write_video_frame(self, img, path):
        if path != self.current_video_path and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.current_video_path = path
        img_numpy = np.array(img)
        img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        if self.video_writer is None:
            save_name = f"{path.stem}_out.mp4" if isinstance(path, Path) else "stream_out.mp4"
            save_path = str(Path(self.trainer.default_root_dir) / save_name)
            fps = self.predict_loader.cap.get(cv2.CAP_PROP_FPS) if hasattr(self.predict_loader, "cap") else 30
            if fps <= 0:
                fps = 30
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            logger.info(f"🎥 Initialized video writer: {save_path} ({w}x{h} @ {fps} FPS)")

        self.video_writer.write(img_bgr)
