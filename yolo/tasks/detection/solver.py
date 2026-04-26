import time
from math import ceil
from pathlib import Path

import cv2
import filetype
import numpy as np
from lightning import LightningModule
from omegaconf import OmegaConf
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.data.loader import create_dataloader
from yolo.deploy import create_inference_backend
from yolo.model.builder import create_model
from yolo.tasks.detection.loss import create_loss_function
from yolo.tasks.detection.postprocess import create_converter, to_metrics_format
from yolo.tasks.registry import register
from yolo.training.optim import create_optimizer, create_scheduler
from yolo.utils.drawer import draw_bboxes
from yolo.utils.model_utils import PostProcess
from yolo.utils.module_utils import unwrap_model


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(self, x):
        return self.model(x)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Strip _orig_mod prefix from state_dict when saving."""
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Add _orig_mod prefix to state_dict when loading if model is compiled."""
        if hasattr(self.model, "_orig_mod"):
            state_dict = checkpoint["state_dict"]
            checkpoint["state_dict"] = {
                k.replace("model.", "model._orig_mod.") if k.startswith("model.") and "_orig_mod." not in k else k: v
                for k, v in state_dict.items()
            }


@register("detection", "validation")
class DetectionValidateModel(BaseModel):
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
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        epoch_metrics.pop("classes", None)
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True, logger=True)
        self.metric.reset()


@register("detection", "train")
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
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            logger=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)

        batch_size = self.cfg.task.data.batch_size
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        equivalent_batch_size = getattr(self.cfg.task.data, "equivalent_batch_size", None)
        if equivalent_batch_size is not None:
            max_accum = max(1, round(equivalent_batch_size / (batch_size * world_size)))
        else:
            max_accum = 1

        # Use dataset length — invariant to loader sharding (e.g. Ray Train or Distributed Sampler
        # wraps the loader per rank, so len(train_loader) would be the per-rank count).
        if hasattr(self.train_loader, "dataset"):
            n_samples = len(self.train_loader.dataset)
            global_batch = batch_size * world_size * max_accum
            drop_last = getattr(self.cfg.task.data, "drop_last", False)
            if drop_last:
                steps_per_epoch = max(1, n_samples // global_batch)
            else:
                steps_per_epoch = max(1, ceil(n_samples / global_batch))
        else:
            steps_per_epoch = max(1, ceil(len(self.train_loader) / max_accum))

        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler, steps_per_epoch, self.cfg.task.epoch)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


@register("detection", "inference")
class DetectionInferenceModel(LightningModule):
    """LightningModule for YOLO detection inference.

    Handles high-performance inference using various backends, real-time
    preview of results, and saving of visualized outputs (images/videos).
    """

    def __init__(self, cfg: Config):
        """Initializes the inference solver.

        Args:
            cfg (Config): System configuration.
        """

        super().__init__()
        self.cfg = cfg
        self.model = create_inference_backend(cfg.task.backend, self.cfg.weight, str(self.device), self.cfg)
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)
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
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
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
            print("🎥 Video saved successfully.")

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
                print(f"💾 Saved visualize image at {save_path}")
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
                print(f"💾 Saved visualize image at {save_path}")

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
            print(f"🎥 Initialized video writer: {save_path} ({w}x{h} @ {fps} FPS)")

        self.video_writer.write(img_bgr)
