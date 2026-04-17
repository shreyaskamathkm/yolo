from math import ceil
from pathlib import Path

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.data.loader import create_dataloader
from yolo.model.builder import create_model
from yolo.tasks.detection.loss import create_loss_function
from yolo.tasks.detection.postprocess import create_converter, to_metrics_format
from yolo.tasks.registry import register
from yolo.training.optim import create_optimizer, create_scheduler
from yolo.utils.drawer import draw_bboxes
from yolo.utils.model_utils import PostProcess


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


@register("detection", "validation")
class DetectionValidateModel(BaseModel):
    def __init__(self, cfg: Config):
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
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True, logger=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
            logger=True,
        )
        self.metric.reset()


@register("detection", "train")
class DetectionTrainModel(DetectionValidateModel):
    def __init__(self, cfg: Config):
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
class DetectionInferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
