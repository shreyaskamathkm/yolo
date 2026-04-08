import os
from math import exp
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import ListConfig
from torch import Tensor, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

from yolo.config.config import (
    IDX_TO_ID,
    DataConfig,
    NMSConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from yolo.model.yolo import YOLO
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box, bbox_nms, transform_bbox
from yolo.utils.logger import logger


def lerp(start: float, end: float, step: Union[int, float], total: int = 1):
    """
    Linearly interpolates between start and end values.

    start * (1 - step) + end * step

    Parameters:
        start (float): The starting value.
        end (float): The ending value.
        step (int): The current step in the interpolation process.
        total (int): The total number of steps.

    Returns:
        float: The interpolated value.
    """
    if total <= 0:
        return end
    return start + (end - start) * step / total


class EMA(Callback):
    """Exponential Moving Average of model weights as a Lightning Callback.

    Keeps a shadow copy of model parameters smoothed over training steps:
        beta = decay * (1 - exp(-step / tau))
        shadow = beta * shadow + (1 - beta) * model

    The tau warmup ramps beta up from ~0 at step 0, so early noisy updates
    don't dominate the shadow. Validation always runs on shadow weights;
    training weights are swapped back immediately after.
    """

    _CHECKPOINT_KEY = "ema_shadow"

    def __init__(self, decay: float = 0.9999, tau: float = 2000.0) -> None:
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step: int = 0
        self.batch_count: int = 0
        self.shadow: Optional[dict] = None
        self._training_weights: Optional[dict] = None

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        """Initialise shadow from the model before training begins.

        Running setup() here (rather than lazily inside update) means lerping
        starts from step 1 instead of step 2, matching the original behaviour.
        The guard keeps it idempotent if setup is called multiple times.
        """
        if self.shadow is None:
            self.shadow = {k: v.detach().clone() for k, v in pl_module.model.state_dict().items()}

    def _beta(self) -> float:
        """Effective smoothing coefficient at the current step."""
        return self.decay * (1 - exp(-self.step / self.tau))

    @no_grad()
    def update(self, pl_module: "LightningModule") -> None:
        """Blend model parameters into shadow; copy buffers directly.

        Parameters (learned weights) are blended via EMA.
        Buffers (e.g. BatchNorm running_mean / running_var) track data
        statistics — averaging them across two different distributions
        produces a value that represents neither, so they are copied as-is.
        """
        current = pl_module.model.state_dict()
        if self.shadow is None:
            # Safety fallback: setup() should have run, but guard anyway.
            self.shadow = {k: v.detach().clone() for k, v in current.items()}
            return

        beta = self._beta()

        # Parameters: fused lerp — shadow = beta*shadow + (1-beta)*model
        param_keys = [k for k, _ in pl_module.model.named_parameters()]
        shadow_params = [self.shadow[k] for k in param_keys]
        model_params = [current[k].detach().to(self.shadow[k].device) for k in param_keys]
        if hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(shadow_params, model_params, 1.0 - beta)
        elif hasattr(torch, "_foreach_mul_"):
            torch._foreach_mul_(shadow_params, beta)
            torch._foreach_add_(shadow_params, model_params, alpha=1.0 - beta)
        else:
            for s, m in zip(shadow_params, model_params):
                s.mul_(beta).add_(m, alpha=1.0 - beta)

        # Buffers: copy directly
        for key, buf in pl_module.model.named_buffers():
            self.shadow[key].copy_(buf.detach().to(self.shadow[key].device))

    @no_grad()
    def apply_shadow(self, pl_module: "LightningModule") -> None:
        """Snapshot training weights then load shadow weights into the model."""
        if self.shadow is None:
            return
        self._training_weights = {k: v.detach().clone() for k, v in pl_module.model.state_dict().items()}
        pl_module.model.load_state_dict(self.shadow, strict=True)

    @no_grad()
    def restore(self, pl_module: "LightningModule") -> None:
        """Reload the training-weight snapshot, discarding the shadow swap."""
        if self._training_weights is None:
            return
        pl_module.model.load_state_dict(self._training_weights, strict=True)
        self._training_weights = None

    @no_grad()
    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.batch_count += 1
        if self.batch_count % trainer.accumulate_grad_batches != 0:
            return
        self.step += 1
        self.update(pl_module)

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.batch_count = 0
        self.apply_shadow(pl_module)

    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.restore(pl_module)

    def on_save_checkpoint(self, trainer: "Trainer", pl_module: "LightningModule", checkpoint: dict) -> None:
        if self.shadow is None:
            return
        checkpoint[self._CHECKPOINT_KEY] = {k: v.detach().cpu() for k, v in self.shadow.items()}
        checkpoint["ema_step"] = self.step
        checkpoint["ema_batch_count"] = self.batch_count

    def on_load_checkpoint(self, trainer: "Trainer", pl_module: "LightningModule", checkpoint: dict) -> None:
        self.step = checkpoint.get("ema_step", 0)
        self.batch_count = checkpoint.get("ema_batch_count", 0)
        if self._CHECKPOINT_KEY not in checkpoint:
            return
        target_device = next(pl_module.model.parameters()).device
        self.shadow = {k: v.detach().clone().to(target_device) for k, v in checkpoint[self._CHECKPOINT_KEY].items()}


class GradientAccumulation(Callback):
    def __init__(self, data_cfg: DataConfig, scheduler_cfg: SchedulerConfig):
        super().__init__()
        self.equivalent_batch_size = data_cfg.equivalent_batch_size
        self.actual_batch_size = data_cfg.batch_size
        self.warmup_epochs = getattr(scheduler_cfg.warmup, "epochs", 0)
        self.current_batch = 0
        self.max_accumulation = 1
        self.warmup_batches = 0
        logger.info(":arrows_counterclockwise: Enable Gradient Accumulation")

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        effective_batch_size = self.actual_batch_size * trainer.world_size
        self.max_accumulation = max(1, round(self.equivalent_batch_size / effective_batch_size))
        batches_per_epoch = int(len(pl_module.train_loader) / trainer.world_size)
        self.warmup_batches = int(self.warmup_epochs * batches_per_epoch)

    def on_train_epoch_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.current_batch = trainer.global_step

    def on_train_batch_start(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        if self.current_batch < self.warmup_batches:
            current_accumulation = round(lerp(1, self.max_accumulation, self.current_batch, self.warmup_batches))
        else:
            current_accumulation = self.max_accumulation
        trainer.accumulate_grad_batches = current_accumulation

    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.current_batch += 1


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "momentum": 0.937, "weight_decay": 0},
        {"params": conv_params, "momentum": 0.937},
        {"params": norm_params, "momentum": 0.937, "weight_decay": 0},
    ]

    def next_epoch(self, batch_num, epoch_idx):
        self.min_lr = self.max_lr
        self.max_lr = [param["lr"] for param in self.param_groups]
        # TODO: load momentum from config instead a fix number
        #       0.937: Start Momentum
        #       0.8  : Normal Momemtum
        #       3    : The warm up epoch num
        self.min_mom = lerp(0.8, 0.937, min(epoch_idx, 3), 3)
        self.max_mom = lerp(0.8, 0.937, min(epoch_idx + 1, 3), 3)
        self.batch_num = batch_num
        self.batch_idx = 0

    def next_batch(self):
        self.batch_idx += 1
        lr_dict = dict()
        for lr_idx, param_group in enumerate(self.param_groups):
            min_lr, max_lr = self.min_lr[lr_idx], self.max_lr[lr_idx]
            param_group["lr"] = lerp(min_lr, max_lr, self.batch_idx, self.batch_num)
            param_group["momentum"] = lerp(self.min_mom, self.max_mom, self.batch_idx, self.batch_num)
            lr_dict[f"LR/{lr_idx}"] = param_group["lr"]
            lr_dict[f"momentum/{lr_idx}"] = param_group["momentum"]
        return lr_dict

    optimizer_class.next_batch = next_batch
    optimizer_class.next_epoch = next_epoch

    optimizer = optimizer_class(model_parameters, **optim_cfg.args)
    optimizer.max_lr = [0.1, 0, 0]
    return optimizer


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: (epoch + 1) / wepoch if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * ((epoch + 1) / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda2, lambda1, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[wepoch - 1])
    return schedule


def initialize_distributed() -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"🔢 Initialized process group; rank: {rank}, size: {world_size}")
    return local_rank


def get_device(device_spec: Union[str, int, List[int]]) -> torch.device:
    ddp_flag = False
    if isinstance(device_spec, (list, ListConfig)):
        ddp_flag = True
        device_spec = initialize_distributed()
    if torch.cuda.is_available() and "cuda" in str(device_spec):
        return torch.device(device_spec), ddp_flag
    if not torch.cuda.is_available():
        if device_spec != "cpu":
            logger.warning(f"❎ Device spec: {device_spec} not support, Choosing CPU instead")
        return torch.device("cpu"), False

    device = torch.device(device_spec)
    return device, ddp_flag


class PostProcess:
    """
    TODO: function document
    scale back the prediction and do nms for pred_bbox
    """

    def __init__(self, converter: Union[Vec2Box, Anc2Box], nms_cfg: NMSConfig) -> None:
        self.converter = converter
        self.nms = nms_cfg

    def __call__(
        self, predict, rev_tensor: Optional[Tensor] = None, image_size: Optional[List[int]] = None
    ) -> List[Tensor]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict["Main"])
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None
        if rev_tensor is not None:
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
        pred_bbox = bbox_nms(pred_class, pred_bbox, self.nms, pred_conf)
        return pred_bbox


def collect_prediction(predict_json: List, local_rank: int) -> List:
    """
    Collects predictions from all distributed processes and gathers them on the main process (rank 0).

    Args:
        predict_json (List): The prediction data (can be of any type) generated by the current process.
        local_rank (int): The rank of the current process. Typically, rank 0 is the main process.

    Returns:
        List: The combined list of predictions from all processes if on rank 0, otherwise predict_json.
    """
    if dist.is_initialized() and local_rank == 0:
        all_predictions = [None for _ in range(dist.get_world_size())]
        dist.gather_object(predict_json, all_predictions, dst=0)
        predict_json = [item for sublist in all_predictions for item in sublist]
    elif dist.is_initialized():
        dist.gather_object(predict_json, None, dst=0)
    return predict_json


def predicts_to_json(img_paths, predicts, rev_tensor):
    """
    TODO: function document
    turn a batch of imagepath and predicts(n x 6 for each image) to a List of diction(Detection output)
    """
    batch_json = []
    for img_path, bboxes, box_reverse in zip(img_paths, predicts, rev_tensor):
        scale, shift = box_reverse.split([1, 4])
        bboxes = bboxes.clone()
        bboxes[:, 1:5] = (bboxes[:, 1:5] - shift[None]) / scale[None]
        bboxes[:, 1:5] = transform_bbox(bboxes[:, 1:5], "xyxy -> xywh")
        for cls, *pos, conf in bboxes:
            bbox = {
                "image_id": int(Path(img_path).stem),
                "category_id": IDX_TO_ID[int(cls)],
                "bbox": [float(p) for p in pos],
                "score": float(conf),
            }
            batch_json.append(bbox)
    return batch_json
