import pytest
import torch
from omegaconf import OmegaConf

from yolo.tasks.detection.head import TOODHead
from yolo.tasks.detection.loss import TaskAlignedFocalLoss, TOODLoss
from yolo.tasks.detection.postprocess import TaskAlignedMatcher


def test_tood_head():
    in_channels = (128, 256)
    num_classes = 80
    head = TOODHead(in_channels, num_classes, reg_max=16)
    x = torch.randn(1, 256, 32, 32)
    cls_score, anchor_x, vector_x = head(x)
    assert cls_score.shape == (1, num_classes, 32, 32)
    assert anchor_x.shape == (1, 16, 4, 32, 32)
    assert vector_x.shape == (1, 4, 32, 32)


def test_task_aligned_focal_loss():
    gamma = 1.0
    loss_fn = TaskAlignedFocalLoss(gamma=gamma)
    predicts_cls = torch.randn(2, 10, 80)
    targets_cls = torch.rand(2, 10, 80)
    cls_norm = torch.tensor(1.0)
    loss = loss_fn(predicts_cls, targets_cls, cls_norm)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_task_aligned_matcher_instantiation():
    matcher_cfg = OmegaConf.create({"iou": "ciou", "topk": 10, "factor": {"iou": 6, "cls": 1}})

    class MockVec2Box:
        def __init__(self):
            self.anchor_grid = torch.randn(100, 2)
            self.scaler = torch.ones(100)

    vec2box = MockVec2Box()
    matcher = TaskAlignedMatcher(matcher_cfg, class_num=80, vec2box=vec2box, reg_max=16)
    assert matcher.alpha == 1.0
    assert matcher.beta == 6.0


def test_tood_loss_instantiation():
    matcher_cfg = OmegaConf.create({"iou": "ciou", "topk": 10, "factor": {"iou": 6, "cls": 1}})
    loss_cfg = OmegaConf.create(
        {
            "objective": {"BoxLoss": 1, "DFLoss": 1, "BCELoss": 1},
            "aux": False,
            "matcher": matcher_cfg,
            "type": "TOOD",
            "gamma": 1.0,
        }
    )

    class MockVec2Box:
        def __init__(self):
            self.anchor_grid = torch.randn(100, 2)
            self.scaler = torch.ones(100)

    vec2box = MockVec2Box()
    tood_loss = TOODLoss(loss_cfg, vec2box, class_num=80, reg_max=16)
    assert isinstance(tood_loss.cls, TaskAlignedFocalLoss)
    assert isinstance(tood_loss.matcher, TaskAlignedMatcher)
