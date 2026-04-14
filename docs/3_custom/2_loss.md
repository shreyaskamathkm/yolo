# Loss Function

The detection loss lives in `yolo/tasks/detection/loss.py`. It is assembled from three composable components and wrapped in a dual-loss strategy for auxiliary and main heads.

## Components

| Class | Description |
|---|---|
| `BCELoss` | Binary cross-entropy on class predictions, normalized by a per-batch class factor |
| `BoxLoss` | IoU-based box regression loss (CIoU by default) |
| `DFLoss` | Distribution Focal Loss on the regression distribution predicted by DFL heads |
| `YOLOLoss` | Combines `BCELoss` + `BoxLoss` + `DFLoss` for a single prediction head. Handles anchor separation and target assignment internally |
| `DualLoss` | Runs `YOLOLoss` for both the auxiliary and main prediction heads, weighting the auxiliary loss at 0.25× |

The entry point used by the solver is:

```python
from yolo.tasks.detection.loss import create_loss_function

loss_fn = create_loss_function(cfg, vec2box)
```

`create_loss_function` builds a `DualLoss` configured from `cfg.task.loss` (`LossConfig`).

## Loss Config

Loss weights are controlled via `yolo/config/task/train.yaml` under the `loss` key:

```yaml
loss:
  box: 7.5
  cls: 0.5
  dfl: 1.5
```

Override from the CLI:

```bash
python -m yolo task=train task.loss.box=10.0
```

## Target Assignment

`YOLOLoss` uses `BoxMatcher` (from `yolo/tasks/detection/postprocess.py`) to assign ground-truth boxes to anchors via a Task-Aligned Assigner (TAL). Loss is computed only on matched anchors.
