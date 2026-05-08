# Loss Function

The detection loss lives in `yolo/tasks/detection/loss.py`. It is assembled from three composable components and wrapped in a dual-loss strategy for auxiliary and main heads.

## Components

| Class | Description |
|---|---|
| `BCELoss` | Binary cross-entropy on class predictions, normalized by a per-batch class factor |
| `TaskAlignedFocalLoss` | TOOD-specific loss using $|t - s|^\gamma$ focal weighting to align tasks |
| `BoxLoss` | IoU-based box regression loss (CIoU by default) |
| `DFLoss` | Distribution Focal Loss on the regression distribution predicted by DFL heads |
| `YOLOLoss` | Standard YOLOv8/v9 loss. Combines `BCELoss` + `BoxLoss` + `DFLoss`. |
| `TOODLoss` | Task-aligned One-stage Object Detection loss. Uses `TaskAlignedFocalLoss` and `TaskAlignedMatcher`. |
| `DualLoss` | Runs the selected loss for both the auxiliary and main prediction heads. |

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
python -m yolo task=train task.loss.type=TOOD task.loss.gamma=1.0
```

### TOOD Specifics
When `type: TOOD` is set, the system uses `TaskAlignedMatcher` with `alpha` and `beta` parameters to explicitly align classification scores and IoU.

## Target Assignment

`YOLOLoss` and `TOODLoss` use different matchers (from `yolo/tasks/detection/postprocess.py`):
- **BoxMatcher**: Standard TAL-like assigner used by `YOLOLoss`.
- **TaskAlignedMatcher**: Strict TOOD assigner using $t = s^\alpha \times u^\beta$ used by `TOODLoss`.
