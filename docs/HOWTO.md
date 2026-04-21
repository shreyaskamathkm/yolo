# How-To Guide

This guide provides practical examples and instructions for common tasks in the YOLO framework, from running basic commands to extending the codebase with custom components.

## Command Line Reference

### Training
```bash
# Standard training on COCO
yolo task=train model=v9-c dataset=coco

# Training with experiment tracking (WandB/TensorBoard)
yolo task=train use_wandb=True use_tensorboard=True name=my_experiment

# Multi-GPU Training (DDP)
torchrun --nproc_per_node=2 -m yolo task=train device=[0,1]
```

### Inference
```bash
# Run on an image or video
yolo task=inference task.data.source=images/test.jpg

# Run on a live stream (Webcam)
yolo task=inference task.data.source=0

# Use optimized backends
yolo task=inference task.backend=onnx weight=weights/v9-c.onnx
```

### Exporting
```bash
# Export a trained model to ONNX and TensorRT
yolo --config-name export weight=weights/v9-c.pt formats=[onnx,trt]
```

---

## Extending the Framework

### 1. Custom Model Architecture
You can define new architectures entirely in YAML without writing Python code.
Navigate to `yolo/config/model/` and create a new file (e.g., `my_custom_model.yaml`).

```yaml
model:
  backbone:
    - Conv:
        args: {out_channels: 64, kernel_size: 3, stride: 2}
    - CustomBlock:
        source: -1
        args: {out_channels: 128}
  neck:
    - SPPELAN:
        args: {out_channels: 256}
```

### 2. Custom NN Blocks
To add a new building block:
1. Define your module in `yolo/model/blocks/basic.py` (or similar).
2. Ensure it inherits from `nn.Module`.
3. Register it in the YAML config as shown above.

```python
class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
```

<a id="custom-data-augmentation"></a>
### 3. Custom Data Augmentation
Add new transformations in `yolo/data/augmentation.py`. They should follow the `(image, boxes) -> (image, boxes)` signature.

```python
class MyAugmentation:
    def __call__(self, image, boxes):
        # Apply logic
        return image, boxes
```

Then add it to your augmentation config in `yolo/config/data/augment.yaml`.

---

## Technical Deep Dive

If you need to modify core logic, here is where the main components live:

| Component | Path | Documentation |
|---|---|---|
| **Model Building** | `yolo/model/builder.py` | [Model API](6_function_docs/0_model.md) |
| **Data Pipelines** | `yolo/data/` | [Data API](6_function_docs/1_data.md) |
| **Training Solvers** | `yolo/tasks/detection/solver.py` | [Training API](6_function_docs/2_training.md) |
| **Loss Functions** | `yolo/tasks/detection/loss.py` | [Detection API](6_function_docs/3_detection.md) |
| **Post-Processing** | `yolo/tasks/detection/postprocess.py` | [Detection API](6_function_docs/3_detection.md) |
| **Optimizers** | `yolo/training/optim.py` | [Training API](6_function_docs/2_training.md) |
| **Deployment** | `yolo/deploy/` | [Deployment Guide](4_deploy/1_deploy.md) |

For more details on why the project is structured this way, see [Project Structure](0_get_start/3_project_structure.md).
