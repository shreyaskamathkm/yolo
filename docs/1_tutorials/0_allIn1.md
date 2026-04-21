# All In One

`yolo` is a packaged entry point that includes **training**, **validation**, and **inference** tasks. The following sections break down each operation and show how to import and call these functions directly.

## Train Model

```bash
python -m yolo task=train
```

Common arguments:

| Argument | Type | Description |
|---|---|---|
| `name` | `str` | Experiment name |
| `model` | `str` | Model backbone: `v9-c`, `v7`, `v9-e`, etc. |
| `task.data.dataloader_workers` | `int` | Number of CPU workers (`num_workers`) |
| `out_path` | `Path` | Output path for models and logs |
| `weight` | `Path \| bool \| None` | Pre-trained weights path. `False` = scratch, `None` = default |
| `use_wandb` | `bool` | Enable Weights & Biases tracking |
| `use_tensorboard` | `bool` | Enable TensorBoard logging |
| `image_size` | `int \| [int, int]` | Input image size |
| `+quiet` | `bool` | Disable all output (optional) |
| `task.epoch` | `int` | Total training epochs |
| `task.data.batch_size` | `int` | Batch size |

### Example

```bash
python -m yolo task=train task.data.batch_size=12 image_size=1280
```

### Multi-GPU Training with DDP

For multi-GPU training using Distributed Data Parallel (DDP), replace `python` with `torchrun --nproc_per_node=[GPU_NUM]`:

=== "bash"
    ```bash
    torchrun --nproc_per_node=2 -m yolo task=train device=[0,1]
    ```

=== "zsh"
    ```bash
    torchrun --nproc_per_node=2 -m yolo task=train device=\[0,1\]
    ```

### Training on a Custom Dataset

Example dataset config (`yolo/config/dataset/dev.yaml`):

```yaml
path: data/dev
train: train
validation: val

class_num: 80
class_list: ['Person', 'Bicycle', 'Car', ...]

auto_download:
```

Config fields:

| Field | Type | Description |
|---|---|---|
| `path` | `str` | Path to the dataset |
| `train`, `validation` | `str` | Directory names under `/images` (and `/labels/` for txt labels) |
| `class_num` | `int` | Number of dataset classes |
| `class_list` | `List[str]` | Optional class names for visualizing bounding boxes |
| `auto_download` | `dict` | Optional: auto-download configuration |

Expected dataset structure:

```text
DataSetName/
├── annotations/
│   ├── train_json_name.json
│   └── val_json_name.json
├── labels/
│   ├── train/
│   │   ├── AnyLabelName.txt
│   │   └── ...
│   └── validation/
│       └── ...
└── images/
    ├── train/
    │   ├── AnyImageNameN.{png,jpg,jpeg}
    │   └── ...
    └── validation/
        └── ...
```

## Validation Model

During training this runs automatically. Run manually to generate a JSON file of predictions for a validation dataset. If the set includes JSON annotations, pycocotools evaluation runs automatically.

Common arguments:

| Argument | Type | Description |
|---|---|---|
| `task.nms.min_confidence` | `float` | Minimum prediction confidence |
| `task.nms.min_iou` | `float` | Minimum IoU threshold for NMS |

### Example

=== "git-cloned"
    ```bash
    python -m yolo task=validation task.nms.min_iou=0.9
    ```

=== "PyPI"
    ```bash
    yolo task=validation task.nms.min_iou=0.9
    ```

## Model Inference

!!! note
    Do not override `dataset` — the model requires `class_num` from it. If classes have names, provide `class_list`.

Common arguments:

| Argument | Type | Description |
|---|---|---|
| `task.backend` | `str` | `onnx`, `trt`, `torch`. `torch` (the default) automatically detaches auxiliary heads. |
| `task.data.source` | `str | Path | int` | Webcam ID, image folder, video/image path, RTSP stream |
| `task.nms.min_confidence` | `float` | Minimum prediction confidence |
| `task.nms.min_iou` | `float` | Minimum IoU threshold for NMS |

### Example

=== "git-cloned"
    ```bash
    python -m yolo model=v9-m task.nms.min_confidence=0.1 task.data.source=0 task.backend=onnx
    ```

=== "PyPI"
    ```bash
    yolo model=v9-m task.nms.min_confidence=0.1 task.data.source=0 task.backend=onnx
    ```
