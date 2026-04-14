# Create Dataset

Create a dataloader with a single call:

```python
from yolo import create_dataloader
dataloader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task, use_ddp)
```

For inference, the dataset is handled by `StreamDataLoader`. For training and validation, it uses `YoloDataLoader`.

| Argument | Type | Description |
|---|---|---|
| `DataConfig` | `DataConfig` | Dataloader configuration |
| `DatasetConfig` | `DatasetConfig` | Dataset configuration |
| `task_name` | `str` | `inference`, `validation`, or `train` |
| `use_ddp` | `bool` | Whether to use Distributed Data Parallel. Default: `False` |

## Train and Validation

### Dataloader Return Type

Each iteration yields:

| Field | Description |
|---|---|
| `batch_size` | Batch size, used to calculate batch-average loss |
| `images` | Input images |
| `targets` | Ground truth for the task |

### Auto Download Dataset

If `auto_download` is configured in the dataset YAML, the dataset is downloaded automatically. Example config (`yolo/config/dataset/mock.yaml`):

```yaml
path: tests/data
train: train
validation: val

class_num: 80
class_list: ['Person', 'Bicycle', 'Car', ...]

auto_download:
  images:
    base_url: https://github.com/shreyaskamathkm/yolo/releases/download/v1-mock-data/
    train:
      file_name: mock_train
      file_num: 5
    val:
      file_name: mock_val
      file_num: 5
  annotations:
    base_url: https://github.com/shreyaskamathkm/yolo/releases/download/v1-mock-data/
    annotations:
      file_name: mock_annotations
```

The dataset is downloaded and unzipped from `{base_url}/{file_name}` and verified to contain `{file_num}` files. After verification, `{train,validation}.cache` files are generated in Tensor format to speed up future loads.

## Inference

In streaming mode, the model infers the most recent frame and draws bounding boxes. In other modes, predictions are saved to `runs/inference/{exp_name}/outputs/` by default.

### Dataloader Return Type

Each iteration of `StreamDataLoader` yields:

| Field | Type | Description |
|---|---|---|
| `images` | `Tensor` | Batch of input images |
| `rev_tensor` | `Tensor` | Reverse tensor for restoring bounding boxes to input shape |
| `origin_frame` | `Tensor` | Original input image |

### Input Types

**Stream:**

| Source | Type | Description |
|---|---|---|
| Webcam | `int` | Camera ID, e.g. `0`, `1` |
| RTMP | `str` | RTMP stream address |

**Single file:**

| Source | Type | Description |
|---|---|---|
| Image | `Path` | `.jpeg`, `.jpg`, `.png`, `.tiff` |
| Video | `Path` | `.mp4` |

**Folder:**

| Source | Type | Description |
|---|---|---|
| Image folder | `Path` | Relative or absolute path to a folder of images |
