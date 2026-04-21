# Inference & Deployment

YOLO supports high-performance inference across multiple backends. You can explicitly export models to specialized formats and run them using optimized backends.

## Exporting Models

The `export` task allows you to convert PyTorch models to other formats like ONNX or TensorRT. This process also optimizes the model by stripping auxiliary training heads.

### Usage

```bash
yolo --config-name export weight=weights/v9-c.pt formats=[onnx,trt] output_dir=exports/
```

### Configuration Options

| Option | Description | Default |
|---|---|---|
| `weight` | Path to the PyTorch `.pt` model | `None` |
| `formats` | List of formats to export (`torch`, `onnx`, `trt`) | `[onnx, torch, trt]` |
| `output_dir` | Directory to save exported models | `exports/` |
| `image_size` | Input resolution `[H, W]` | `[640, 640]` |

Each format will produce a file with the corresponding extension (e.g., `.onnx`, `.trt`, `.pt`) in the `output_dir`.

---

## Inference Backends

Choose a backend at runtime using `task.backend`.

| Backend | Key | Requires | Extension |
|---|---|---|---|
| Torch (Deploy) | `torch` | nothing extra | `.pt` |
| ONNX | `onnx` | `onnxruntime` | `.onnx` |
| TensorRT | `trt` | `torch2trt`, CUDA | `.trt` |

### Running Inference

To run inference using a specific backend:

```bash
# Using PyTorch (automatically strips aux heads)
yolo task=inference task.backend=torch weight=weights/v9-c.pt

# Using ONNX
yolo task=inference task.backend=onnx weight=exports/v9-c.onnx

# Using TensorRT
yolo task=inference task.backend=trt weight=exports/v9-c.trt
```

### Backend Details

#### ONNX
High-performance cross-platform inference.
```bash
pip install onnxruntime        # CPU
pip install onnxruntime-gpu    # GPU
```

#### TensorRT
Maximum performance on NVIDIA GPUs.
```bash
pip install torch2trt
```
!!! note
    TensorRT is not supported on MPS (Apple Silicon). If selected on non-CUDA devices, it may fail or fall back depending on the environment.

## API

::: yolo.deploy.ModelExporter
    options:
      members:
        - __init__
        - __call__
