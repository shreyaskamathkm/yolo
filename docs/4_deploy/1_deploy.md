# Inference & Deployment

YOLO supports three fast inference backends selectable at runtime via `task.fast_inference`.

## Modes

| Mode | Flag | Requires |
|---|---|---|
| PyTorch (default) | _(omit flag)_ | nothing extra |
| Deploy | `task.fast_inference=deploy` | nothing extra |
| ONNX | `task.fast_inference=onnx` | `onnxruntime` |
| TensorRT | `task.fast_inference=trt` | `torch2trt`, CUDA |

### Deploy mode

Strips the auxiliary head from the model before running inference. No extra dependencies — just a lighter forward pass.

```bash
python -m yolo task=inference task.fast_inference=deploy
```

### ONNX

Exports the model to ONNX on the first run, then reuses the `.onnx` file on subsequent runs.

```bash
pip install onnxruntime        # CPU
pip install onnxruntime-gpu    # GPU

python -m yolo task=inference task.fast_inference=onnx
python -m yolo task=inference task.fast_inference=onnx device=cpu
```

The exported file is saved as `<weight_stem>.onnx` next to the weight file.

### TensorRT

Builds a TensorRT engine on the first run (requires a CUDA GPU), then reuses the `.trt` file.

```bash
pip install torch2trt

python -m yolo task=inference task.fast_inference=trt
```

!!! note
    TensorRT is not supported on MPS (Apple Silicon). The loader falls back to the standard PyTorch model automatically.

## API

::: yolo.utils.deploy_utils.FastModelLoader
    options:
      members: true
      undoc-members: true
      show-inheritance: true
