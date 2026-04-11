# Install YOLO

This guide will help you set up YOLO on your machine. We recommend starting with [Git & GitHub](#git-github) for more flexible customization. If you only need inference or simple customization, install via [pip](#pypi-pip-install).

## Torch Requirements

=== "Linux"

    === "CUDA"
        PyTorch: 1.12+

    === "CPU"
        PyTorch: 1.12+

=== "MacOS"

    === "MPS"
        PyTorch: 2.2+

    === "CPU"
        PyTorch: 2.2+

=== "Windows"

    === "CUDA"
        [WIP]

    === "CPU"
        [WIP]

## Git & GitHub

Clone the repository:

```bash
git clone https://github.com/WongKinYiu/YOLO.git
```

Or download directly via the [ZIP archive](https://github.com/WongKinYiu/YOLO/archive/refs/heads/main.zip).

Install all dependencies:

```bash
make setup
```

This creates a `.venv` virtual environment (if one doesn't exist), installs all dependencies (`requirements-dev.txt` + editable package install), and configures pre-commit hooks. You can override defaults:

```bash
make setup VENV=myenv PYTHON=python3.11
```

For ONNX or TensorRT, see [ONNX](../4_deploy/2_onnx.md) and [TensorRT](../4_deploy/3_tensorrt.md).

## PyPI (pip install)

!!! note
    Due to the `yolo` name already being occupied on PyPI, we provide installation via GitHub. Ensure `git` and `pip` are available in your shell.

```bash
pip install git+https://github.com/WongKinYiu/YOLO.git
```

## Docker

```bash
docker pull henrytsui000/yolo
docker run --gpus all -it henrytsui000/yolo
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Conda

Coming soon.
