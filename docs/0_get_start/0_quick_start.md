# Quick Start

!!! note
    We expect all customizations to be done primarily by passing arguments or modifying the YAML config files.
    If more detailed modifications are needed, custom content should be modularized as much as possible to avoid extensive code modifications.

## Install YOLO

Clone the repository and install the dependencies:

```bash
git clone https://github.com/shreyaskamathkm/yolo.git
cd yolo
make setup
# Make sure to work inside the cloned folder.
```

This creates a `.venv` virtual environment (if one doesn't exist), installs all dependencies, and sets up pre-commit hooks. You can override defaults with `make setup VENV=myenv PYTHON=python3.11`.

Alternatively, for a simple change:

!!! note
    In the following examples, replace `python -m yolo` with `yolo` if installed via pip.

```bash
pip install git+https://github.com/shreyaskamathkm/yolo.git
```

## Train Model

```bash
python -m yolo task=train

yolo task=train  # if installed via pip
```

- Override `dataset` to customize your dataset via a dataset config.
- Override `model` to select a backbone: `v9-c`, `v9-m`, etc.
- More details at [Train Tutorials](../1_tutorials/4_train.md).

```bash
python -m yolo task=train dataset=AYamlFilePath model=v9-m

yolo task=train dataset=AYamlFilePath model=v9-m  # if installed via pip
```

## Inference & Deployment

Inference is the default task of `yolo`. More details at [Inference Tutorials](../1_tutorials/5_inference.md).

```bash
python -m yolo task.data.source=AnySource

yolo task.data.source=AnySource  # if installed via pip
```

Enable fast inference modes by adding `task.fast_inference={onnx, trt, deploy}`. See [Inference & Deployment](../4_deploy/1_deploy.md) for setup instructions.
