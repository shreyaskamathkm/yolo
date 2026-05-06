# YOLO Documentation

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system designed for both efficiency and accuracy. This documentation provides comprehensive guidance on how to set up, configure, and effectively use YOLO for object detection tasks.

## Key Features
- **Unified Registry System**: Centralized hub for models, blocks, losses, and transforms, enabling easy extension.
- **Lightning Fast**: Built on PyTorch Lightning for effortless multi-GPU and mixed-precision training.
- **Modular Design**: Decoupled architecture where backbones, necks, and heads can be swapped via YAML.
- **Production Ready**: Optimized export pipelines for ONNX and TensorRT.

## Extensibility
The YOLO repository is built with modularity as a first-class citizen. Using our new registry system, you can easily integrate your own components:

- **Custom Blocks**: Register any `nn.Module` with `@BLOCKS.register_module()` and use it directly in your YAML architecture.
- **Custom Models**: Plug in complex architectures like **FTNet** by registering them in `MODELS`.
- **Custom Losses**: Implement task-specific losses and register them in `LOSSES` using tuple keys: `@LOSSES.register_module(name=("task", "name"))`.
- **Custom Solvers**: Register task solvers in `SOLVERS` using tuple keys: `@SOLVERS.register_module(name=("task", "mode"))`.
- **Dynamic Transforms**: Add new data augmentations to `TRANSFORMS` without modifying the core loader.

Check the [Project Structure](0_get_start/3_project_structure.md) to see where everything lives.

## Interactive Demonstrations

Explore the new data pipeline and augmentation system interactively:
- **[Dataloader Demo](../notebooks/coco_dataloader_demo.ipynb)**: Detailed look at the detection pipeline, including Mosaic/MixUp visualizations.
- **[Segmentation Demo](../notebooks/coco_segmentation_demo.ipynb)**: Demonstrates synchronized image and mask transformations for instance segmentation.

## Core Components

- **[Data Pipeline](dataset_architecture.md)**: Modular dataset and dataloader architecture using type-safe Enums.
- **[Augmentation System](augmentation.md)**: Synchronized, immutable transformations for images, boxes, and masks.
- **[Model Architecture](2_model_zoo/index.md)**: Overview of supported YOLO backbones and tasks.

## Acknowledgments

This project is a fork of [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO/tree/main/yolo). Many thanks to the MultimediaTechLab team for their work on the original implementation, which served as the foundation for this repository.

## License

YOLO is provided under the MIT License. See the [LICENSE](https://github.com/shreyaskamathkm/yolo/blob/main/LICENSE) file for full license text.
