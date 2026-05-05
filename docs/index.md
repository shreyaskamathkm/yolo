# YOLO Documentation

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system designed for both efficiency and accuracy. This documentation provides comprehensive guidance on how to set up, configure, and effectively use YOLO for object detection tasks.

## Project Features

- **Real-time Processing**: YOLO processes images in real-time with high accuracy, suitable for applications requiring instant detection.
- **Multitasking Capabilities**: Supports multitasking, allowing multiple object detection tasks simultaneously.
- **Open Source**: Released under the MIT License, encouraging community contributions.

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
