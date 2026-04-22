# YOLO: YOLOv9, YOLOv7, YOLO-RD

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://shreyaskamathkm.github.io/yolo/)

![GitHub License](https://img.shields.io/github/license/shreyaskamathkm/yolo)

[![Integration Tests](https://github.com/shreyaskamathkm/yolo/actions/workflows/integration.yaml/badge.svg)](https://github.com/shreyaskamathkm/yolo/actions/workflows/integration.yaml)

[![Docker Publish](https://github.com/shreyaskamathkm/yolo/actions/workflows/docker.yaml/badge.svg)](https://github.com/shreyaskamathkm/yolo/actions/workflows/docker.yaml)

[![Release](https://github.com/shreyaskamathkm/yolo/actions/workflows/release.yaml/badge.svg)](https://github.com/shreyaskamathkm/yolo/actions/workflows/release.yaml)

<!-- > [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
>
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems. -->

This repository contains an implementation of YOLOv7[^1], YOLOv9[^2], and YOLO-RD[^3], forked and extended from [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO/tree/main/yolo). It includes the complete codebase, pre-trained models, and detailed instructions for training and deploying YOLO models.

## TL;DR

- This is the official YOLO model implementation with an MIT License.
- For quick deployment: you can directly install by pip+git:

```shell
pip install git+https://github.com/shreyaskamathkm/yolo.git
yolo task.data.source=0 # source could be a single file, video, image folder, webcam ID
```

## Introduction

- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
- [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)

## Installation

To get started using YOLOv9's developer mode, clone this repository and run:

```shell
git clone git@github.com:shreyaskamathkm/yolo.git
cd yolo
make setup
```

This will create a `.venv` virtual environment (if one doesn't already exist), install all dependencies, and set up pre-commit hooks. You can override the defaults with:

```shell
make setup VENV=myenv PYTHON=python3.11
```

## Features

<table>
<tr><td>

## Task

These are simple examples. For more customization details, please refer to [Notebooks](examples) and lower-level modifications **[HOWTO](docs/HOWTO.md)**.

## Training

To train YOLO on your machine/dataset:

1. Modify the configuration file `yolo/config/dataset/**.yaml` to point to your dataset.
2. Run the training script:

```shell
python yolo/lazy.py task=train dataset=** use_wandb=True
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c weight=False # or more args
```

### Transfer Learning

To perform transfer learning with YOLOv9:

```shell
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c dataset={dataset_config} device={cpu, mps, cuda}
```

### Inference

To use a model for object detection, use:

```shell
python yolo/lazy.py # if cloned from GitHub
python yolo/lazy.py task=inference \ # default is inference
                    name=AnyNameYouWant \ # AnyNameYouWant
                    device=cpu \ # hardware cuda, cpu, mps
                    model=v9-s \ # model version: v9-c, m, s
                    task.nms.min_confidence=0.1 \ # nms config
                    task.fast_inference=onnx \ # onnx, trt, deploy
                    task.data.source=data/toy/images/train \ # file, dir, webcam
                    +quiet=True \ # Quiet Output
yolo task.data.source={Any Source} # if pip installed
yolo task=inference task.data.source={Any}
```

### Validation

To validate model performance, or generate a json file in COCO format:

```shell
python yolo/lazy.py task=validation
python yolo/lazy.py task=validation dataset=toy
```

## Contributing

Contributions to the YOLO project are welcome! See [CONTRIBUTING](docs/CONTRIBUTING.md) for guidelines on how to contribute.

## To-Do List
- [ ] Test End to End to check model training on COCO
- [ ] Add MLFLow
- [ ] Add  yolo v9-e version
- [ ] Check DDP
- [ ] Refactor utils folder by moving the files to their own folders

## Acknowledgments

This project is a fork of [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO/tree/main/yolo). Many thanks to the MultimediaTechLab team for their work on the original implementation, which served as the foundation for this repository.

## Inference Results

![Inference Result 1](docs/assets/img_0000.jpg)
![Inference Result 2](docs/assets/img_0001.jpg)

### Images in inference borrowed from
- Image - [img_0000.jpg] (https://pixabay.com/photos/man-woman-bicycle-bike-air-sky-3861989/)
- Image - [img_0001.jpg](https://pixabay.com/photos/night-rome-italy-street-urban-7530755/)
## Citations

```
@inproceedings{wang2022yolov7,
      title={{YOLOv7}: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors},
      author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
      year={2023},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},

}
@inproceedings{wang2024yolov9,
      title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
      author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
      year={2024},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
}
@inproceedings{tsui2024yolord,
      author={Tsui, Hao-Tang and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
      title={{YOLO-RD}: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary},
      booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2025},
}

```

[^1]: [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)

[^2]: [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

[^3]: [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)
