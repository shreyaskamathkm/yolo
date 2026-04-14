# How To modified YOLO

To facilitate easy customization of the YOLO model, we've structured the codebase to allow for changes through configuration files and minimal code adjustments. This guide will walk you through the steps to customize various components of the model including the architecture, blocks, data loaders, and loss functions.

## Examples

```shell
# Train
python -m yolo task=train dataset=dev use_wandb=True

# Validate
python -m yolo task=validation
python -m yolo task=validation model=v9-s
python -m yolo task=validation dataset=toy
python -m yolo task=validation dataset=toy name=validation

# Inference
python -m yolo task=inference
python -m yolo task=inference device=cpu
python -m yolo task=inference +quiet=True
python -m yolo task=inference name=AnyNameYouWant
python -m yolo task=inference image_size=\[480,640]
python -m yolo task=inference task.nms.min_confidence=0.1
python -m yolo task=inference task.fast_inference=deploy
python -m yolo task=inference task.fast_inference=onnx device=cpu
python -m yolo task=inference task.data.source=data/toy/images/train
```

## Custom Model Architecture

You can change the model architecture simply by modifying the YAML configuration file. Here's how:

1. **Modify Architecture in Config:**

   Navigate to your model's configuration file (typically formate like `yolo/config/model/v9-c.yaml`).
   - Adjust the architecture settings under the `architecture` section. Ensure that every module you reference exists in one of the `model/blocks/` files, or refer to the next section on how to add new modules.

    ```yaml
    model:
      foo:
        - ADown:
            args: {out_channels: 256}
        - RepNCSPELAN:
            source: -2
            args: {out_channels: 512, part_channels: 256}
            tags: B4
      bar:
        - Concat:
            source: [-2, B4]
    ```

   `tags`: Use this to labels any module you want, and could be the module source.

   `source`: Set this to the index of the module output you wish to use as input; default is `-1` which refers to the last module's output. Capable tags, relative position, absolute position

   `args`: A dictionary used to initialize parameters for convolutional or bottleneck layers.

   `output`: Whether to serve as the output of the model.

## Custom Block

To add or modify a block in the model:

1. **Create a New Module:**

   Define a new class in the appropriate `model/blocks/` file that inherits from `nn.Module`.

   Place basic building blocks (Conv, Pool) in `model/blocks/basic.py`, backbone blocks in `model/blocks/backbone.py`, neck blocks in `model/blocks/neck.py`, and YOLOR implicit blocks in `model/blocks/implicit.py`.

   The constructor should accept `in_channels` as a parameter. Make sure to calculate `out_channels` based on your model's requirements or configure it through the YAML file using `args`.

    ```python
    class CustomBlock(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.module = # conv, bool, ...
        def forward(self, x):
            return self.module(x)
    ```

2. **Reference in Config:**
   ```yaml
    ...
    - CustomBlock:
        args: {out_channels: int, etc: ...}
        ...
    ...
   ```


## Custom Data Augmentation

Custom transformations should be designed to accept an image and its bounding boxes, and return them after applying the desired changes. Here’s how you can define such a transformation:


1. **Define Dataset:**

    Your class must have a `__call__` method that takes a PIL image and its corresponding bounding boxes as input, and returns them after processing.


   ```python
    class CustomTransform:
        def __init__(self, prob=0.5):
            self.prob = prob

        def __call__(self, image, boxes):
            return image, boxes
   ```
2. **Update CustomTransform in Config:**

    Specify your custom transformation in a YAML config `yolo/config/data/augment.yaml`. For examples:
    ```yaml
    Mosaic: 1
    # ... (Other Transform)
    CustomTransform: 0.5
    ```


- **Utils**
    - **tasks/detection/postprocess** (bounding box utilities)
        - `class` Vec2Box: transform predicted vectors to bounding boxes
        - `class` Anc2Box: transform predicted anchors to bounding boxes (v7-style)
        - `class` BoxMatcher: given predictions and ground truth, assign best matching GT box
        - `func` calculate_iou: calculate IoU for two lists of bboxes
        - `func` transform_bbox: transform bbox between formats (xywh, xyxy, xycwh)
        - `func` generate_anchors: given image size, generate anchor points
        - `func` bbox_nms: apply NMS to predicted bounding boxes
        - `func` create_converter: factory that returns Vec2Box or Anc2Box based on model name
        - `func` to_metrics_format: convert predictions to torchmetrics-compatible format
    - **dataset_utils**
        - `func` locate_label_paths:
        - `func` create_image_metadata:
        - `func` organize_annotations_by_image:
        - `func` scale_segmentation:
    - **logging_utils**
        - `func` custom_log: custom loguru, overriding the origin logger
        - `class` ProgressTracker: A class to handle output for each batch, epoch
        - `func` log_model_structure: give a torch model, print it as a table
        - `func` validate_log_directory: for given experiment, check if the log folder already exists
    - **training/callbacks** (EMA and gradient accumulation)
        - `class` EMA: Lightning Callback that maintains an exponential moving average of model weights
        - `class` GradientAccumulation: Lightning Callback that ramps gradient accumulation steps during warmup
    - **training/optim**
        - `func` lerp: linear interpolation between two values
        - `class` LinearWarmupPolicy: uniform LR ramp from 0 → initial_lr over warmup epochs
        - `class` YOLOWarmupPolicy: YOLO-style warmup — bias group drops, other groups rise
        - `class` WarmupBatchScheduler: batch-level LR and momentum scheduler with epoch-aware warmup
        - `func` create_optimizer: return an optimizer, for example SGD, Adam
        - `func` create_scheduler: return a WarmupBatchScheduler wrapping an epoch-level scheduler
    - **module_utils**
        - `func` get_layer_map:
        - `func` auto_pad: given a convolution block, return how many pixel should conv padding
        - `func` create_activation_function: given a `func` name, return an activation function
        - `func` round_up: given number and divider, return a number that is a multiple of divider
        - `func` divide_into_chunks: for a given list and n, separate list into n sub-lists
    - **training/solver**
        - `class` BaseModel: base Lightning module wrapping the YOLO model
        - `class` ValidateModel: adds validation loop (metrics, EMA, val dataloader)
        - `class` TrainModel: adds training loop, loss, and optimizer configuration
        - `class` InferenceModel: runs prediction and optionally saves visualized output
- **Data** (`yolo/data/`)
    - **data/augmentation**
        - `class` AugmentationComposer: compose a list of data augmentation strategies
        - `class` VerticalFlip: random vertical flip augmentation
        - `class` Mosaic: Mosaic augmentation strategy
    - **data/dataset**
        - `class` YoloDataset: custom PyTorch Dataset for YOLO training
        - `func` collate_fn: batches images and targets for the DataLoader
    - **data/loader**
        - `func` create_dataloader: given a config, return a DataLoader or StreamDataLoader
        - `class` StreamDataLoader: streams images/video from disk, folder, or RTSP for inference
    - **data/preparation**
        - `func` prepare_dataset: auto-download and verify the dataset
        - `func` prepare_weight: download pretrained weights if missing
- **Tasks** (`yolo/tasks/`)
    - **tasks/detection/loss**
        - `class` BCELoss: binary cross-entropy classification loss
        - `class` BoxLoss: CIoU-based bounding box regression loss
        - `class` DFLoss: Distribution Focal Loss for anchor distribution
        - `class` YOLOLoss: combines BCE + Box + DFL losses with target assignment
        - `class` DualLoss: wraps YOLOLoss for AUX + Main dual-head training
        - `func` create_loss_function: factory returning a DualLoss instance
    - **tasks/detection/head**
        - `class` Detection: single YOLO detection head
        - `class` IDetection: YOLOv7-style implicit detection head
        - `class` MultiheadDetection: multi-scale detection head (dual/triple detect)
    - **utils/format_converters** (format conversion utilities)
        - `func` discretize_categories: remap COCO category IDs to contiguous indices
        - `func` convert_annotations: convert JSON annotations to YOLO txt format
