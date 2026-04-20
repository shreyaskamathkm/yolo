# Project Structure

The codebase is organized into horizontal shared infrastructure and vertical task packages.

```
yolo/
├── cli.py                      # CLI entry point (Hydra @main)
├── __main__.py                 # Enables `python -m yolo`
├── config/
│   ├── schemas/
│   │   ├── model.py            # AnchorConfig, ModelConfig, YOLOLayer
│   │   ├── data.py             # DatasetConfig, DataConfig, DownloadOptions
│   │   ├── training.py         # OptimizerConfig, SchedulerConfig, TrainConfig, EMAConfig, LossConfig
│   │   └── task.py             # NMSConfig, InferenceConfig, ValidationConfig, ExportConfig
│   ├── config.py               # Assembles Config from schemas; re-exports all dataclasses
│   └── [yaml files]            # Hydra config groups: model/, task/, dataset/, general/
├── deploy/                     # Optimized inference and exporting
│   ├── backends/               # Format-specific executors (ONNX, TRT, Torch)
│   ├── export.py               # ModelExporter implementation
│   ├── factory.py              # create_inference_backend factory
│   └── protocol.py             # InferenceBackend typing protocol
├── model/
│   ├── builder.py              # YOLO nn.Module and create_model()
│   └── blocks/
│       ├── __init__.py         # Re-exports all block classes
│       ├── basic.py            # Conv, Pool, Concat, UpSample
│       ├── backbone.py         # RepConv, Bottleneck, RepNCSP, ELAN, AConv, ADown
│       ├── neck.py             # SPPELAN, SPPCSPConv, CBLinear, CBFuse
│       └── implicit.py         # ImplicitA, ImplicitM, DConv, RepNCSPELAND
├── tasks/
│   ├── detection/
│   │   ├── head.py             # Detection, IDetection, MultiheadDetection
│   │   ├── loss.py             # BCELoss, BoxLoss, DFLoss, YOLOLoss, DualLoss
│   │   ├── postprocess.py      # Vec2Box, Anc2Box, BoxMatcher, bbox_nms, create_converter
│   │   └── solver.py           # DetectionTrainModel, DetectionValidateModel, DetectionInferenceModel
│   ├── segmentation/
│   │   └── head.py             # Segmentation, MultiheadSegmentation
│   └── classification/
│       └── head.py             # Classification
├── data/
│   ├── dataset.py              # YoloDataset, collate_fn
│   ├── loader.py               # create_dataloader, StreamDataLoader
│   ├── augmentation.py         # AugmentationComposer and transform classes
│   └── preparation.py          # prepare_dataset, prepare_weight
├── training/
│   ├── optim.py                # lerp, warmup policies, WarmupBatchScheduler, create_optimizer/scheduler
│   └── callbacks.py            # EMA, GradientAccumulation Lightning callbacks
└── utils/
    ├── logger.py               # Loguru logger instance
    ├── logging_utils.py        # Progress bars, WandB/TensorBoard setup, ImageLogger
    ├── drawer.py               # draw_bboxes, draw_model
    ├── model_utils.py          # PostProcess, distributed utilities (collect_prediction, get_device)
    ├── module_utils.py         # get_layer_map, auto_pad, round_up
    ├── dataset_utils.py        # locate_label_paths, create_image_metadata, scale_segmentation
    ├── format_converters.py    # discretize_categories, annotation conversion utilities
    └── solver_utils.py         # make_ap_table and other display helpers
```

## Where to Add Things

| What you want to add | Where it goes |
|---|---|
| New NN building block (conv, attention, etc.) | `model/blocks/basic.py` or `backbone.py` / `neck.py` |
| New backbone or neck architecture | New YAML in `config/model/` + blocks in `model/blocks/` |
| New data augmentation transform | `data/augmentation.py` |
| Detection loss tweak | `tasks/detection/loss.py` |
| Post-processing or NMS change | `tasks/detection/postprocess.py` |
| Training callback (LR logging, etc.) | `training/callbacks.py` |
| Config field | Add dataclass field to the relevant `config/schemas/` file |
