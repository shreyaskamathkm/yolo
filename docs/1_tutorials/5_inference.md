# Inference

## Inference on an Image

```bash
python -m yolo task=inference task.data.source=demo/images/inference/image.png
```

## Inference on a Video

```bash
python -m yolo task=inference task.data.source=demo/videos/sample.mp4
```

## Inference Config Reference

```yaml
task: inference

fast_inference:  # onnx, trt, deploy, or empty
data:
    source: demo/images/inference/image.png
    image_size: ${image_size}
    data_augment: {}
nms:
  min_confidence: 0.5
  min_iou: 0.5
# save_predict: True
```

See [All In One](0_allIn1.md#model-inference) for the full argument reference.
