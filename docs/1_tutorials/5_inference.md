# Inference

YOLO supports various input sources and optimized backends.

## Inference Sources

You can run inference on images, videos, folders, or live streams.

### Single Image or Folder
```bash
yolo task=inference task.data.source=demo/images/inference/image.png
```

### Video File
```bash
yolo task=inference task.data.source=demo/videos/sample.mp4
```

### Live Streams (Webcam/RTSP)
```bash
# Webcam (ID 0)
yolo task=inference task.data.source=0

# RTSP Stream
yolo task=inference task.data.source="rtsp://admin:password@192.168.1.100:554/live"
```

## Configuration Reference

```yaml
task: inference
backend: torch  # onnx, trt, or torch (default)
data:
    source: demo/images/inference/image.png
    image_size: ${image_size}
    data_augment: {}
nms:
  min_confidence: 0.5
  min_iou: 0.5
save_predict: True
```

See [Inference & Deployment](../4_deploy/1_deploy.md) for more details on optimized backends.
