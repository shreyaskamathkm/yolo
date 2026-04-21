# Detection

## Loss

::: yolo.tasks.detection.loss
    options:
      members:
        - create_loss_function
        - YOLOLoss
        - DualLoss
      undoc-members: true
      show-inheritance: true


## Postprocess

::: yolo.tasks.detection.postprocess
    options:
      members:
        - calculate_iou
        - transform_bbox
        - bbox_nms
        - create_converter
        - BoxMatcher
        - Vec2Box
        - Anc2Box
      undoc-members: true
      show-inheritance: true

## Head

::: yolo.tasks.detection.head
    options:
      members:
        - Detection
        - IDetection
        - MultiheadDetection
      undoc-members: true
      show-inheritance: true
