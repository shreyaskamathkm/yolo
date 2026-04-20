# Training

## Solver

::: yolo.tasks.detection.solver
    options:
      members:
        - DetectionTrainModel
        - DetectionValidateModel
        - DetectionInferenceModel
      undoc-members: true
      show-inheritance: true


## Optimizer & Scheduler

::: yolo.training.optim
    options:
      members:
        - create_optimizer
        - create_scheduler
        - WarmupBatchScheduler
      undoc-members: true
      show-inheritance: true


## Callbacks

::: yolo.training.callbacks
    options:
      members:
        - EMA
        - GradientAccumulation
      undoc-members: true
      show-inheritance: true
