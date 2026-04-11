# Config

::: yolo.config.config.Config
    options:
      members: true
      undoc-members: true

::: yolo.config.config
    options:
      members: true
      undoc-members: true

## Schema

### Model Config

```mermaid
classDiagram
class AnchorConfig {
    List~int~ strides
    Optional~int~ reg_max
    Optional~int~ anchor_num
    List~List~int~~ anchor
}

class LayerConfig {
    Dict args
    Union~List~int~~ source
    str tags
}

class BlockConfig {
    List~Dict~LayerConfig~~ block
}

class ModelConfig {
    Optional~str~ name
    AnchorConfig anchor
    Dict~BlockConfig~ model
}

AnchorConfig --> ModelConfig
LayerConfig --> BlockConfig
BlockConfig --> ModelConfig
```

### Dataset Config

```mermaid
classDiagram
class DownloadDetail {
    str url
    int file_size
}

class DownloadOptions {
    Dict~DownloadDetail~ details
}

class DatasetConfig {
    str path
    int class_num
    List~str~ class_list
    Optional~DownloadOptions~ auto_download
}

class DataConfig {
    bool shuffle
    int batch_size
    bool pin_memory
    int cpu_num
    List~int~ image_size
    Dict~int~ data_augment
    Optional~Union~str~~ source
    Optional~bool~ dynamic_shape
    Optional~int~ equivalent_batch_size
    bool drop_last
}

DownloadDetail --> DownloadOptions
DownloadOptions --> DatasetConfig
```

### Train Config

```mermaid
classDiagram
class OptimizerArgs {
    float lr
    float weight_decay
    float momentum
}

class OptimizerConfig {
    str type
    OptimizerArgs args
}

class MatcherConfig {
    str iou
    int topk
    Dict~str~ factor
}

class LossConfig {
    Dict~str~ objective
    Union~bool~ aux
    MatcherConfig matcher
}

class WarmupConfig {
    float epochs
    float start_momentum
    float end_momentum
}

class SchedulerConfig {
    str type
    WarmupConfig warmup
    Dict~str~ args
}

class EMAConfig {
    bool enable
    float decay
}

class TrainConfig {
    str task
    int epoch
    DataConfig data
    OptimizerConfig optimizer
    LossConfig loss
    SchedulerConfig scheduler
    EMAConfig ema
    ValidationConfig validation
}

class NMSConfig {
    float min_confidence
    float min_iou
    int max_bbox
}

class InferenceConfig {
    str task
    NMSConfig nms
    DataConfig data
    Optional~None~ fast_inference
    bool save_predict
}

class ValidationConfig {
    str task
    NMSConfig nms
    DataConfig data
}

OptimizerArgs --> OptimizerConfig
OptimizerConfig --> TrainConfig
MatcherConfig --> LossConfig
LossConfig --> TrainConfig
WarmupConfig --> SchedulerConfig
SchedulerConfig --> TrainConfig
EMAConfig --> TrainConfig
NMSConfig --> InferenceConfig
NMSConfig --> ValidationConfig
```

### General Config

```mermaid
classDiagram
class GeneralConfig {
    str name
    Optional~str~ accelerator
    Union~str~ device
    int cpu_num
    List~int~ image_size
    str out_path
    bool exist_ok
    int lucky_number
    bool use_wandb
    bool use_tensorboard
    Optional~str~ weight
}
```

### Top-level Config

```mermaid
classDiagram
class Config {
    Union~ValidationConfig~ task
    DatasetConfig dataset
    ModelConfig model
    GeneralConfig general
}

DatasetConfig --> Config
DataConfig --> TrainConfig
DataConfig --> InferenceConfig
DataConfig --> ValidationConfig
InferenceConfig --> Config
ValidationConfig --> Config
TrainConfig --> Config
GeneralConfig --> Config
```
