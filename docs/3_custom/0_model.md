# Custom Model

## Modified Architecture

YOLO model architectures are defined in YAML files under `yolo/config/model/`. The YAML describes the backbone, neck, and head as a sequence of layers. You can create a new model by writing a new YAML file and passing `model=<your_file_stem>` on the CLI.

Each layer entry in the YAML has the form:

```yaml
- [source, number_of_repeats, module_name, [args...]]
```

| Field | Description |
|---|---|
| `source` | Index of the input layer(s). `-1` = previous layer; a list = multi-input; a tag string = tagged layer |
| `number_of_repeats` | How many times to repeat this block (depth multiplier) |
| `module_name` | Class name — must be registered in the `BLOCKS` registry |
| `args` | Constructor arguments passed directly to the block |

See `yolo/config/model/v9-c.yaml` for a complete example.

## Adding a New Block

1. Write your `nn.Module` subclass and register it using `@BLOCKS.register_module()`:

   ```python
   from yolo.registry import BLOCKS

   @BLOCKS.register_module()
   class MyNewBlock(nn.Module):
       def __init__(self, in_channels, out_channels, **kwargs):
           super().__init__()
           # ... implementation ...
   ```

2. Standard blocks are located in `yolo/model/blocks/`:
   - `basic.py` — lightweight ops (Conv, pooling, upsampling, concat)
   - `backbone.py` — repeatable backbone blocks (Bottleneck, ELAN, RepNCSP, …)
   - `neck.py` — feature pyramid / neck blocks (SPPELAN, CBFuse, …)
   - `implicit.py` — implicit knowledge / anchor blocks
   - `head.py` — prediction heads (Detection, IDetection, TOODHead)

3. Reference it in a model YAML:

   ```yaml
   - [-1, 1, MyNewBlock, [256, 3]]
   ```

## Adding a Custom Model Class

If your model cannot be described by a sequential YAML architecture (e.g., non Yolo Models), you can register a custom model class in the `MODELS` registry.

1. Define your model class:

   ```python
   from yolo.registry import MODELS

   @MODELS.register_module()
   class MyCustomModel(nn.Module):
       def __init__(self, model_cfg, class_num=80):
           super().__init__()
           # ... custom building logic ...
   ```

2. Use it in your config by setting the `type` field:

   ```yaml
   model:
     name: my_custom
     type: MyCustomModel
     # ... other config ...
   ```

3. `create_model` will automatically instantiate your custom class from the registry.
