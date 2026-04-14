# Custom Model

## Modified Architecture

YOLO model architectures are defined entirely in YAML files under `yolo/config/model/`. The YAML describes the backbone, neck, and head as a sequence of layers. You can create a new model by writing a new YAML file and passing `model=<your_file_stem>` on the CLI.

Each layer entry in the YAML has the form:

```yaml
- [source, number_of_repeats, module_name, [args...]]
```

| Field | Description |
|---|---|
| `source` | Index of the input layer(s). `-1` = previous layer; a list = multi-input |
| `number_of_repeats` | How many times to repeat this block (depth multiplier) |
| `module_name` | Class name — must be registered in `get_layer_map()` |
| `args` | Constructor arguments passed directly to the block |

See `yolo/config/model/v9-c.yaml` for a complete example.

## Adding a New Block

1. Write your `nn.Module` subclass in the appropriate file under `yolo/model/blocks/`:
   - `basic.py` — lightweight ops (Conv, pooling, upsampling, concat)
   - `backbone.py` — repeatable backbone blocks (Bottleneck, ELAN, RepNCSP, …)
   - `neck.py` — feature pyramid / neck blocks (SPPELAN, CBFuse, …)
   - `implicit.py` — implicit knowledge / anchor blocks

2. Export the class from `yolo/model/blocks/__init__.py`:

   ```python
   from yolo.model.blocks.basic import MyNewBlock
   ```

3. `get_layer_map()` in `yolo/utils/module_utils.py` auto-discovers all classes exported from `yolo.model.blocks`, so your block is immediately available in YAML configs by its class name — no further registration needed.

4. Reference it in a model YAML:

   ```yaml
   - [-1, 1, MyNewBlock, [256, 3]]
   ```
