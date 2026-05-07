import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import ListConfig, OmegaConf
from torch import nn

import yolo.model.blocks
from yolo.config.config import ModelConfig, YOLOLayer
from yolo.data.preparation import prepare_weight
from yolo.registry import BLOCKS, MODELS
from yolo.utils.module_utils import align_state_dict, clean_state_dict

logger = logging.getLogger(__name__)


@MODELS.register_module()
class ConfigModel(nn.Module):
    """A task-agnostic model class that assembles layers from a configuration.

    This class dynamically builds the neural network based on the architecture
    defined in a configuration. It handles layer instantiation, source indexing,
    and weights management using the BLOCKS registry.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        """Initializes the ConfigModel.

        Args:
            model_cfg (ModelConfig): Architecture and anchor configuration.
            class_num (int, optional): Number of output classes. Defaults to 80.
        """
        super().__init__()

        self.num_classes = class_num
        self.reg_max = getattr(model_cfg.anchor, "reg_max", 16)
        self.meta = {
            "num_classes": self.num_classes,
            "reg_max": self.reg_max,
        }

        self.model: List[YOLOLayer] = nn.ModuleList()
        self.layer_index = {}
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict]]]):
        """Assembles the model from an architecture specification.

        Args:
            model_arch (Dict): Dictionary containing architecture stages (e.g., backbone, head).
        """
        output_dim, layer_idx = [3], 1
        logger.info(f":tractor: Building Model")

        for stage_name, stage_spec in model_arch.items():
            if stage_spec:
                logger.info(f"  :building_construction:  Building {stage_name}")

            for layer_spec in stage_spec:
                layer_type, layer_info = next(iter(layer_spec.items()))

                # Use a plain dict to avoid OmegaConf struct issues
                if isinstance(layer_info, (OmegaConf, ListConfig)):
                    layer_args = OmegaConf.to_container(layer_info.get("args", {}), resolve=True)
                else:
                    layer_args = dict(layer_info.get("args", {}))

                # Resolve source indices
                source = self.get_source_idx(layer_info.get("source", -1), layer_idx)

                # Standardized channel passing
                if isinstance(source, list):
                    layer_args["in_channels"] = [output_dim[idx] for idx in source]
                else:
                    layer_args["in_channels"] = output_dim[source]

                # Inject meta attributes
                layer_args.update(self.meta)

                # Create layer
                layer = self.create_layer(layer_type, source, layer_info, **layer_args)
                self.model.append(layer)

                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(f"Duplicate tag '{layer.tags}' found.")
                    self.layer_index[layer.tags] = layer_idx

                # Calculate output channels
                out_channels = self.get_out_channels(layer, layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)

                layer_idx += 1

    def forward(
        self, x: torch.Tensor, external: Optional[Dict] = None, shortcut: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Performs a forward pass."""
        y = {0: x, **(external or {})}
        output = dict()
        for index, layer in enumerate(self.model, start=1):
            if isinstance(layer.source, list):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]

            external_input = {source_name: y[source_name] for source_name in layer.external}

            x = layer(model_input, **external_input)
            y[-1] = x
            if layer.usable:
                y[index] = x
            if layer.output:
                output[layer.tags] = x
                if layer.tags == shortcut:
                    return output
        return output

    def get_out_channels(
        self, layer: nn.Module, layer_type: str, layer_args: dict, output_dim: List[int], source: Union[int, List[int]]
    ) -> int:
        """Calculates output channels for a layer."""
        if "out_channels" in layer_args:
            out_c = layer_args["out_channels"]
            if isinstance(out_c, list):
                return sum(out_c)
            return out_c

        if hasattr(layer, "out_channels"):
            out_c = layer.out_channels
            if isinstance(out_c, list):
                return sum(out_c)
            return out_c

        # Heuristics for common blocks
        if layer_type == "Concat":
            return sum(output_dim[idx] for idx in source)
        if layer_type == "CBFuse":
            return output_dim[source[-1]]

        # Fallback: assume input dimension
        if isinstance(source, int):
            return output_dim[source]
        return output_dim[source[-1]]

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int) -> Union[int, List[int]]:
        """Resolves relative or tagged source indices to absolute indices."""
        if isinstance(source, ListConfig):
            return [self.get_source_idx(index, layer_idx) for index in source]
        if isinstance(source, str):
            source = self.layer_index.get(source)
            if source is None:
                raise ValueError(f"Tag '{source}' not found.")
        if source < -1:
            source += layer_idx
        if source > 0:
            self.model[source - 1].usable = True
        return source

    def create_layer(
        self, layer_type: str, source: Union[int, List[int]], layer_info: Dict, **kwargs: Any
    ) -> YOLOLayer:
        """Instantiates a layer from the registry."""
        if layer_type not in BLOCKS:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        # Filter meta arguments: only pass if they are in the signature
        sig = inspect.signature(BLOCKS[layer_type].__init__)
        meta_keys = ["num_classes", "reg_max", "in_channels"]
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k in meta_keys:
                if k in sig.parameters:
                    filtered_kwargs[k] = v
            else:
                filtered_kwargs[k] = v

        layer = BLOCKS[layer_type](**filtered_kwargs)

        # Attach metadata required by builder
        setattr(layer, "layer_type", layer_type)
        setattr(layer, "source", source)
        setattr(layer, "in_c", kwargs.get("in_channels", None))
        setattr(layer, "output", layer_info.get("output", False))
        setattr(layer, "tags", layer_info.get("tags", None))
        setattr(layer, "external", layer_info.get("external", []))
        setattr(layer, "usable", False)

        return layer

    def save_load_weights(self, weights: Union[Path, dict], weight_key: str = "state_dict", strict: bool = False):
        """Robust weight loading logic with SOLID separation.

        Args:
            weights: Path to weights file or state dict.
            weight_key: Key in the weights dict to load from.
            strict: If True, raises ValueError if not all keys are matched.
        """
        loaded_dict = self._extract_state_dict(weights, weight_key)
        loaded_dict = clean_state_dict(loaded_dict)
        aligned_state, matched_count = align_state_dict(self.model.state_dict(), loaded_dict)
        self._apply_aligned_weights(aligned_state, matched_count, strict)

    def _extract_state_dict(self, weights: Union[Path, dict], weight_key: str) -> dict:
        """Extracts the state dictionary from various input formats."""
        if isinstance(weights, (str, Path)):
            weights = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
        return weights.get(weight_key, weights.get("state_dict", weights))

    def _apply_aligned_weights(self, aligned_state: dict, matched_count: int, strict: bool):
        """Verifies matching results and applies the aligned state dict to the model."""
        total_keys = len(aligned_state)

        if matched_count == 0:
            logger.error("❌ No weights were matched!")
            if strict:
                raise ValueError("No weights were matched during loading.")
        elif matched_count < total_keys:
            msg = f"⚠️ Only matched {matched_count}/{total_keys} weight tensors"
            if strict:
                logger.error(msg)
                raise ValueError(msg)
            else:
                logger.warning(msg)
        else:
            logger.info(f"✅ Successfully matched all {matched_count} weight tensors")

        self.model.load_state_dict(aligned_state, strict=True)


def create_model(
    model_cfg: ModelConfig,
    weight_path: Union[bool, Path] = True,
    class_num: int = 80,
    weight_key: str = "state_dict",
    strict: bool = False,
) -> nn.Module:
    """Factory function to construct a model from registry or default ConfigModel."""
    if isinstance(model_cfg, (OmegaConf, ListConfig)):
        OmegaConf.set_struct(model_cfg, False)

    model = _init_model_instance(model_cfg, class_num)
    _load_model_weights(model, model_cfg, weight_path, weight_key, strict)
    model = _compile_model(model, model_cfg)
    return model


def _init_model_instance(model_cfg: ModelConfig, class_num: int) -> nn.Module:
    """Internal: Handle registry lookup and model instantiation."""
    model_type = getattr(model_cfg, "type", "ConfigModel")
    if model_type not in MODELS:
        logger.warning(f"⚠️ Model type '{model_type}' not found in registry, fallback to ConfigModel")
        model_type = "ConfigModel"
    return MODELS[model_type](model_cfg, class_num)


def _load_model_weights(
    model: nn.Module, model_cfg: ModelConfig, weight_path: Union[bool, Path], weight_key: str, strict: bool
):
    """Internal: Handle weight path resolution, downloading, and loading."""
    if not weight_path:
        logger.info(":white_check_mark: Success load model (no weights)")
        return

    if weight_path is True:
        weight_path = Path("weights") / f"{model_cfg.name}.pt"
    elif isinstance(weight_path, str):
        weight_path = Path(weight_path)

    if not weight_path.exists():
        logger.info(f"🌐 Weight {weight_path} not found, try downloading")
        prepare_weight(weight_path=weight_path)

    if weight_path.exists():
        model.save_load_weights(weight_path, weight_key=weight_key, strict=strict)
        logger.info(":white_check_mark: Success load model & weight")


def _compile_model(model: nn.Module, model_cfg: ModelConfig) -> nn.Module:
    """Internal: Handle torch.compile optimization."""
    if not (model_cfg.compile and model_cfg.compile.enabled):
        return model

    if hasattr(torch, "compile"):
        logger.info(f"⚡ Compiling model (mode={model_cfg.compile.mode})")
        return torch.compile(
            model,
            mode=model_cfg.compile.mode,
            fullgraph=model_cfg.compile.fullgraph,
            dynamic=model_cfg.compile.dynamic,
            backend=model_cfg.compile.backend,
        )
    logger.warning("⚠️ torch.compile not available.")
    return model
