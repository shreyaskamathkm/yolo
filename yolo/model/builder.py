from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import ListConfig, OmegaConf
from torch import nn

from yolo.config.config import ModelConfig, YOLOLayer
from yolo.data.preparation import prepare_weight
from yolo.model import blocks
from yolo.registry import BLOCKS
from yolo.utils.logger import logger


class YOLO(nn.Module):
    """The core YOLO model class that assembles layers from a configuration.

    This class dynamically builds the neural network based on the architecture
    defined in a YAML configuration file. It handles layer instantiation,
    source indexing for skip-connections, and weights management.

    Attributes:
        num_classes (int): Number of detection classes.
        model (nn.ModuleList): The sequential list of layers comprising the model.
        reg_max (int): Maximum regression distance for box predictions.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        """Initializes the YOLO model.

        Args:
            model_cfg (ModelConfig): Architecture and anchor configuration.
            class_num (int, optional): Number of output classes. Defaults to 80.
        """
        super(YOLO, self).__init__()

        self.num_classes = class_num
        self.model: List[YOLOLayer] = nn.ModuleList()
        self.reg_max = getattr(model_cfg.anchor, "reg_max", 16)
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        """Assembles the model from an architecture specification.

        Args:
            model_arch (Dict): Dictionary containing backbone, neck, and head specs.

        Raises:
            ValueError: If a duplicate tag is found in the architecture.
        """
        self.layer_index = {}
        output_dim, layer_idx = [3], 1
        logger.info(f":tractor: Building YOLO")
        for arch_name in model_arch:
            if model_arch[arch_name]:
                logger.info(f"  :building_construction:  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=layer_idx):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})

                # Get input source
                source = self.get_source_idx(layer_info.get("source", -1), layer_idx)

                # Find in channels
                if any(module in layer_type for module in ["Conv", "ELAN", "ADown", "AConv", "CBLinear"]):
                    layer_args["in_channels"] = output_dim[source]
                if any(module in layer_type for module in ["Detection", "Segmentation", "Classification"]):
                    if isinstance(source, list):
                        layer_args["in_channels"] = [output_dim[idx] for idx in source]
                    else:
                        layer_args["in_channel"] = output_dim[source]
                    layer_args["num_classes"] = self.num_classes
                    layer_args["reg_max"] = self.reg_max

                # create layers
                layer = self.create_layer(layer_type, source, layer_info, **layer_args)
                self.model.append(layer)

                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(f"Duplicate tag '{layer_info['tags']}' found.")
                    self.layer_index[layer.tags] = layer_idx

                out_channels = self.get_out_channels(layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)
            layer_idx += 1

    def forward(
        self, x: torch.Tensor, external: Optional[Dict] = None, shortcut: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Performs a forward pass.

        Args:
            x (Tensor): Input image tensor.
            external (Optional[Dict]): External feature maps for skip-connections.
            shortcut (Optional[str]): If provided, returns the output of this specific tag early.

        Returns:
            Dict[str, Tensor]: Map of layer tags to their respective output tensors.
        """
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
        self, layer_type: str, layer_args: dict, output_dim: List[int], source: Union[int, List[int]]
    ) -> int:
        """Calculates the number of output channels for a layer.

        Args:
            layer_type (str): The type of the layer.
            layer_args (dict): Arguments passed to the layer.
            output_dim (list): List of output channels for all previous layers.
            source (Union[int, list]): The index or indices of input layers.

        Returns:
            int: The calculated number of output channels.
        """
        if hasattr(layer_args, "out_channels"):
            return layer_args["out_channels"]
        if layer_type == "CBFuse":
            return output_dim[source[-1]]
        if isinstance(source, int):
            return output_dim[source]
        if isinstance(source, list):
            return sum(output_dim[idx] for idx in source)

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int) -> Union[int, List[int]]:
        """Resolves the source index for a layer.

        Handles relative indices, named tags, and ListConfig.

        Args:
            source (Union[ListConfig, str, int]): The raw source identifier.
            layer_idx (int): The index of the current layer.

        Returns:
            Union[int, list]: The resolved absolute index or list of indices.
        """
        if isinstance(source, ListConfig):
            return [self.get_source_idx(index, layer_idx) for index in source]
        if isinstance(source, str):
            source = self.layer_index[source]
        if source < -1:
            source += layer_idx
        if source > 0:  # Using Previous Layer's Output
            self.model[source - 1].usable = True
        return source

    def create_layer(
        self, layer_type: str, source: Union[int, List[int]], layer_info: Dict, **kwargs: Any
    ) -> YOLOLayer:
        """Instantiates a layer from the registry.

        Args:
            layer_type (str): The registered name of the block.
            source (Union[int, list]): Source layer indices.
            layer_info (Dict): Configuration dictionary for the layer.
            **kwargs: Arguments for the layer's constructor.

        Returns:
            YOLOLayer: The instantiated layer.

        Raises:
            ValueError: If the layer_type is not registered.
        """
        if layer_type in BLOCKS:
            layer = BLOCKS[layer_type](**kwargs)
            setattr(layer, "layer_type", layer_type)
            setattr(layer, "source", source)
            setattr(layer, "in_c", kwargs.get("in_channels", None))
            setattr(layer, "output", layer_info.get("output", False))
            setattr(layer, "tags", layer_info.get("tags", None))
            setattr(layer, "external", layer_info.get("external", []))
            setattr(layer, "usable", 0)
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def save_load_weights(self, weights: Union[Path, dict], weight_key: str = "state_dict"):
        """Loads weights into the model with robust key matching.

        Args:
            weights (Union[Path, dict]): Path to a weights file or a state_dict.
            weight_key (str, optional): Key to look for in the weights file.
                Defaults to "state_dict".
        """
        if isinstance(weights, (str, Path)):
            weights = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)

        if weight_key in weights:
            loaded_dict = weights[weight_key]
        elif "state_dict" in weights:
            logger.warning(f"⚠️ Key '{weight_key}' not found, falling back to 'state_dict'")
            loaded_dict = weights["state_dict"]
        else:
            loaded_dict = weights

        from yolo.utils.module_utils import clean_state_dict

        loaded_dict = clean_state_dict(loaded_dict)
        new_state_dict = self.model.state_dict()

        matched_keys = 0
        for model_key in new_state_dict.keys():
            search_keys = [model_key, f"model.{model_key}", f"model.model.{model_key}"]
            found = False
            for k in search_keys:
                if k in loaded_dict:
                    if new_state_dict[model_key].shape == loaded_dict[k].shape:
                        new_state_dict[model_key] = loaded_dict[k]
                        matched_keys += 1
                        found = True
                        break
                    else:
                        logger.warning(
                            f"⚠️ Shape mismatch for {model_key}: "
                            f"expected {new_state_dict[model_key].shape}, "
                            f"got {loaded_dict[k].shape}"
                        )
            if not found:
                logger.debug(f"ℹ️ Layer {model_key} not found in loaded weights")

        if matched_keys == 0:
            logger.error("❌ No weights were matched!")
        else:
            logger.info(f"✅ Successfully matched {matched_keys}/{len(new_state_dict)} weight tensors")

        self.model.load_state_dict(new_state_dict, strict=True)


def create_model(
    model_cfg: ModelConfig, weight_path: Union[bool, Path] = True, class_num: int = 80, weight_key: str = "state_dict"
) -> YOLO:
    """Constructs and returns a YOLO model.

    Args:
        model_cfg (ModelConfig): The architecture configuration.
        weight_path (Union[bool, Path], optional): Path to weights. If True,
            attempts to load default weights. Defaults to True.
        class_num (int, optional): Number of output classes. Defaults to 80.
        weight_key (str, optional): Key for weights in the file. Defaults to "state_dict".

    Returns:
        YOLO: The assembled model.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg, class_num)
    if weight_path:
        if weight_path is True:
            weight_path = Path("weights") / f"{model_cfg.name}.pt"
        elif isinstance(weight_path, str):
            weight_path = Path(weight_path)

        if not weight_path.exists():
            logger.info(f"🌐 Weight {weight_path} not found, try downloading")
            prepare_weight(weight_path=weight_path)
        if weight_path.exists():
            model.save_load_weights(weight_path, weight_key=weight_key)
            logger.info(":white_check_mark: Success load model & weight")
    else:
        logger.info(":white_check_mark: Success load model")

    if model_cfg.compile and model_cfg.compile.enabled:
        if hasattr(torch, "compile"):
            logger.info(f"⚡ Compiling model with torch.compile (mode={model_cfg.compile.mode})")
            model = torch.compile(
                model,
                mode=model_cfg.compile.mode,
                fullgraph=model_cfg.compile.fullgraph,
                dynamic=model_cfg.compile.dynamic,
                backend=model_cfg.compile.backend,
            )
        else:
            logger.warning("⚠️ torch.compile is not available.")

    return model
