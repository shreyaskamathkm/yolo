# Merged from: tools/format_converters.py + tools/data_conversion.py
# TODO Phase 2: update solver import path (yolo.tools.solver -> yolo.training.solver)
import json
from pathlib import Path
from typing import Dict, List, Optional

from rich.progress import track

# ─── Weight conversion helpers ──────────────────────────────────────────────

convert_dict = {
    "19.cv1": "19.conv",
    "16.cv1": "16.conv",
    ".7.cv1": ".7.conv",
    ".5.cv1": ".5.conv",
    ".3.cv1": ".3.conv",
    ".28.": ".29.",
    ".25.": ".26.",
    ".22.": ".23.",
    "cv": "conv",
    ".m.": ".bottleneck.",
}

HEAD_NUM = "29"


def convert_weight(old_state_dict, new_state_dict, model_size: int = 38):
    new_weight_set = set(new_state_dict.keys())
    for weight_name, weight_value in old_state_dict.items():
        if HEAD_NUM in weight_name:
            _, _, conv_name, conv_id, *post_fix = weight_name.split(".")
            head_id = 30 if conv_name in ["cv2", "cv3"] else 22
            head_type = "anchor_conv" if conv_name in ["cv2", "cv4"] else "class_conv"
            weight_name = ".".join(["model", str(head_id), "heads", conv_id, head_type, *post_fix])
        else:
            for old_name, new_name in convert_dict.items():
                if old_name in weight_name:
                    weight_name = weight_name.replace(old_name, new_name)
        if weight_name in new_weight_set:
            assert new_state_dict[weight_name].shape == weight_value.shape, f"shape miss match {weight_name}"
            new_state_dict[weight_name] = weight_value
            new_weight_set.remove(weight_name)
    return new_state_dict


head_converter = {
    "head_conv": "m",
    "implicit_a": "ia",
    "implicit_m": "im",
}

SPP_converter = {
    "pre_conv.0": "cv1",
    "pre_conv.1": "cv3",
    "pre_conv.2": "cv4",
    "post_conv.0": "cv5",
    "post_conv.1": "cv6",
    "short_conv": "cv2",
    "merge_conv": "cv7",
}

REP_converter = {"conv1": "rbr_dense", "conv2": "rbr_1x1", "conv": "0", "bn": "1"}


def convert_weight_v7(old_state_dict, new_state_dict):
    for key_name in new_state_dict.keys():
        new_shape = new_state_dict[key_name].shape
        old_key_name = "model." + key_name
        if old_key_name not in old_state_dict.keys():
            if "heads" in key_name:
                layer_idx, _, conv_idx, conv_name, *details = key_name.split(".")
                old_key_name = ".".join(["model", str(layer_idx), head_converter[conv_name], conv_idx, *details])
            elif any(k in key_name for k in SPP_converter):
                for key, value in SPP_converter.items():
                    if key in key_name:
                        key_name = key_name.replace(key, value)
                old_key_name = "model." + key_name
            elif "conv1" in key_name or "conv2" in key_name:
                for key, value in REP_converter.items():
                    if key in key_name:
                        key_name = key_name.replace(key, value)
                old_key_name = "model." + key_name
        assert old_key_name in old_state_dict.keys(), f"Weight Name Mismatch!! {old_key_name}"
        old_shape = old_state_dict[old_key_name].shape
        assert new_shape == old_shape, f"Weight Shape Mismatch!! {old_key_name}"
        new_state_dict[key_name] = old_state_dict[old_key_name]
    return new_state_dict


replace_dict = {"cv": "conv", ".m.": ".bottleneck."}


def convert_weight_seg(old_state_dict, new_state_dict):
    diff = -1
    for old_weight_name in old_state_dict.keys():
        old_idx = int(old_weight_name.split(".")[1])
        if old_idx == 23:
            diff = 3
        elif old_idx == 41:
            diff = -19
        new_idx = old_idx + diff
        new_weight_name = old_weight_name.replace(f".{old_idx}.", f".{new_idx}.")
        for key, val in replace_dict.items():
            new_weight_name = new_weight_name.replace(key, val)

        if new_weight_name not in new_state_dict.keys():
            heads = "heads"
            _, _, conv_name, conv_idx, *details = old_weight_name.split(".")
            if "proto" in conv_name or "dfl" in old_weight_name:
                continue
            if conv_name in ("cv2", "cv3", "cv6"):
                layer_idx = 44
                heads = "detect.heads"
            if conv_name in ("cv4", "cv5", "cv7"):
                layer_idx = 25
                heads = "detect.heads"

            if conv_name in ("cv2", "cv4"):
                conv_task = "anchor_conv"
            elif conv_name in ("cv3", "cv5"):
                conv_task = "class_conv"
            elif conv_name in ("cv6", "cv7"):
                conv_task = "mask_conv"
                heads = "heads"
            else:
                continue

            new_weight_name = ".".join(["model", str(layer_idx), heads, conv_idx, conv_task, *details])

        if (
            new_weight_name not in new_state_dict.keys()
            or new_state_dict[new_weight_name].shape != old_state_dict[old_weight_name].shape
        ):
            print(f"new: {new_weight_name}, old: {old_weight_name}")
        new_state_dict[new_weight_name] = old_state_dict[old_weight_name]
    return new_state_dict


# ─── Annotation conversion helpers ──────────────────────────────────────────


def discretize_categories(categories: List[Dict]) -> Dict[int, int]:
    """Maps each category id to a sequential integer index."""
    sorted_categories = sorted(categories, key=lambda c: c["id"])
    return {c["id"]: idx for idx, c in enumerate(sorted_categories)}


def normalize_segmentation(segmentation: List[float], img_width: int, img_height: int) -> List[str]:
    return [
        f"{coord / img_width:.6f}" if i % 2 == 0 else f"{coord / img_height:.6f}"
        for i, coord in enumerate(segmentation)
    ]


def process_annotation(annotation: Dict, image_dims: tuple, id_to_idx: Optional[Dict[int, int]], file) -> None:
    category_id = annotation["category_id"]
    segmentation = (
        annotation["segmentation"][0]
        if annotation["segmentation"] and isinstance(annotation["segmentation"][0], list)
        else None
    )
    if segmentation is None:
        return
    img_width, img_height = image_dims
    normalized = normalize_segmentation(segmentation, img_width, img_height)
    if id_to_idx:
        category_id = id_to_idx.get(category_id, category_id)
    file.write(f"{category_id} {' '.join(normalized)}\n")


def process_annotations(
    image_annotations: Dict[int, List[Dict]],
    image_info_dict: Dict[int, tuple],
    output_dir: Path,
    id_to_idx: Optional[Dict[int, int]] = None,
) -> None:
    for image_id, annotations in track(image_annotations.items(), description="Processing annotations"):
        file_path = output_dir / f"{image_id:0>12}.txt"
        if not annotations:
            continue
        with open(file_path, "w") as file:
            for annotation in annotations:
                process_annotation(annotation, image_info_dict[image_id], id_to_idx, file)


def convert_annotations(json_file: str, output_dir: str) -> None:
    with open(json_file) as file:
        data = json.load(file)
    Path(output_dir).mkdir(exist_ok=True)
    image_info_dict = {img["id"]: (img["width"], img["height"]) for img in data.get("images", [])}
    id_to_idx = discretize_categories(data.get("categories", [])) if "categories" in data else None
    image_annotations = {img_id: [] for img_id in image_info_dict}
    for annotation in data.get("annotations", []):
        if not annotation.get("iscrowd", False):
            image_annotations[annotation["image_id"]].append(annotation)
    process_annotations(image_annotations, image_info_dict, Path(output_dir), id_to_idx)
