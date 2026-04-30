from pathlib import Path
from typing import Any, List, Tuple

import torch
from rich.progress import track
from torch import Tensor

from yolo.data.datasets import DATASETS
from yolo.data.datasets.base import DetectionDataset, SegmentationDataset
from yolo.data.helper import create_image_metadata, scale_segmentation
from yolo.utils.logger import logger


@DATASETS.register_module(name="detect_json")
class COCODetectionDataset(DetectionDataset):
    """Dataset for COCO-format detection labels (.json)."""

    def load_valid_labels(self, dataset_path: Path, phase_name: str) -> List[Tuple[Path, Tensor, float]]:
        json_path = Path(phase_name)
        if json_path.is_file():
            img_dir = json_path.stem.replace("instances_", "")
        else:
            json_path = dataset_path / "annotations" / f"instances_{phase_name}.json"
            img_dir = phase_name

        annotations_index, image_info_dict = create_image_metadata(json_path)
        data = []
        # Sort image names for determinism
        sorted_image_stems = sorted(image_info_dict.keys())

        for stem in track(sorted_image_stems, description="Filtering"):
            info = image_info_dict[stem]
            img_path = dataset_path / "images" / img_dir / info["file_name"]
            if not img_path.exists():
                continue

            annos = annotations_index.get(info["id"], [])
            bboxes = []
            for anno in annos:
                x, y, w, h = anno["bbox"]
                # Convert [x,y,w,h] to [class, x1, y1, x2, y2] normalized
                bboxes.append(
                    [
                        anno["category_id"],
                        x / info["width"],
                        y / info["height"],
                        (x + w) / info["width"],
                        (y + h) / info["height"],
                    ]
                )
            data.append((img_path, torch.tensor(bboxes).reshape(-1, 5), info["width"] / info["height"]))

        return sorted(data, key=lambda x: x[2], reverse=True)


@DATASETS.register_module(name="segment_json")
class COCOSegmentationDataset(SegmentationDataset):
    """Dataset for COCO-format segmentation labels (.json)."""

    def load_valid_labels(self, dataset_path: Path, phase_name: str) -> List[Tuple[Path, List[Tensor], float]]:
        json_path = Path(phase_name)
        if json_path.is_file():
            img_dir = json_path.stem.replace("instances_", "")
        else:
            json_path = dataset_path / "annotations" / f"instances_{phase_name}.json"
            img_dir = phase_name

        annotations_index, image_info_dict = create_image_metadata(json_path)
        data = []
        sorted_image_stems = sorted(image_info_dict.keys())

        for stem in track(sorted_image_stems, description="Filtering"):
            info = image_info_dict[stem]
            img_path = dataset_path / "images" / img_dir / info["file_name"]
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            annos = annotations_index.get(info["id"], [])
            scaled_annos = scale_segmentation(annos, info)  # returns [ [class, x1, y1, ...], ... ]
            if scaled_annos:
                scaled_annos = [torch.tensor(poly) for poly in scaled_annos]
            data.append((img_path, scaled_annos, info["width"] / info["height"]))

        return sorted(data, key=lambda x: x[2], reverse=True)
