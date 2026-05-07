import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Tuple

import torch
from rich.progress import track
from torch import Tensor

from yolo.data.annotations import create_image_metadata, scale_segmentation
from yolo.data.base.detect import DetectionDataset
from yolo.data.base.segment import SegmentationDataset
from yolo.data.schema import TrainerTaskType
from yolo.registry import DATASETS

logger = logging.getLogger(__name__)

MAX_WORKERS = 1


@DATASETS.register_module(name=(TrainerTaskType.DETECTION, "coco"))
class COCODetectionDataset(DetectionDataset):
    """Dataset for COCO-format detection labels (.json)."""

    def load_valid_labels(self, dataset_path: Path, phase_path: str) -> List[Tuple[Path, Tensor, float]]:
        # 1. Determine JSON path
        phase_path_obj = Path(phase_path)
        if phase_path_obj.is_file():
            json_path = phase_path_obj
        elif (dataset_path / phase_path).is_file():
            json_path = dataset_path / phase_path
        elif (dataset_path / phase_path_obj.name).is_file():
            json_path = dataset_path / phase_path_obj.name
        else:
            # Try standard COCO naming: annotations/instances_{phase_path}.json
            # and also try stripping potential prefixes from phase_path
            clean_phase = phase_path_obj.stem.replace("instances_", "")
            json_path = dataset_path / "annotations" / f"instances_{clean_phase}.json"
            if not json_path.is_file():
                # Try as a relative path to dataset_path but only the tail part
                json_path = dataset_path / phase_path_obj.name
                if not json_path.is_file():
                    raise FileNotFoundError(
                        f"Could not find COCO annotations for phase '{phase_path}' in '{dataset_path}'"
                    )

        # 2. Determine image directory
        # If the JSON filename is instances_{NAME}.json, then NAME is our img_dir
        if "instances_" in json_path.stem:
            img_dir = json_path.stem.replace("instances_", "")
        else:
            # Fallback to the stem of the JSON file
            img_dir = json_path.stem

        annotations_index, image_info_dict = create_image_metadata(json_path)
        data = []
        # Sort image names for determinism
        sorted_image_stems = sorted(image_info_dict.keys())

        def process_info(stem):
            info = image_info_dict[stem]
            img_path = dataset_path / "images" / img_dir / info["file_name"]
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                return None

            annos = annotations_index.get(info["id"], [])
            bboxes = []
            for anno in annos:
                x, y, w, h = anno["bbox"]
                bboxes.append(
                    [
                        anno["category_id"],
                        x / info["width"],
                        y / info["height"],
                        (x + w) / info["width"],
                        (y + h) / info["height"],
                    ]
                )
            return (img_path, torch.tensor(bboxes).reshape(-1, 5), info["width"] / info["height"])

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(
                track(
                    executor.map(process_info, sorted_image_stems),
                    total=len(sorted_image_stems),
                    description="Filtering",
                )
            )

        data = [r for r in results if r is not None]

        return sorted(data, key=lambda x: x[2], reverse=True)


@DATASETS.register_module(name=(TrainerTaskType.SEGMENTATION, "coco"))
class COCOSegmentationDataset(SegmentationDataset):
    """Dataset for COCO-format segmentation labels (.json)."""

    def load_valid_labels(self, dataset_path: Path, phase_path: str) -> List[Tuple[Path, List[Tensor], float]]:
        # 1. Determine JSON path
        phase_path_obj = Path(phase_path)
        if phase_path_obj.is_file():
            json_path = phase_path_obj
        elif (dataset_path / phase_path).is_file():
            json_path = dataset_path / phase_path
        elif (dataset_path / phase_path_obj.name).is_file():
            json_path = dataset_path / phase_path_obj.name
        else:
            # Try standard COCO naming: annotations/instances_{phase_path}.json
            clean_phase = phase_path_obj.stem.replace("instances_", "")
            json_path = dataset_path / "annotations" / f"instances_{clean_phase}.json"
            if not json_path.is_file():
                json_path = dataset_path / phase_path_obj.name
                if not json_path.is_file():
                    raise FileNotFoundError(
                        f"Could not find COCO annotations for phase '{phase_path}' in '{dataset_path}'"
                    )

        # 2. Determine image directory
        if "instances_" in json_path.stem:
            img_dir = json_path.stem.replace("instances_", "")
        else:
            img_dir = json_path.stem

        annotations_index, image_info_dict = create_image_metadata(json_path)
        data = []
        sorted_image_stems = sorted(image_info_dict.keys())

        def process_info(stem):
            info = image_info_dict[stem]
            img_path = dataset_path / "images" / img_dir / info["file_name"]
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                return None

            annos = annotations_index.get(info["id"], [])
            scaled_annos = scale_segmentation(annos, info)  # returns [ [class, x1, y1, ...], ... ]
            if scaled_annos:
                scaled_annos = [torch.tensor(poly) for poly in scaled_annos]
            return (img_path, scaled_annos, info["width"] / info["height"])

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(
                track(
                    executor.map(process_info, sorted_image_stems),
                    total=len(sorted_image_stems),
                    description="Filtering",
                )
            )

        data = [r for r in results if r is not None]

        return sorted(data, key=lambda x: x[2], reverse=True)
