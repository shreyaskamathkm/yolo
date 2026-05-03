# Dataset Architecture

The `yolo` repository uses a modular, inheritance-based architecture for data loading to support multiple tasks (Detection, Segmentation, Pose) and multiple label formats (COCO JSON, YOLO TXT) seamlessly.

## Data Structures

We use structured dataclasses to pass data through the pipeline, ensuring type safety and clarity.

- **`Sample`**: Represents a single image and its associated labels (bboxes, masks, keypoints).
- **`Batch`**: Represents a collated batch of samples. Data is accessed via attributes (e.g., `batch.images`, `batch.targets`).

## Class Hierarchy

All datasets inherit from a common base to share logic for image loading, caching, and dynamic shape management.

```mermaid
graph TD
    BaseDataset["BaseDataset (Abstract)"] --> DetectionDataset["DetectionDataset"]
    BaseDataset --> SegmentationDataset["SegmentationDataset"]

    DetectionDataset --> YOLODetectionDataset["YOLODetectionDataset (TXT)"]
    DetectionDataset --> COCODetectionDataset["COCODetectionDataset (JSON)"]

    SegmentationDataset --> YOLOSegmentationDataset["YOLOSegmentationDataset (TXT)"]
    SegmentationDataset --> COCOSegmentationDataset["COCOSegmentationDataset (JSON)"]
```

### 1. `BaseDataset`
Handles core infrastructure:
- Image loading and resizing.
- Thread-safe caching of labels.
- Dynamic image size management (for multi-scale training).
- Integration with `AugmentationComposer`.
- Provides `get_more_data(n)` standard interface to fetch additional samples for multi-image transforms (like `Mosaic` and `MixUp`).

### 2. Task-Specific Datasets (`DetectionDataset`, `SegmentationDataset`)
Define how labels are structured for a particular computer vision task:
- `DetectionDataset`: Returns `[class, x1, y1, x2, y2]` boxes.
- `SegmentationDataset`: Returns both boxes and masks (polygons).

### 3. Format-Specific Datasets (`COCO...`, `YOLO...`)
Implement the actual parsing of label files:
- **COCO**: Parses JSON annotations using an internal index.
- **YOLO**: Parses `.txt` files where each line follows the task-specific YOLO format.

## Usage

You should rarely need to instantiate these classes directly. Instead, use the `create_dataloader` factory:

```python
from yolo.data.loader import create_dataloader

# For Detection
dataloader = create_dataloader(data_cfg, dataset_cfg, task="train")

# For Segmentation
dataloader = create_dataloader(data_cfg, dataset_cfg, task="segmentation")
```

## Adding a New Task

To add a new task (e.g., Pose Estimation):
1. Create `PoseDataset(BaseDataset)` in `yolo/data/dataset.py`.
2. Implement `__getitem__` to return `Sample(..., keypoints=...)`.
3. Create format-specific subclasses (e.g., `COCOPoseDataset`).
4. Update the factory in `yolo/data/loader.py`.
