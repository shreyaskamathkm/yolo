import threading
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Generator, List, Optional, Tuple

import cv2
import filetype
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from yolo.config.config import DataConfig, DatasetConfig
from yolo.data.augmentation import AugmentationComposer
from yolo.data.dataset import YoloDataset, collate_fn
from yolo.data.preparation import prepare_dataset

# Sentinel to signal end-of-source cleanly (avoids timeout-based StopIteration)
_STREAM_DONE = object()
_STREAM_SOURCE = ("rtmp://", "rtsp://", "http://", "https://")


class StreamDataLoader:
    """A threaded data loader for streaming and batch processing of various input sources.

    This loader supports images, videos, directories, and live streams (RTSP, RTMP, Webcam).
    It uses background threads to pre-process frames and maintain a queue for inference.

    Attributes:
        source (Union[str, int]): The input source (path or stream ID).
        is_stream (bool): Whether the source is a live stream or video.
        total_frames (int): Total number of frames in the source (proxied for streams).
    """

    def __init__(self, data_cfg: DataConfig):
        """Initializes the StreamDataLoader.

        Args:
            data_cfg (DataConfig): Configuration containing source, image size, etc.
        """

        self.source = data_cfg.source
        self.running = True
        source_str = str(self.source)

        self.is_stream = (
            isinstance(self.source, int)
            or source_str.lower().startswith(_STREAM_SOURCE)
            or (Path(source_str).is_file() and filetype.is_video(source_str))
        )

        if not isinstance(self.source, int):
            p = Path(source_str)
            self.path: Any = p if (p.is_file() or p.is_dir()) else self.source
        else:
            self.path = self.source

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()

        if self.is_stream:
            self.cap = cv2.VideoCapture(self.source if isinstance(self.source, int) else source_str)
            if not self.cap.isOpened():
                raise ValueError(f"Error opening source: {self.source}")
            self.total_frames = self._count_stream_frames(source_str)
            self.queue: Queue = Queue(maxsize=8)
            self._stream_thread = Thread(target=self._stream_worker, daemon=True)
            self._stream_thread.start()
            print(f"✅ Streaming from: {self.source}")
        else:
            self.source = Path(source_str)
            self.total_frames = self._count_folder_or_file_frames(self.source)
            self.queue = Queue(maxsize=32)
            self._load_thread = Thread(target=self.load_source, daemon=True)
            self._load_thread.start()
            print(f"✅ Loading from: {self.source} ({self.total_frames} frames found)")

    def _count_stream_frames(self, source_str: str) -> int:
        """Count frames for a stream source. Only opens a file once."""
        if isinstance(self.source, int):
            return 1_000_000  # Live webcam / RTSP: placeholder
        p = Path(source_str)
        if p.is_file():
            cap = cv2.VideoCapture(source_str)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count if count > 0 else 1_000_000
        return 1_000_000

    def _count_folder_or_file_frames(self, source: Path) -> int:
        """Count frames for a folder or single image/video file."""
        count = 0
        if source.is_dir():
            for p in sorted(source.rglob("*")):  # sorted for determinism
                if not p.is_file():
                    continue
                if filetype.is_image(str(p)):
                    count += 1
                elif filetype.is_video(str(p)):
                    cap = cv2.VideoCapture(str(p))
                    c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    count += max(c, 0)
        elif filetype.is_image(str(source)):
            count = 1
        elif filetype.is_video(str(source)):
            cap = cv2.VideoCapture(str(source))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return count

    def _stream_worker(self):
        """Reads frames from a live stream in a background thread."""
        try:
            while self.running and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break
                processed = self._process_frame(frame, path=self.path)
                try:
                    self.queue.put(processed, timeout=1)
                except Full:
                    continue
        finally:
            self.queue.put(_STREAM_DONE)
            self.cap.release()

    def load_source(self):
        """Entry point for the file/folder background loader thread."""
        try:
            source = self.source
            if source.is_dir():
                self._load_folder(source)
            elif filetype.is_video(str(source)):
                self._load_video_file(source)
            elif filetype.is_image(str(source)):
                self._process_and_enqueue_image(source)
        finally:
            self.queue.put(_STREAM_DONE)

    def _load_folder(self, folder: Path):
        for file_path in sorted(folder.rglob("*")):
            if self.stop_event.is_set():
                break
            if not file_path.is_file():
                continue
            if filetype.is_image(str(file_path)):
                self._process_and_enqueue_image(file_path)
            elif filetype.is_video(str(file_path)):
                self._load_video_file(file_path)

    def _process_and_enqueue_image(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        processed = self._process_frame(image, path=image_path)
        self.queue.put(processed)

    def _load_video_file(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        try:
            while self.running and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                processed = self._process_frame(frame, path=video_path)
                self.queue.put(processed)
        finally:
            cap.release()

    def _process_frame(self, frame, path: Optional[Path] = None) -> Tuple[Tensor, Tensor, Image.Image, Optional[Path]]:
        """Convert a raw frame to a processed tensor tuple. Thread-safe: no shared state."""
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        origin_frame = frame
        processed, _, rev_tensor = self.transform(frame, torch.zeros(0, 5))
        return processed[None], rev_tensor[None], origin_frame, path

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Image.Image, Optional[Path]]:
        try:
            item = self.queue.get(timeout=5)
        except Empty:
            raise StopIteration

        if item is _STREAM_DONE:
            raise StopIteration

        return item

    def stop(self):
        self.running = False
        self.stop_event.set()

        if self.is_stream:
            self._stream_thread.join(timeout=5)
        else:
            self._load_thread.join(timeout=5)

    def __len__(self) -> int:
        return self.total_frames


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
    """Factory function to create the appropriate data loader based on the task.

    For inference tasks, it returns a `StreamDataLoader`. For training and validation,
    it returns a standard PyTorch `DataLoader` wrapping the `YoloDataset`.

    Args:
        data_cfg (DataConfig): Data-specific configuration (batch size, source, etc.).
        dataset_cfg (DatasetConfig): Dataset-specific configuration (classes, paths).
        task (str, optional): The current task ('train', 'validation', or 'inference').
            Defaults to "train".

    Returns:
        Union[StreamDataLoader, DataLoader]: The requested data loader instance.
    """

    if task == "inference":
        return StreamDataLoader(data_cfg)

    if getattr(dataset_cfg, "auto_download", False):
        prepare_dataset(dataset_cfg, task)
    dataset = YoloDataset(data_cfg, dataset_cfg, task)

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.dataloader_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=data_cfg.drop_last,
    )
