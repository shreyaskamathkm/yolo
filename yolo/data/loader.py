from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generator, List

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


class StreamDataLoader:
    def __init__(self, data_cfg: DataConfig):
        self.source = data_cfg.source
        self.running = True
        source_str = str(self.source)
        self.is_stream = (
            isinstance(self.source, int)
            or source_str.lower().startswith(("rtmp://", "rtsp://", "http://", "https://"))
            or (Path(source_str).is_file() and filetype.is_video(source_str))
        )
        self.path = (
            Path(source_str)
            if not isinstance(self.source, int) and (Path(source_str).is_file() or Path(source_str).is_dir())
            else self.source
        )

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()
        self.total_frames = self._count_total_frames()

        if self.is_stream:
            self.cap = cv2.VideoCapture(self.source if isinstance(self.source, int) else source_str)
            if not self.cap.isOpened():
                raise ValueError(f"Error opening source: {self.source}")
            print(f"✅ Streaming from: {self.source}")
        else:
            self.source = Path(self.source)
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.daemon = True
            self.thread.start()
            print(f"✅ Loading from: {self.source} ({self.total_frames} frames found)")

    def _count_total_frames(self):
        count = 0
        source_path = Path(str(self.source))
        if source_path.is_dir():
            for p in source_path.rglob("*"):
                if not p.is_file():
                    continue
                if filetype.is_image(p):
                    count += 1
                elif filetype.is_video(p):
                    cap = cv2.VideoCapture(str(p))
                    c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    count += c if c > 0 else 0
                    cap.release()
        elif self.is_stream:
            if not isinstance(self.source, int) and Path(self.source).is_file():
                cap = cv2.VideoCapture(str(self.source))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                count = 1000000  # Placeholder for live streams
        elif filetype.is_image(source_path):
            count = 1
        return count

    def load_source(self):
        if self.source.is_dir():  # image folder
            self.load_folder(self.source)
        elif filetype.is_video(self.source):  # Video file
            self.load_video_file(self.source)
        elif filetype.is_image(self.source):  # Single image
            self.process_image(self.source)

    def load_folder(self, folder):
        folder_path = Path(folder)
        for file_path in folder_path.rglob("*"):
            if self.stop_event.is_set():
                break
            if not file_path.is_file():
                continue
            if filetype.is_image(file_path):
                self.process_image(file_path)
            elif filetype.is_video(file_path):
                self.load_video_file(file_path)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        self.process_frame(image, path=image_path)

    def load_video_file(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, path=video_path)
        cap.release()

    def process_frame(self, frame, path=None):
        if isinstance(frame, np.ndarray):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        origin_frame = frame
        frame, _, rev_tensor = self.transform(frame, torch.zeros(0, 5))
        frame = frame[None]
        rev_tensor = rev_tensor[None]
        if not self.is_stream:
            self.queue.put((frame, rev_tensor, origin_frame, path))
        else:
            self.current_frame = (frame, rev_tensor, origin_frame, path)

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tensor:
        if self.is_stream:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                raise StopIteration
            self.process_frame(frame, path=self.path)
            return self.current_frame
        else:
            try:
                frame = self.queue.get(timeout=1)
                return frame
            except Empty:
                raise StopIteration

    def stop(self):
        self.running = False
        if self.is_stream:
            self.cap.release()
        else:
            self.thread.join(timeout=1)

    def __len__(self):
        return self.total_frames


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
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
