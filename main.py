import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Callable
from torch.utils.data import Dataset, DataLoader
import torch


class LabelsToNP:
    def __init__(self, file_name: str):
        self.file_name: str = file_name[:-5]

    def read_from_txt_convert_to_np(self) -> np.ndarray:
        labels = list()
        with open(f"labeled/{self.file_name}.txt", "r") as f:
            for row in f:
                labels.append(np.array(list(map(float, row.split(" ")))))
        
        return np.array(labels)
          
    def save_as_npy(self, arr: np.ndarray) -> None:
        np.save(f"labeled/{self.file_name}.npy", arr)
    
    def main(self) -> np.ndarray:
        if os.path.isfile(f"{self.file_name}.npy"):
            return
        
        labels = self.read_from_txt_convert_to_np()
        self.save_as_npy(labels)
    

class HEVCDataloader(Dataset):
    def __init__(self, file_names: List[str], batch_size: int=8, transform: Callable=None):
        self.file_names: List[str] = file_names
        self.batch_size: int = batch_size
        self.transform: Callable = transform

        self.video_idx: int = 0
        self.frame_idx: int = 0
        self.capture: cv2.VideoCapture = None
        self.video_labels: np.ndarray = None

        for file_name in self.file_names:
            LabelsToNP(file_name).main()

        self._open_next_hevc()

    def _open_next_hevc(self):
        if self.capture is not None:
            self.capture.release()
        
        if self.video_idx >= len(self.file_names):
            self.capture = None
            return
        
        video_path = f"labeled/{self.file_names[self.video_idx]}"
        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.video_labels = np.load(f"{video_path.replace('.hevc', '.npy')}")
        self.video_idx += 1
        self.frame_idx = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.capture is None:
            raise StopIteration
        
        batch_frames = list()
        batch_labels = list() 

        while len(batch_frames) < self.batch_size:
            ret, frame = self.capture.read()
            if not ret:
                self._open_next_hevc()
                if self.capture is None:
                    break
                continue

            if self.transform:
                frame = self.transform(frame)

            batch_frames.append(frame)
            batch_labels.append(self.video_labels[self.frame_idx])
            self.frame_idx += 1
        
        if len(batch_frames) == 0:
            raise StopIteration
        
        batch_frames = np.array(batch_frames)
        batch_frames = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float()
        batch_labels = torch.tensor(batch_labels).long()

        if len(batch_labels) < self.batch_size:
            raise StopIteration

        return batch_frames, batch_labels


if __name__ == "__main__":
    file_paths = ["0.hevc", "1.hevc", "2.hevc", "3.hevc", "4.hevc"]
    dataloader = HEVCDataloader(file_paths)
