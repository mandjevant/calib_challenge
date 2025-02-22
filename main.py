import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Callable,Tuple
from torch.utils.data import Dataset
import torch


class LabelsToNP:
    """
    Read labels from .txt, convert to np.ndarray and save to .npy
    """

    def __init__(self, file_name: str):
        """
        Parameters
        ----------
        file_name: str
            File name of .hevc to convert
        """
        if file_name[-5:] == ".hevc":
            self.file_name: str = file_name[:-5]
        else:
            self.file_name = file_name

    def read_from_txt_convert_to_np(self) -> np.ndarray:
        """
        Read each row in the .txt, split on space and convert to array

        Returns
        -----------
        np.ndarray
            Array with labels
        """
        labels = list()
        with open(f"labeled/{self.file_name}.txt", "r") as f:
            for row in f:
                labels.append(np.array(list(map(float, row.split(" ")))))
        
        return np.array(labels)
          
    def save_as_npy(self, arr: np.ndarray) -> None:
        """
        Save the array

        Parameters
        ----------
        arr: np.ndarray
            Array to save
        """
        np.save(f"labeled/{self.file_name}.npy", arr)
    
    def main(self) -> None:
        """
        Verify if the .npy version of the labels already exists
         if so, return. If not, get the labels, convert and save
        """
        if os.path.isfile(f"{self.file_name}.npy"):
            return
        
        labels = self.read_from_txt_convert_to_np()
        self.save_as_npy(labels)
    

class HEVCDataloader(Dataset):
    """
    HEVC Dataloader class. 
     Iterates over frames in .hevc over list of .hevc files
     For each .hevc finds the corresponding labels
     Creates and returns batches of size batch_size
    """

    def __init__(self, file_names: List[str], batch_size: int=8, transform: Callable=None):
        """
        Parameters
        ----------
        file_names: List[str]
            list of .hevc file names
        batch_size: int
            desired batch size
        transform: Callable
            transformation to apply
        """
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

    def _open_next_hevc(self) -> None:
        """
        Open the next .hevc file and set self.capture to read the frames
        Subsequently, find and set the labels for this .hevc
        """
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
        """
        Convert the object to an iterator (similar to pytorch's DataLoader)
        """
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets the next batch data by reading self.capture for self.batch_size frames
        Gets the batch labels by indexing the self.video_labels array
        Transforms the batches to conform by pytorch input (batch_size, channels, height, width)

        Returns
        ----------
        Tuple[np.ndarray, np.ndarray]
            Array of batch data, Array of batch labels 
        """
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
        batch_labels = torch.tensor(np.array(batch_labels)).long()

        if len(batch_labels) < self.batch_size:
            raise StopIteration

        return batch_frames, batch_labels


class Net(nn.Module):
    """
    Simple CNN with downsampling and dropout.
    Supports input of (B, 3, 874, 1164) and gives output (B, 2)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 55 * 73, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        ----------
        torch.Tensor
            output tensor
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x


def _test(file_paths: List[str] = ["0.hevc"]) -> None:
    """
    Validate the dataloader and net by checking input and output sizes

    Parameters
    ----------
    file_paths: List[str]
        List of file paths
    """
    dataloader = HEVCDataloader(file_paths)
    CNN_model = Net()
    
    for batch_input, batch_labels in dataloader:
        break
    
    assert batch_input.size()[0]==batch_labels.size()[0], "Dataloader sizes are off."
    assert batch_input.size() == torch.Size([batch_input.size()[0], 3, 874, 1164]), "Dataloader input size is off."
    assert batch_labels.size() == torch.Size([batch_input.size()[0], 2]), "Dataloader label size is off."
    input = torch.randn(8, 3, 874, 1164)
    output = CNN_model(input)
    assert output.size() == torch.Size([8, 2]), "Output size is off."
    input = torch.randn(4, 3, 874, 1164)
    output = CNN_model(input)
    assert output.size() == torch.Size([4, 2]), "Output size is off."


if __name__ == "__main__":
    _test()

    file_paths = ["0.hevc", "1.hevc", "2.hevc", "3.hevc", "4.hevc"]
    dataloader = HEVCDataloader(file_paths)
    CNN_model = Net()
    
    for batch_input, batch_labels in dataloader:
        break
    