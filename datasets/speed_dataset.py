import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import Any


class SpeedDataset(Dataset):
    """Dataset for loading video frames and speed labels"""
    def __init__(
        self, 
        video_path: str, 
        speed_file: str, 
        num_frames: int = 8, 
        frame_size: tuple[int, int] = (256, 256) # base for the model
    ) -> None:
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        self.speeds = np.loadtxt(speed_file)

        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.valid_indices = list(range(0, self.total_frames - num_frames + 1))
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_frame = self.valid_indices[idx]
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_list: list[NDArray[Any]] = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frame_list.append(frame)
        
        cap.release()
        
        frames: NDArray[np.float32] = np.array(frame_list).astype(np.float32) / 255.0
        # ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std
        
        # [T, H, W, C] -> [T, C, H, W]
        frames = frames.transpose(0, 3, 1, 2)
    
        speed_idx = start_frame + self.num_frames // 2
        speed = self.speeds[min(speed_idx, len(self.speeds) - 1)]
        
        return torch.FloatTensor(frames), torch.FloatTensor([speed])