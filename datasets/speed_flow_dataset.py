import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import Any
from utils.optical_flow_extractor import OpticalFlowExtractor


class SpeedDatasetWithFlow(Dataset):
    def __init__(
        self, 
        video_path: str, 
        speed_file: str, 
        num_frames: int = 8, 
        frame_size: tuple[int, int] = (224, 224)
    ) -> None:
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.flow_extractor = OpticalFlowExtractor()
        
        self.speeds = np.loadtxt(speed_file)
        
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        self.valid_indices = list(range(0, self.total_frames - num_frames + 1))
        
        print(f"Dataset initialized: {len(self.valid_indices)} sequences")
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        start_frame = self.valid_indices[idx]
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames: list[NDArray[Any]] = []
        flow_frames: list[NDArray[Any]] = []
        
        raw_frames: list[NDArray[Any]] = []
        for i in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {start_frame + i}")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            raw_frames.append(frame)
        
        cap.release()
        
        if len(raw_frames) != self.num_frames:
            print(f"Warning: Expected {self.num_frames} frames, got {len(raw_frames)}")
            while len(raw_frames) < self.num_frames:
                raw_frames.append(raw_frames[-1])
        
        for i in range(len(raw_frames)):
            frame = raw_frames[i]
            frames.append(frame)
            
            if i > 0:
                try:
                    flow = self.flow_extractor.extract_flow(raw_frames[i-1], frame)
                    flow_rgb = self.flow_extractor.flow_to_rgb(flow)
                    flow_frames.append(flow_rgb)
                except Exception as e:
                    print(f"Flow extraction failed at frame {i}: {e}")
                    zero_flow = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                    flow_frames.append(zero_flow)
            else:
                zero_flow = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                flow_frames.append(zero_flow)
        
        while len(flow_frames) < len(frames):
            flow_frames.append(flow_frames[-1])
        
        frames_array: NDArray[np.float32] = np.array(frames).astype(np.float32) / 255.0
        flow_frames_array: NDArray[np.float32] = np.array(flow_frames).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames_array = (frames_array - mean) / std
        
        flow_frames_array = (flow_frames_array - 0.5) / 0.5
        
        frames_array = frames_array.transpose(0, 3, 1, 2)
        flow_frames_array = flow_frames_array.transpose(0, 3, 1, 2)
        
        speed_idx = start_frame + self.num_frames // 2
        speed = self.speeds[min(speed_idx, len(self.speeds) - 1)]
        
        return (torch.FloatTensor(frames_array), torch.FloatTensor(flow_frames_array)), torch.FloatTensor([speed])
