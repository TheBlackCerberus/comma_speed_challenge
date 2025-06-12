import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Any


class OpticalFlowExtractor:
    """Extract optical flow features from consecutive frames"""
    def __init__(
        self, 
        method: Literal['farneback'] = 'farneback',
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0
    ) -> None:
        self.method = method
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        
    def extract_flow(self, frame1: NDArray[Any], frame2: NDArray[Any]) -> NDArray[np.float32]:
        if frame1.shape != frame2.shape:
            raise ValueError(f"Frame size mismatch: {frame1.shape} vs {frame2.shape}")
        
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = frame2
        
        gray1 = gray1.astype(np.uint8)
        gray2 = gray2.astype(np.uint8)
        
        if self.method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback( 
                gray1, gray2, None, 
                pyr_scale=self.pyr_scale, 
                levels=self.levels, 
                winsize=self.winsize,
                iterations=self.iterations, 
                poly_n=self.poly_n, 
                poly_sigma=self.poly_sigma, 
                flags=self.flags
            )
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")
        
        return flow 
    
    def flow_to_rgb(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        mag, ang = cv2.cartToPolar(fx, fy)
        
        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:,:,0] = ang * 180 / np.pi / 2  # Hue 
        hsv[:,:,1] = 255  # Saturation
        hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value 
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb