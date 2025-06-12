import torch
import torch.nn as nn
from transformers import Swinv2Model


class SpeedEstimatorSwin(nn.Module):
    """Video Swin Transformer + MLP head for speed regression"""
    def __init__(
        self, 
        num_frames: int = 8, 
        model_name: str = "microsoft/swinv2-tiny-patch4-window8-256",
        freeze_backbone: bool = True
    ) -> None:
        super().__init__()
        self.backbone = Swinv2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.backbone.config.num_frames = num_frames
        
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Temporal pooling layer to aggregate features
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C, H, W]
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape for backbone: [B*T, C, H, W]
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Pass through backbone
        outputs = self.backbone(pixel_values=x, return_dict=True)
        
        # Get sequence output and pool patches
        sequence_output = outputs.last_hidden_state  # [B*T, num_patches, hidden_size]
        pooled_output = sequence_output.mean(dim=1)  # [B*T, hidden_size]
        
        # Reshape back to separate batch and time dimensions
        pooled_output = pooled_output.view(batch_size, num_frames, -1)  # [B, T, hidden_size]
        
        # Temporal pooling - average across frames
        temporal_features = pooled_output.transpose(1, 2)  # [B, hidden_size, T]
        temporal_features = self.temporal_pool(temporal_features).squeeze(-1)  # [B, hidden_size]
        
        # Predict speed
        speed = self.regressor(temporal_features)
        return speed


class SpeedEstimatorSwinOpticalFlow(nn.Module):
    """Dual-stream Swin Transformer: RGB + Optical Flow"""
    def __init__(
        self, 
        num_frames: int = 8,
        rgb_model_name: str = "microsoft/swinv2-tiny-patch4-window8-256",
        flow_model_name: str = "microsoft/swinv2-tiny-patch4-window8-256",
        fusion_method: str = 'concat',
        freeze_backbone: bool = True
    ) -> None:
        super().__init__()
        
        self.num_frames = num_frames
        self.fusion_method = fusion_method
        
        # RGB stream
        self.rgb_backbone = Swinv2Model.from_pretrained(rgb_model_name)
        self.rgb_hidden_size = self.rgb_backbone.config.hidden_size
        
        # Optical flow stream
        self.flow_backbone = Swinv2Model.from_pretrained(flow_model_name)
        self.flow_hidden_size = self.flow_backbone.config.hidden_size
        
        # Freeze backbones if requested
        if freeze_backbone:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = False
            for param in self.flow_backbone.parameters():
                param.requires_grad = False
        
        # Temporal aggregation
        self.rgb_temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.flow_temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_input_size = self.rgb_hidden_size + self.flow_hidden_size
        elif fusion_method == 'add':
            fusion_input_size = self.rgb_hidden_size
            if self.flow_hidden_size != self.rgb_hidden_size:
                self.flow_projection = nn.Linear(self.flow_hidden_size, self.rgb_hidden_size)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # MLP regression head
        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, rgb_frames: torch.Tensor, flow_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_frames: [B, T, C, H, W] - RGB video frames
            flow_frames: [B, T, C, H, W] - Optical flow visualizations
        """
        
        rgb_features = self._process_stream(rgb_frames, self.rgb_backbone, self.rgb_temporal_pool)
        flow_features = self._process_stream(flow_frames, self.flow_backbone, self.flow_temporal_pool)
        
        # Fusion
        if self.fusion_method == 'concat':
            fused_features = torch.cat([rgb_features, flow_features], dim=1)
        elif self.fusion_method == 'add':
            if hasattr(self, 'flow_projection'):
                flow_features = self.flow_projection(flow_features)
            fused_features = rgb_features + flow_features
        
        # Predict speed
        speed = self.regressor(fused_features)
        return speed
    
    def _process_stream(self, frames: torch.Tensor, backbone: Swinv2Model, temporal_pool: nn.Module) -> torch.Tensor:
        """Process a stream of frames through backbone"""
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Reshape to process all frames at once
        frames = frames.view(batch_size * num_frames, channels, height, width)
        
        # Pass through backbone
        outputs = backbone(pixel_values=frames, return_dict=True)
        
        # Global average pooling across patches
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # [B*T, hidden_size]
        
        # Reshape back to separate batch and time dimensions
        pooled_output = pooled_output.view(batch_size, num_frames, -1)
        
        # Temporal pooling
        temporal_features = pooled_output.transpose(1, 2)  # [B, hidden_size, T]
        temporal_features = temporal_pool(temporal_features).squeeze(-1)  # [B, hidden_size]
        
        return temporal_features 