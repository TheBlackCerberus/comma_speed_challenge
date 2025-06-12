import pytest
import torch
from typing import Dict
from models.swin_models import SpeedEstimatorSwin


@pytest.fixture
def model_params() -> Dict[str, int]:
    """Test parameters for the model"""
    return {
        "batch_size": 2,
        "num_frames": 8,
        "channels": 3,
        "height": 256,
        "width": 256
    }


@pytest.fixture
def swin_model(model_params: Dict[str, int]) -> SpeedEstimatorSwin:
    """Create a SpeedEstimatorSwin model fixture"""
    model = SpeedEstimatorSwin(num_frames=model_params["num_frames"])
    model.eval() 
    return model


def test_model_initialization(swin_model: SpeedEstimatorSwin) -> None:
    """Test if the model initializes correctly"""
    assert swin_model is not None
    assert hasattr(swin_model, "backbone")
    assert hasattr(swin_model, "regressor")
    assert hasattr(swin_model, "temporal_pool")


def test_model_backbone_config(swin_model: SpeedEstimatorSwin) -> None:
    """Test if the backbone is configured correctly"""
    assert swin_model.backbone.config.hidden_size > 0
    assert hasattr(swin_model.backbone.config, "num_frames")


def test_forward_pass(swin_model: SpeedEstimatorSwin, model_params: Dict[str, int]) -> None:
    """Test the forward pass of the model"""
    dummy_input: torch.Tensor = torch.randn(
        model_params["batch_size"],
        model_params["num_frames"],
        model_params["channels"],
        model_params["height"],
        model_params["width"]
    )
    
    with torch.no_grad():
        output: torch.Tensor = swin_model(dummy_input)
    
    assert output.shape == (model_params["batch_size"], 1)
    assert torch.all(torch.isfinite(output))


def test_different_batch_sizes(swin_model: SpeedEstimatorSwin, model_params: Dict[str, int]) -> None:
    """Test the model with different batch sizes"""
    batch_sizes: list[int] = [1, 4, 8]
    
    for batch_size in batch_sizes:
        test_input: torch.Tensor = torch.randn(
            batch_size,
            model_params["num_frames"],
            model_params["channels"],
            model_params["height"],
            model_params["width"]
        )
        
        with torch.no_grad():
            test_output: torch.Tensor = swin_model(test_input)
        
        assert test_output.shape == (batch_size, 1)


def test_different_frame_counts(model_params: Dict[str, int]) -> None:
    """Test the model with different numbers of frames"""
    frame_counts: list[int] = [4, 16]
    
    for frames in frame_counts:
        test_model: SpeedEstimatorSwin = SpeedEstimatorSwin(num_frames=frames)
        test_model.eval()
        
        test_input: torch.Tensor = torch.randn(
            model_params["batch_size"],
            frames,
            model_params["channels"],
            model_params["height"],
            model_params["width"]
        )
        
        with torch.no_grad():
            test_output: torch.Tensor = test_model(test_input)
        
        assert test_output.shape == (model_params["batch_size"], 1) 