import pytest
import torch
from models.swin_models import SpeedEstimatorSwinOpticalFlow


@pytest.fixture
def model_params() -> dict[str, int]:
    return {
        "batch_size": 2,
        "num_frames": 8,
        "channels": 3,
        "height": 256,
        "width": 256
    }


@pytest.fixture
def swin_flow_model(model_params: dict[str, int]) -> SpeedEstimatorSwinOpticalFlow:
    """Create a SpeedEstimatorSwinOpticalFlow model fixture with default fusion method"""
    model = SpeedEstimatorSwinOpticalFlow(num_frames=model_params["num_frames"], fusion_method='concat')
    model.eval() 
    return model


def test_model_initialization(swin_flow_model: SpeedEstimatorSwinOpticalFlow) -> None:
    """Test if the model initializes correctly"""
    assert swin_flow_model is not None
    assert hasattr(swin_flow_model, "rgb_backbone")
    assert hasattr(swin_flow_model, "flow_backbone")
    assert hasattr(swin_flow_model, "regressor")
    assert hasattr(swin_flow_model, "rgb_temporal_pool")
    assert hasattr(swin_flow_model, "flow_temporal_pool")


def test_backbone_config(swin_flow_model: SpeedEstimatorSwinOpticalFlow) -> None:
    """Test if the backbones are configured correctly"""
    assert swin_flow_model.rgb_hidden_size > 0
    assert swin_flow_model.flow_hidden_size > 0
    assert swin_flow_model.rgb_hidden_size == swin_flow_model.flow_hidden_size


def test_forward_pass(swin_flow_model: SpeedEstimatorSwinOpticalFlow, model_params: dict[str, int]) -> None:
    """Test the forward pass of the model"""
    # Create dummy inputs
    rgb_frames: torch.Tensor = torch.randn(
        model_params["batch_size"],
        model_params["num_frames"],
        model_params["channels"],
        model_params["height"],
        model_params["width"]
    )
    
    flow_frames: torch.Tensor = torch.randn(
        model_params["batch_size"],
        model_params["num_frames"],
        model_params["channels"],
        model_params["height"],
        model_params["width"]
    )
    
    # Forward pass
    with torch.no_grad():
        output: torch.Tensor = swin_flow_model(rgb_frames, flow_frames)
    
    # Check output shape
    assert output.shape == (model_params["batch_size"], 1)
    
    # Check output values are finite
    assert torch.all(torch.isfinite(output))


@pytest.mark.parametrize("fusion_method", ["concat", "add"])
def test_fusion_methods(model_params: dict[str, int], fusion_method: str) -> None:
    """Test different fusion methods"""
    model: SpeedEstimatorSwinOpticalFlow = SpeedEstimatorSwinOpticalFlow(
        num_frames=model_params["num_frames"],
        fusion_method=fusion_method
    )
    model.eval()
    
    rgb_frames: torch.Tensor = torch.randn(
        model_params["batch_size"],
        model_params["num_frames"],
        model_params["channels"],
        model_params["height"],
        model_params["width"]
    )
    
    flow_frames: torch.Tensor = torch.randn(
        model_params["batch_size"],
        model_params["num_frames"],
        model_params["channels"],
        model_params["height"],
        model_params["width"]
    )
    
    with torch.no_grad():
        output: torch.Tensor = model(rgb_frames, flow_frames)
    
    assert output.shape == (model_params["batch_size"], 1)


def test_different_batch_sizes(swin_flow_model: SpeedEstimatorSwinOpticalFlow, model_params: dict[str, int]) -> None:
    """Test the model with different batch sizes"""
    batch_sizes: list[int] = [1, 4, 8]
    
    for batch_size in batch_sizes:
        # Create test inputs
        rgb_frames: torch.Tensor = torch.randn(
            batch_size,
            model_params["num_frames"],
            model_params["channels"],
            model_params["height"],
            model_params["width"]
        )
        
        flow_frames: torch.Tensor = torch.randn(
            batch_size,
            model_params["num_frames"],
            model_params["channels"],
            model_params["height"],
            model_params["width"]
        )
        
        
        with torch.no_grad():
            output: torch.Tensor = swin_flow_model(rgb_frames, flow_frames)
        
        
        assert output.shape == (batch_size, 1)


def test_process_stream_method(swin_flow_model: SpeedEstimatorSwinOpticalFlow, model_params: dict[str, int]) -> None:
    """Test the internal _process_stream method"""
    

    frames: torch.Tensor = torch.randn(
        model_params["batch_size"],
        model_params["num_frames"],
        model_params["channels"],
        model_params["height"],
        model_params["width"]
    )
    
    with torch.no_grad():
        features: torch.Tensor = swin_flow_model._process_stream(
            frames, 
            swin_flow_model.rgb_backbone, 
            swin_flow_model.rgb_temporal_pool
        )
    
    assert features.shape == (model_params["batch_size"], swin_flow_model.rgb_hidden_size)
    assert torch.all(torch.isfinite(features)) 