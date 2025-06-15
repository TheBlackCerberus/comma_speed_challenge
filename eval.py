import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Subset
from typing import Any

from datasets.speed_dataset import SpeedDataset
from datasets.speed_flow_dataset import SpeedDatasetWithFlow
from models.swin_models import SpeedEstimatorSwin, SpeedEstimatorSwinOpticalFlow
from utils.dataset_split import create_train_val_test_split
from utils.config import load_config


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> tuple[float, list[float], list[float]]:
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss: float = 0.0
    all_predictions: list[float] = []
    all_targets: list[float] = []
    
    with torch.no_grad():
        for frames, speeds in tqdm(test_loader, desc='Evaluating'):
            frames, speeds = frames.to(device), speeds.to(device)
            pred_speeds = model(frames)
            loss = criterion(pred_speeds, speeds)
            total_loss += loss.item()
            
            all_predictions.extend(pred_speeds.cpu().numpy().flatten().tolist())
            all_targets.extend(speeds.cpu().numpy().flatten().tolist())
    
    avg_loss: float = total_loss / len(test_loader)
    return avg_loss, all_predictions, all_targets


def evaluate_dual_stream_model(
    model: SpeedEstimatorSwinOpticalFlow,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> tuple[float, list[float], list[float]]:
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss: float = 0.0
    all_predictions: list[float] = []
    all_targets: list[float] = []
    
    with torch.no_grad():
        for (rgb_frames, flow_frames), speeds in tqdm(test_loader, desc='Evaluating'):
            rgb_frames = rgb_frames.to(device)
            flow_frames = flow_frames.to(device)
            speeds = speeds.to(device)
            
            pred_speeds = model(rgb_frames, flow_frames)
            loss = criterion(pred_speeds, speeds)
            total_loss += loss.item()
            
            all_predictions.extend(pred_speeds.cpu().numpy().flatten().tolist())
            all_targets.extend(speeds.cpu().numpy().flatten().tolist())
    
    avg_loss: float = total_loss / len(test_loader)
    return avg_loss, all_predictions, all_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate speed estimation models")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    config: dict[str, Any] = load_config(args.config)
    
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config['model_type'] == "swin":
        print("Creating standard RGB dataset...")
        dataset = SpeedDataset(
            config['data']['video_path'], 
            config['data']['speed_file'], 
            num_frames=config['model_config']['num_frames'], 
            frame_size=(config['data']['frame_size'], config['data']['frame_size'])
        )
        
        model = SpeedEstimatorSwin(
            num_frames=config['model_config']['num_frames'],
            model_name=config['model_config']['model_name'],
            freeze_backbone=config['model_config']['freeze_backbone']
        )
        
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        _, _, test_indices = create_train_val_test_split(
            dataset, config['data']['train_ratio'], config['data']['val_ratio']
        )
        
        test_dataset = Subset(dataset, test_indices)
        
        test_loader: DataLoader = DataLoader(
            test_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            num_workers=config['training']['num_workers']
        )
        
        print(f"Starting evaluation on {len(test_dataset)} test samples...")
        avg_mse, _ , _ = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        print(f"Test MSE: {avg_mse:.4f}")
        
    elif config['model_type'] == "swin_flow":
        print("Creating dual-stream RGB+Flow dataset...")
        dataset = SpeedDatasetWithFlow(
            config['data']['video_path'], 
            config['data']['speed_file'], 
            num_frames=config['model_config']['num_frames'], 
            frame_size=(config['data']['frame_size'], config['data']['frame_size'])
        )
        
        model = SpeedEstimatorSwinOpticalFlow(
            num_frames=config['model_config']['num_frames'],
            rgb_model_name=config['model_config']['rgb_model_name'],
            flow_model_name=config['model_config']['flow_model_name'],
            fusion_method=config['model_config']['fusion_method'],
            freeze_backbone=config['model_config']['freeze_backbone']
        )
        
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
        _, _, test_indices = create_train_val_test_split(
            dataset, config['data']['train_ratio'], config['data']['val_ratio']
        )
        
        test_dataset = Subset(dataset, test_indices)
        
        test_loader: DataLoader = DataLoader(
            test_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            num_workers=config['training']['num_workers']
        )
        
        print(f"Starting evaluation on {len(test_dataset)} test samples...")
        avg_mse, _ , _ = evaluate_dual_stream_model(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        print(f"Test MSE: {avg_mse:.4f}")
        
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")


if __name__ == "__main__":
    main() 