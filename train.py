import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from typing import Any

from datasets.speed_dataset import SpeedDataset
from datasets.speed_flow_dataset import SpeedDatasetWithFlow
from models.swin_models import SpeedEstimatorSwin, SpeedEstimatorSwinOpticalFlow
from utils.dataset_split import create_train_val_test_split
from utils.config import load_config


def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    config: dict[str, Any], 
    device: str = 'cuda'
) -> None:
    model = model.to(device)
    
    num_epochs: int = config['training']['num_epochs']
    lr: float = config['training']['learning_rate']
    save_dir: str = config['training']['save_dir']
    save_best_only: bool = config['training']['save_best_only']
    
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss: float = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss: float = 0
        for frames, speeds in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
            frames, speeds = frames.to(device), speeds.to(device)
            optimizer.zero_grad()
            pred_speeds = model(frames)
            loss = criterion(pred_speeds, speeds)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss: float = 0
        with torch.no_grad():
            for frames, speeds in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Val'):
                frames, speeds = frames.to(device), speeds.to(device)
                pred_speeds = model(frames)
                loss = criterion(pred_speeds, speeds)
                val_loss += loss.item()
        
        avg_train_loss: float = train_loss / len(train_loader)
        avg_val_loss: float = val_loss / len(val_loader)
        
        if config['wandb']['enable']:
            wandb.log({
                "epoch": epoch + 1,
                "train_mse_loss": avg_train_loss,
                "val_mse_loss": avg_val_loss,
                "learning_rate": lr
            })
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        if save_best_only and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path: str = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
            
            if config['wandb']['enable']:
                wandb.log({
                    "best_val_mse_loss": best_val_loss,
                    "best_model_epoch": epoch + 1
                })
        elif not save_best_only:
            model_save_path: str = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
    
    if config['wandb']['enable']:
        wandb.log({"final_best_val_mse_loss": best_val_loss})
        wandb.finish()


def train_dual_stream_model(
    model: SpeedEstimatorSwinOpticalFlow, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    config: dict[str, Any], 
    device: str = 'cuda'
) -> None:
    model = model.to(device)
    
    num_epochs: int = config['training']['num_epochs']
    lr: float = config['training']['learning_rate']
    save_dir: str = config['training']['save_dir']
    save_best_only: bool = config['training']['save_best_only']
    
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss: float = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss: float = 0
        for (rgb_frames, flow_frames), speeds in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
            rgb_frames = rgb_frames.to(device)
            flow_frames = flow_frames.to(device)
            speeds = speeds.to(device)
            
            optimizer.zero_grad()
            pred_speeds = model(rgb_frames, flow_frames)
            loss = criterion(pred_speeds, speeds)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss: float = 0
        with torch.no_grad():
            for (rgb_frames, flow_frames), speeds in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Val'):
                rgb_frames = rgb_frames.to(device)
                flow_frames = flow_frames.to(device)
                speeds = speeds.to(device)
                
                pred_speeds = model(rgb_frames, flow_frames)
                loss = criterion(pred_speeds, speeds)
                val_loss += loss.item()
        
        avg_train_loss: float = train_loss / len(train_loader)
        avg_val_loss: float = val_loss / len(val_loader)
        
        if config['wandb']['enable']:
            wandb.log({
                "epoch": epoch + 1,
                "train_mse_loss": avg_train_loss,
                "val_mse_loss": avg_val_loss,
                "learning_rate": lr
            })
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        if save_best_only and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path: str = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
            
            if config['wandb']['enable']:
                wandb.log({
                    "best_val_mse_loss": best_val_loss,
                    "best_model_epoch": epoch + 1
                })
        elif not save_best_only:
            model_save_path: str = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
    
    if config['wandb']['enable']:
        wandb.log({"final_best_val_mse_loss": best_val_loss})
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train speed estimation models with YAML config")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    config: dict[str, Any] = load_config(args.config)
    
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id: str = f"{config['model_type']}_{timestamp}"
    
    if 'save_dir' not in config['training']:
        config['training']['save_dir'] = 'checkpoints'
    save_dir: str = os.path.join(config['training']['save_dir'], experiment_id)
    config['training']['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if config['wandb']['enable']:
        wandb.init(
            project=config['wandb']['project_name'],
            name=config['wandb']['run_name'] or experiment_id,
            config=config
        )
    
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
        print(f"Created Swin Transformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        train_indices, val_indices, _ = create_train_val_test_split(
            dataset, config['data']['train_ratio'], config['data']['val_ratio']
        )
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader: DataLoader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            num_workers=config['training']['num_workers']
        )
        val_loader: DataLoader = DataLoader(
            val_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            num_workers=config['training']['num_workers']
        )
        
        print(f"Starting training for {config['training']['num_epochs']} epochs...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
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
        print(f"Created dual-stream model with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        train_indices, val_indices, _ = create_train_val_test_split(
            dataset, config['data']['train_ratio'], config['data']['val_ratio']
        )
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader: DataLoader = DataLoader(
            train_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True, 
            num_workers=config['training']['num_workers']
        )
        val_loader: DataLoader = DataLoader(
            val_dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=False, 
            num_workers=config['training']['num_workers']
        )
        
        print(f"Starting training for {config['training']['num_epochs']} epochs...")
        train_dual_stream_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")


if __name__ == "__main__":
    main()