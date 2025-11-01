"""
U-Net Training Script
Train U-Net model for image enhancement
"""

import argparse
import os
from pathlib import Path
from typing import Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.modules.enhancer import UNet
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== Dataset ====================

class KKEnhancementDataset(Dataset):
    """
    Dataset for U-Net training
    Pairs of (noisy/original crop, clean/enhanced crop)
    """
    
    def __init__(
        self,
        data_dir: str,
        input_size: int = 256,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augment = augment
        
        # Find all image pairs
        self.input_dir = self.data_dir / "input"
        self.target_dir = self.data_dir / "target"
        
        self.input_images = sorted(list(self.input_dir.glob("*.png")) + list(self.input_dir.glob("*.jpg")))
        self.target_images = sorted(list(self.target_dir.glob("*.png")) + list(self.target_dir.glob("*.jpg")))
        
        assert len(self.input_images) == len(self.target_images), \
            "Input and target directories must have same number of images"
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        
        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),
            ])
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        # Load images
        input_img = Image.open(self.input_images[idx]).convert("RGB")
        target_img = Image.open(self.target_images[idx]).convert("RGB")
        
        # Apply augmentation
        if self.augment:
            # Apply same augmentation to both
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            input_img = self.augment_transform(input_img)
            torch.manual_seed(seed)
            target_img = self.augment_transform(target_img)
        
        # Transform to tensor
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)
        
        return input_tensor, target_tensor


# ==================== Loss Functions ====================

class CombinedLoss(nn.Module):
    """
    Combined L1 + SSIM loss for better perceptual quality
    """
    
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
    
    def ssim_loss(self, pred, target, window_size=11):
        """Simplified SSIM loss"""
        # Simplified implementation - consider using pytorch-msssim for full SSIM
        mu_pred = nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = nn.functional.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = nn.functional.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * ssim


# ==================== Training ====================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Validate one epoch"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_unet(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: str = "cuda",
    val_split: float = 0.2
):
    """
    Train U-Net model
    
    Args:
        data_dir: Directory containing input/ and target/ subdirectories
        output_dir: Directory to save model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use (cuda or cpu)
        val_split: Validation split ratio
    """
    logger.info(
        "Starting U-Net training",
        extra={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device
        }
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = KKEnhancementDataset(data_dir, augment=True)
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset size: {len(dataset)} (train: {train_size}, val: {val_size})")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=3)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logger.info(
            f"Epoch {epoch + 1}/{epochs}",
            extra={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]['lr']
            }
        )
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_path / "unet_best.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            
            logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_path / f"unet_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for KK image enhancement")
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    
    args = parser.parse_args()
    
    train_unet(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        val_split=args.val_split
    )
