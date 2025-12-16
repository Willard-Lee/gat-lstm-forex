"""
Training script for GAT-LSTM Forex Forecasting model.

Usage:
    python train.py --data data/EURUSD_daily.csv --epochs 50
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    DEVICE, SEED, FEATURE_NODES, SEQ_LEN, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, MODEL_DROPOUT,
    LOSS_WEIGHT_DIRECTION, LOSS_WEIGHT_VOLATILITY, LOSS_WEIGHT_RETURN,
    MODEL_SAVE_PATH
)
from src.data_loader import prepare_data_pipeline
from src.graph import build_adjacency_matrix
from src.dataset import create_data_loaders
from src.models import HybridGATLSTM


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    criterion_dir: nn.Module,
    criterion_reg: nn.Module,
    adj_tensor: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with loss and accuracy metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y_d, y_r, y_v in train_loader:
        x = x.to(device)
        y_d = y_d.to(device)
        y_r = y_r.to(device)
        y_v = y_v.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        p_d, p_r, p_v = model(x, adj_tensor)
        
        # Calculate losses
        loss_d = criterion_dir(p_d.squeeze(), y_d)
        loss_r = criterion_reg(p_r.squeeze(), y_r)
        loss_v = criterion_reg(p_v[:, 0], y_v)
        
        # Weighted total loss
        loss = (LOSS_WEIGHT_DIRECTION * loss_d + 
                LOSS_WEIGHT_VOLATILITY * loss_v + 
                LOSS_WEIGHT_RETURN * loss_r)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track accuracy
        preds = (torch.sigmoid(p_d.squeeze()) > 0.5).float()
        correct += (preds == y_d).sum().item()
        total += y_d.size(0)
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }


def validate(
    model: nn.Module,
    val_loader,
    criterion_dir: nn.Module,
    criterion_reg: nn.Module,
    adj_tensor: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Validate the model.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y_d, y_r, y_v in val_loader:
            x = x.to(device)
            y_d = y_d.to(device)
            y_r = y_r.to(device)
            y_v = y_v.to(device)
            
            p_d, p_r, p_v = model(x, adj_tensor)
            
            loss_d = criterion_dir(p_d.squeeze(), y_d)
            loss_r = criterion_reg(p_r.squeeze(), y_r)
            loss_v = criterion_reg(p_v[:, 0], y_v)
            
            loss = (LOSS_WEIGHT_DIRECTION * loss_d + 
                    LOSS_WEIGHT_VOLATILITY * loss_v + 
                    LOSS_WEIGHT_RETURN * loss_r)
            
            total_loss += loss.item()
            
            preds = (torch.sigmoid(p_d.squeeze()) > 0.5).float()
            correct += (preds == y_d).sum().item()
            total += y_d.size(0)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total
    }


def train(
    data_path: str,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    save_path: str = MODEL_SAVE_PATH
):
    """
    Main training function.
    
    Args:
        data_path: Path to CSV data file
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_path: Path to save trained model
    """
    print("=" * 60)
    print("GAT-LSTM TRAINING")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Random seed: {SEED}")
    
    # Prepare data
    print("\n[1/4] Preparing data...")
    train_df, val_df, test_df, scaler = prepare_data_pipeline(data_path)
    
    # Build graph
    print("\n[2/4] Building graph...")
    adj_tensor = build_adjacency_matrix(train_df, FEATURE_NODES)
    
    # Create data loaders
    print("\n[3/4] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        shuffle_train=True
    )
    
    # Initialize model
    print("\n[4/4] Initializing model...")
    model = HybridGATLSTM(
        nfeat=1,
        n_nodes=len(FEATURE_NODES),
        dropout=MODEL_DROPOUT
    ).to(DEVICE)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss functions
    criterion_dir = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            criterion_dir, criterion_reg,
            adj_tensor, DEVICE
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader,
            criterion_dir, criterion_reg,
            adj_tensor, DEVICE
        )
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2%} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2%}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {save_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train GAT-LSTM model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--output', type=str, default=MODEL_SAVE_PATH,
                        help='Path to save model')
    
    args = parser.parse_args()
    
    train(
        data_path=args.data,
        epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.output
    )


if __name__ == "__main__":
    main()
