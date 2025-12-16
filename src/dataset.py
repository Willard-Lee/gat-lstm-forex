"""
PyTorch Dataset module for GAT-LSTM Forex Forecasting.

This module handles:
- Sliding window sequence creation
- Multi-task target handling
- Efficient data loading
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import FEATURE_NODES, SEQ_LEN, BATCH_SIZE


class ForexGraphDataset(Dataset):
    """
    PyTorch Dataset for GAT-LSTM forex prediction.
    
    Creates sliding window sequences from time series data for
    training the hybrid GAT-LSTM model.
    
    Each sample contains:
    - Input: [SEQ_LEN, N_NODES, 1] tensor of feature values
    - Targets: direction (binary), return (float), volatility (float)
    
    Attributes:
        data: Source DataFrame
        features: List of feature column names
        seq_len: Length of input sequences
        x_tensor: Pre-computed feature tensor
        y_dir: Direction targets
        y_ret: Return targets
        y_vol: Volatility targets
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = FEATURE_NODES,
        seq_len: int = SEQ_LEN
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with features and targets
            features: List of feature column names
            seq_len: Length of input sequences
        """
        self.data = data
        self.features = features
        self.seq_len = seq_len
        
        # Pre-convert to tensors for faster loading
        # Shape: [T, N_nodes, 1] where T is total timesteps
        self.x_tensor = torch.FloatTensor(
            data[features].values
        ).unsqueeze(-1)
        
        # Target tensors
        self.y_dir = torch.FloatTensor(data['target_dir'].values)
        self.y_ret = torch.FloatTensor(data['target_return'].values)
        self.y_vol = torch.FloatTensor(data['target_vol'].values)
        
    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (x, y_direction, y_return, y_volatility)
        """
        # Input sequence of seq_len days
        x = self.x_tensor[idx:idx + self.seq_len]
        
        # Targets are at the END of the sequence
        # (predicting next day after the sequence)
        target_idx = idx + self.seq_len - 1
        y_d = self.y_dir[target_idx]
        y_r = self.y_ret[target_idx]
        y_v = self.y_vol[target_idx]
        
        return x, y_d, y_r, y_v
    
    def get_dates(self) -> pd.DatetimeIndex:
        """Get valid prediction dates (aligned with targets)."""
        return self.data.index[self.seq_len:]


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str] = FEATURE_NODES,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        features: List of feature column names
        seq_len: Sequence length
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ForexGraphDataset(train_df, features, seq_len)
    val_dataset = ForexGraphDataset(val_df, features, seq_len)
    test_dataset = ForexGraphDataset(test_df, features, seq_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"✅ DataLoaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=200, freq='B')
    
    sample_df = pd.DataFrame({
        'rsi_14': np.random.rand(200),
        'macd': np.random.randn(200),
        'macd_signal': np.random.randn(200),
        'macd_hist': np.random.randn(200),
        'ema_20': np.random.rand(200) + 1,
        'log_return': np.random.randn(200) * 0.01,
        'rolling_vol_14': np.random.rand(200) * 0.02,
        'momentum_5': np.random.randn(200) * 0.05,
        'rsi_momentum': np.random.randn(200),
        'macd_momentum': np.random.randn(200),
        'price_ema_dist': np.random.randn(200) * 0.01,
        'target_dir': np.random.randint(0, 2, 200),
        'target_return': np.random.randn(200) * 0.01,
        'target_vol': np.abs(np.random.randn(200) * 0.01)
    }, index=dates)
    
    # Create dataset
    dataset = ForexGraphDataset(sample_df, seq_len=30)
    print(f"\nDataset length: {len(dataset)}")
    
    # Get a sample
    x, y_d, y_r, y_v = dataset[0]
    print(f"Sample shapes:")
    print(f"  x: {x.shape}")
    print(f"  y_dir: {y_d.shape}")
    print(f"  y_ret: {y_r.shape}")
    print(f"  y_vol: {y_v.shape}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  x: {batch[0].shape}")
    print(f"  y_dir: {batch[1].shape}")
