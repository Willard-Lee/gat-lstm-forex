"""
Data loading and preprocessing module for GAT-LSTM Forex Forecasting.

This module handles:
- Loading raw OHLCV data from CSV
- Data cleaning and standardization
- Train/Val/Test splitting with temporal integrity
- Feature scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Tuple, Optional
import sys
import os

# Add parent directory for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    DATA_PATH, DATA_SEPARATOR, FEATURE_NODES,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    SCALER_SAVE_PATH
)
from .features import engineer_features, create_targets


def load_raw_data(
    filepath: str = DATA_PATH,
    separator: str = DATA_SEPARATOR
) -> pd.DataFrame:
    """
    Load and standardize raw OHLCV data from CSV file.
    
    Supports MetaTrader export format (tab-separated with angle brackets)
    and standard CSV formats.
    
    Args:
        filepath: Path to the CSV file
        separator: CSV separator character (default: tab)
        
    Returns:
        DataFrame with standardized column names and datetime index
        
    Raises:
        ValueError: If no date column is found
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath, sep=separator)
    
    # Handle single-column result (wrong separator)
    if len(df.columns) == 1:
        df = pd.read_csv(filepath)
    
    # Remove angle brackets from column names (MetaTrader format)
    df.columns = df.columns.str.replace('<', '', regex=False)
    df.columns = df.columns.str.replace('>', '', regex=False)
    
    # Standardize column names to lowercase
    rename_map = {
        'DATE': 'date',
        'TIME': 'time',
        'OPEN': 'open',
        'HIGH': 'high',
        'LOW': 'low',
        'CLOSE': 'close',
        'TICKVOL': 'volume',
        'VOL': 'real_volume',
        'SPREAD': 'spread'
    }
    df = df.rename(columns={c: rename_map.get(c, c.lower()) for c in df.columns})
    
    # Create datetime index
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError("No date or datetime column found in data!")
    
    # Set datetime as index and sort
    df = df.sort_values('datetime').set_index('datetime')
    
    # Drop unnecessary columns
    cols_to_drop = ['date', 'time', 'real_volume', 'spread']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Ensure standard column order
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols]
    
    print(f"✅ Data loaded: {len(df)} rows")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def create_temporal_splits(
    data: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    val_start: str = VAL_START,
    val_end: str = VAL_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets based on time periods.
    
    Uses walk-forward methodology to prevent data leakage:
    - Train: Historical data for model training
    - Validation: Recent data for hyperparameter tuning
    - Test: Most recent data for final evaluation
    
    Args:
        data: DataFrame with datetime index
        train_start: Training period start date
        train_end: Training period end date
        val_start: Validation period start date
        val_end: Validation period end date
        test_start: Test period start date
        test_end: Test period end date
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_mask = (data.index >= train_start) & (data.index < train_end)
    val_mask = (data.index >= val_start) & (data.index < val_end)
    test_mask = (data.index >= test_start) & (data.index < test_end)
    
    train_df = data.loc[train_mask].copy()
    val_df = data.loc[val_mask].copy()
    test_df = data.loc[test_mask].copy()
    
    print(f"✅ Temporal splits created:")
    print(f"   Train: {len(train_df):,} samples ({train_start} to {train_end})")
    print(f"   Val:   {len(val_df):,} samples ({val_start} to {val_end})")
    print(f"   Test:  {len(test_df):,} samples ({test_start} to {test_end})")
    
    return train_df, val_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list = FEATURE_NODES,
    save_scaler: bool = True,
    scaler_path: str = SCALER_SAVE_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale features using MinMaxScaler fitted ONLY on training data.
    
    This is critical for preventing data leakage - the scaler must not
    see validation or test data during fitting.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        features: List of feature column names to scale
        save_scaler: Whether to save the fitted scaler to disk
        scaler_path: Path to save the scaler
        
    Returns:
        Tuple of (scaled_train, scaled_val, scaled_test, scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit ONLY on training data
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved to {scaler_path}")
    
    print("✅ Features scaled (fitted on training data only)")
    
    return train_df, val_df, test_df, scaler


def prepare_data_pipeline(
    filepath: str = DATA_PATH,
    save_scaler: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Execute the complete data preparation pipeline.
    
    Pipeline steps:
    1. Load raw OHLCV data
    2. Engineer technical indicator features
    3. Create prediction targets
    4. Split into train/val/test sets
    5. Scale features
    
    Args:
        filepath: Path to raw data CSV file
        save_scaler: Whether to save the fitted scaler
        
    Returns:
        Tuple of (train_df, val_df, test_df, scaler)
    """
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw data
    df = load_raw_data(filepath)
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Create targets
    df = create_targets(df)
    
    # Step 4: Split data temporally
    train_df, val_df, test_df = create_temporal_splits(df)
    
    # Step 5: Scale features
    train_df, val_df, test_df, scaler = scale_features(
        train_df, val_df, test_df,
        save_scaler=save_scaler
    )
    
    print("=" * 60)
    print("✅ Data pipeline complete!")
    print("=" * 60)
    
    return train_df, val_df, test_df, scaler


if __name__ == "__main__":
    # Test the pipeline
    try:
        train_df, val_df, test_df, scaler = prepare_data_pipeline()
        print(f"\nDataset shapes:")
        print(f"  Train: {train_df.shape}")
        print(f"  Val:   {val_df.shape}")
        print(f"  Test:  {test_df.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists at the specified path.")
