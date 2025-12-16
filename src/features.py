"""
Feature engineering module for GAT-LSTM Forex Forecasting.

This module handles:
- Technical indicator calculation (graph nodes)
- Target variable creation
- Derived feature computation
"""

import numpy as np
import pandas as pd
import ta
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import FEATURE_NODES


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators that serve as graph nodes.
    
    The following indicators are computed:
    - RSI (14-day): Momentum oscillator
    - MACD: Trend-following momentum
    - MACD Signal: Signal line for MACD
    - MACD Histogram: Difference between MACD and Signal
    - EMA (20-day): Exponential moving average
    - Log Return: Daily logarithmic return
    - Rolling Volatility (14-day): Standard deviation of returns
    - Momentum (5-day): Price momentum
    - RSI Momentum: Rate of change of RSI
    - MACD Momentum: Rate of change of MACD
    - Price-EMA Distance: Normalized distance from EMA
    
    Args:
        data: DataFrame with OHLCV columns (open, high, low, close, volume)
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    df = data.copy()
    
    # Validate required columns
    required_cols = ['close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # --- Primary Indicators ---
    
    # RSI (Relative Strength Index)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()
    
    # EMA (Exponential Moving Average)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    
    # --- Derived Features ---
    
    # Log Return (daily)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling Volatility (14-day standard deviation of log returns)
    df['rolling_vol_14'] = df['log_return'].rolling(window=14).std()
    
    # Price Momentum (5-day)
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    
    # RSI Momentum (rate of change)
    df['rsi_momentum'] = df['rsi_14'].diff()
    
    # MACD Momentum (rate of change)
    df['macd_momentum'] = df['macd'].diff()
    
    # Price-EMA Distance (normalized)
    df['price_ema_dist'] = (df['close'] - df['ema_20']) / df['ema_20']
    
    print(f"✅ Features engineered: {len(FEATURE_NODES)} technical indicators")
    
    return df


def create_targets(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create prediction target variables.
    
    Three targets are created for multi-task learning:
    1. Direction (Binary): Whether next day's return is positive
    2. Return (Continuous): Next day's log return
    3. Volatility (Continuous): Absolute value of next day's return
    
    Args:
        data: DataFrame with features
        
    Returns:
        DataFrame with target columns added
        
    Note:
        Rows with NaN values (from shifting) are dropped.
    """
    df = data.copy()
    
    # Target 1: Direction (Binary Classification)
    # 1 if next day close > today's close, else 0
    next_day_return = np.log(df['close'].shift(-1) / df['close'])
    df['target_dir'] = (next_day_return > 0).astype(int)
    
    # Target 2: Return (Regression)
    # Log return to next day
    df['target_return'] = next_day_return
    
    # Target 3: Volatility (Regression)
    # Absolute return as volatility proxy
    df['target_vol'] = df['target_return'].abs()
    
    # Drop rows with NaN (created by shifting operations)
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    
    print(f"✅ Targets created: direction, return, volatility")
    print(f"   Dropped {dropped} rows with NaN values")
    
    return df


def get_feature_descriptions() -> dict:
    """
    Get descriptions of all feature nodes.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        'rsi_14': 'Relative Strength Index (14-day momentum oscillator)',
        'macd': 'MACD Line (12-26 EMA difference)',
        'macd_signal': 'MACD Signal Line (9-day EMA of MACD)',
        'macd_hist': 'MACD Histogram (MACD - Signal)',
        'ema_20': '20-day Exponential Moving Average',
        'log_return': 'Daily Logarithmic Return',
        'rolling_vol_14': '14-day Rolling Volatility (Std Dev)',
        'momentum_5': '5-day Price Momentum',
        'rsi_momentum': 'RSI Rate of Change',
        'macd_momentum': 'MACD Rate of Change',
        'price_ema_dist': 'Normalized Price-EMA Distance'
    }


if __name__ == "__main__":
    # Test feature engineering with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
    sample_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Engineer features
    df_feat = engineer_features(sample_df)
    print(f"\nFeature columns: {FEATURE_NODES}")
    
    # Create targets
    df_final = create_targets(df_feat)
    print(f"\nFinal shape: {df_final.shape}")
    print(f"Target columns: ['target_dir', 'target_return', 'target_vol']")
