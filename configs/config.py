"""
Configuration file for GAT-LSTM EUR/USD Forecasting System.

This module contains all hyperparameters, paths, and settings
for reproducible experiments.
"""

import torch

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_PATH = "data/EURUSD_daily.csv"
DATA_SEPARATOR = "\t"

# Temporal splits (no lookahead bias)
TRAIN_START = "2014-01-01"
TRAIN_END = "2020-01-01"
VAL_START = "2020-01-01"
VAL_END = "2022-01-01"
TEST_START = "2022-01-01"
TEST_END = "2025-01-01"

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
FEATURE_NODES = [
    'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'ema_20',
    'log_return', 'rolling_vol_14', 'momentum_5',
    'rsi_momentum', 'macd_momentum', 'price_ema_dist'
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Sequence and batch
SEQ_LEN = 30
BATCH_SIZE = 32  # Smaller batch for better generalization

# GAT parameters (increased capacity)
GAT_HIDDEN_DIM = 32  # Increased from 16
GAT_OUTPUT_DIM = 16  # Increased from 8
GAT_DROPOUT = 0.2
GAT_ALPHA = 0.2  # LeakyReLU negative slope

# LSTM parameters (increased capacity)
LSTM_HIDDEN_DIM = 128  # Increased from 64
LSTM_NUM_LAYERS = 3  # Increased from 2
LSTM_DROPOUT = 0.2

# Graph construction
CORRELATION_THRESHOLD = 0.6
CORRELATION_METHOD = 'spearman'

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
EPOCHS = 100  # Increased for better convergence
LEARNING_RATE = 0.0005  # Lower for stable training
MODEL_DROPOUT = 0.3  # Reduced to prevent underfitting

# Multi-task loss weights (prioritize direction prediction)
LOSS_WEIGHT_DIRECTION = 2.0  # Increased - most important for trading
LOSS_WEIGHT_VOLATILITY = 0.3  # Reduced
LOSS_WEIGHT_RETURN = 0.2  # Slightly increased for better return prediction

# =============================================================================
# BACKTESTING CONFIGURATION
# =============================================================================
BACKTEST_CONFIG = {
    'INITIAL_CAPITAL': 100000.0,
    'COMMISSION_FEE': 0.0002,
    'CONF_THRESHOLD': 0.50,
    'MAX_RISK_PER_TRADE_PCT': 0.02,
    'LEVERAGE_RATIO': 100.0,
    'LOT_SIZE': 100000,
    'R_R_RATIO': 1.0,
    'TRADING_DAYS_YEAR': 252,
}

# =============================================================================
# OUTPUT PATHS
# =============================================================================
MODEL_SAVE_PATH = "models/gat_lstm_model.pth"
SCALER_SAVE_PATH = "models/scaler.pkl"
METRICS_OUTPUT_PATH = "outputs/metrics.csv"
PREDICTIONS_OUTPUT_PATH = "outputs/predictions.csv"

# =============================================================================
# DISSERTATION TARGETS
# =============================================================================
TARGET_ACCURACY = 0.55  # > 55%
TARGET_RMSE = 0.5       # < 0.5
TARGET_SHARPE = 1.2     # > 1.2
TARGET_AUC = 0.55       # > 0.55
