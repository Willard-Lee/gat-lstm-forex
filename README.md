# GAT-LSTM EUR/USD Forex Forecasting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel hybrid Graph Attention Network (GAT) and Long Short-Term Memory (LSTM) neural network system for EUR/USD forex price direction forecasting.

## 📋 Abstract

This project implements a cutting-edge deep learning approach that combines Graph Attention Networks with LSTM for financial time series prediction. The system models technical indicators as interconnected graph nodes, where correlations between indicators inform the attention mechanism, while LSTM captures temporal dependencies.

**Key Innovation:** Unlike traditional approaches that treat technical indicators as independent features, this system explicitly models the relationships between indicators using graph structures, allowing the model to learn which indicator combinations are most predictive.

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Input: OHLCV Time Series        │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │     Feature Engineering (11 Nodes)      │
                    │  RSI, MACD, EMA, Momentum, Volatility   │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
    ┌─────────▼─────────┐                         ┌───────────▼───────────┐
    │   Path A: GAT     │                         │    Path B: LSTM       │
    │ (Spatial/Graph)   │                         │    (Temporal)         │
    │                   │                         │                       │
    │ Embedding → GAT1  │                         │  2-Layer LSTM (64)    │
    │     → GAT2        │                         │                       │
    └─────────┬─────────┘                         └───────────┬───────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         Feature Concatenation           │
                    │        (64 + 88 = 152 features)         │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
    │  Direction Head   │   │   Return Head     │   │  Volatility Head  │
    │  (Binary Class)   │   │   (Regression)    │   │   (Regression)    │
    └───────────────────┘   └───────────────────┘   └───────────────────┘
```

## 📊 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Direction Accuracy | > 55% | vs. 50% random walk baseline |
| RMSE | < 0.5 | Return prediction error |
| Sharpe Ratio | > 1.2 | Risk-adjusted returns |
| AUC-ROC | > 0.55 | Prediction discriminative power |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gat-lstm-forex.git
cd gat-lstm-forex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Streamlit Dashboard

```bash
streamlit run app.py
```

### Training the Model

```bash
python src/train.py --data data/EURUSD_daily.csv --epochs 50
```

### Running Evaluation

```bash
python src/evaluate.py --model models/gat_lstm_model.pth --data data/EURUSD_daily.csv
```

## 📁 Project Structure

```
gat_lstm_forex/
├── app.py                      # Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── configs/
│   └── config.py               # Hyperparameters and settings
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── features.py             # Feature engineering
│   ├── graph.py                # Graph construction
│   ├── dataset.py              # PyTorch Dataset class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gat.py              # Graph Attention Layer
│   │   ├── lstm.py             # LSTM baseline
│   │   └── hybrid.py           # Hybrid GAT-LSTM model
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation and metrics
│   └── backtest.py             # Backtesting engine
│
├── data/
│   └── EURUSD_daily.csv        # Sample data (not included)
│
├── models/
│   ├── gat_lstm_model.pth      # Trained model weights
│   └── scaler.pkl              # Fitted scaler
│
├── outputs/
│   ├── metrics.csv             # Evaluation metrics
│   └── predictions.csv         # Model predictions
│
├── notebooks/
│   └── GATLSTMv2.ipynb         # Development notebook
│
└── tests/
    ├── test_data_loader.py
    ├── test_models.py
    └── test_backtest.py
```

## 🔧 Configuration

All hyperparameters are centralized in `configs/config.py`:

```python
# Model Architecture
SEQ_LEN = 30              # Sequence length (days)
GAT_HIDDEN_DIM = 16       # GAT hidden dimension
LSTM_HIDDEN_DIM = 64      # LSTM hidden dimension
LSTM_NUM_LAYERS = 2       # Number of LSTM layers

# Training
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DROPOUT = 0.5

# Graph Construction
CORRELATION_THRESHOLD = 0.6
CORRELATION_METHOD = 'spearman'

# Data Splits
TRAIN_PERIOD = "2014-01-01 to 2020-01-01"
VAL_PERIOD = "2020-01-01 to 2022-01-01"
TEST_PERIOD = "2022-01-01 to 2024-12-31"
```

## 📈 Feature Nodes (Graph Vertices)

The following 11 technical indicators serve as nodes in the graph structure:

| Node | Description | Category |
|------|-------------|----------|
| `rsi_14` | Relative Strength Index (14-day) | Momentum |
| `macd` | MACD Line | Trend |
| `macd_signal` | MACD Signal Line | Trend |
| `macd_hist` | MACD Histogram | Trend |
| `ema_20` | 20-day Exponential Moving Average | Trend |
| `log_return` | Logarithmic Daily Return | Returns |
| `rolling_vol_14` | 14-day Rolling Volatility | Volatility |
| `momentum_5` | 5-day Price Momentum | Momentum |
| `rsi_momentum` | RSI Rate of Change | Momentum |
| `macd_momentum` | MACD Rate of Change | Trend |
| `price_ema_dist` | Price-EMA Distance (Normalized) | Mean Reversion |

## 🧪 Reproducibility

To ensure reproducibility:

```python
import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{gatlstm_forex_2025,
    title={A Novel GAT-LSTM Hybrid Neural Network System for EUR/USD Forex Forecasting},
    author={Willard},
    school={UOW Malaysia KDU Penang University College},
    year={2025},
    supervisor={Prof J. Joshua Thomas}
}
```

## ⚠️ Disclaimer

**This is a proof-of-concept system developed for academic research purposes. It is NOT intended for real trading or financial decision-making.** 

- Past performance does not guarantee future results
- Forex trading involves substantial risk of loss
- This system has not been validated for live trading environments
- The authors assume no responsibility for any financial losses

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Prof J. Joshua Thomas (Dissertation Supervisor)
- UOW Malaysia KDU Penang University College
- PyTorch team for the deep learning framework
- `ta` library for technical analysis indicators

## 📧 Contact

For questions about this research, please open an issue on GitHub or contact the author through the university.
# gat-lstm-forex
