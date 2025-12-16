"""
GAT-LSTM EUR/USD Forecasting System

A novel hybrid Graph Attention Network and LSTM neural network
for forex price direction forecasting.

Author: Willard
Institution: UOW Malaysia KDU Penang University College
"""

__version__ = "1.0.0"
__author__ = "Willard"

from .data_loader import prepare_data_pipeline
from .features import engineer_features, create_targets
from .graph import build_adjacency_matrix
from .dataset import ForexGraphDataset
