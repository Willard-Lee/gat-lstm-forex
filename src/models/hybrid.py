"""
Hybrid GAT-LSTM model for EUR/USD Forex Forecasting.

This module implements the main model architecture combining:
- Graph Attention Networks for spatial dependencies between indicators
- LSTM for temporal dependencies in the time series
- Multi-task prediction heads for direction, return, and volatility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat import GraphAttentionLayer


class HybridGATLSTM(nn.Module):
    """
    Hybrid GAT-LSTM model for forex price direction forecasting.
    
    Architecture Overview:
    ---------------------
    Path A (Spatial): GAT layers process relationships between technical indicators
    Path B (Temporal): LSTM processes time series patterns
    
    The outputs from both paths are concatenated and fed to three
    prediction heads for multi-task learning.
    
    Args:
        nfeat: Number of input features per node (default: 1)
        n_nodes: Number of graph nodes (technical indicators)
        dropout: Dropout rate for regularization
        
    Input Shape:
        x: [Batch, Sequence, Nodes, Features]
        adj: [Nodes, Nodes] or [Batch, Nodes, Nodes]
        
    Output Shape:
        pred_dir: [Batch, 1] - Direction logits
        pred_ret: [Batch, 1] - Return prediction
        pred_vol: [Batch, 2] - Volatility quantiles
    """
    
    def __init__(self, nfeat: int = 1, n_nodes: int = 11, dropout: float = 0.2):
        super(HybridGATLSTM, self).__init__()
        
        self.n_nodes = n_nodes
        self.dropout = dropout
        
        # =====================================================================
        # PATH A: GRAPH ATTENTION (Spatial Dependencies)
        # =====================================================================
        
        # Embedding layer: Project scalar features to vectors
        # This gives the GAT more expressive power
        self.embedding = nn.Linear(1, 16)
        
        # GAT Layer 1: 16 -> 16 features with attention
        self.gat1 = GraphAttentionLayer(
            in_features=16,
            out_features=16,
            dropout=dropout,
            alpha=0.2,
            concat=True  # Apply ELU activation
        )
        
        # GAT Layer 2: 16 -> 8 features
        self.gat2 = GraphAttentionLayer(
            in_features=16,
            out_features=8,
            dropout=dropout,
            alpha=0.2,
            concat=False  # No activation (final GAT layer)
        )
        
        # =====================================================================
        # PATH B: LSTM (Temporal Dependencies)
        # =====================================================================
        
        # LSTM processes the raw node values over time
        self.lstm = nn.LSTM(
            input_size=n_nodes,      # Number of features per timestep
            hidden_size=64,          # LSTM hidden dimension
            num_layers=2,            # Stacked LSTM layers
            batch_first=True,        # Input shape: [Batch, Seq, Features]
            dropout=dropout
        )
        
        # =====================================================================
        # COMBINATION AND PREDICTION HEADS
        # =====================================================================
        
        # Combined dimension: LSTM output (64) + GAT output (8 * n_nodes)
        combined_dim = 64 + (8 * n_nodes)
        
        # Direction prediction head (binary classification)
        self.head_dir = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Return prediction head (regression)
        self.head_ret = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Volatility prediction head (quantile regression)
        self.head_vol = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # Two quantiles for uncertainty
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Forward pass of the hybrid model.
        
        Args:
            x: Input tensor [Batch, Sequence, Nodes, Features]
            adj: Adjacency matrix [Nodes, Nodes]
            
        Returns:
            Tuple of (direction_logits, return_pred, volatility_pred)
        """
        B, S, N, F = x.size()
        
        # =================================================================
        # PATH A: GAT Processing
        # =================================================================
        
        # Embed features: [B, S, N, 1] -> [B, S, N, 16]
        x_emb = self.embedding(x)
        
        # Flatten batch and sequence for GAT processing
        # [B, S, N, 16] -> [B*S, N, 16]
        x_flat = x_emb.view(B * S, N, -1)
        
        # Expand adjacency matrix for batch processing
        if adj.dim() == 2:
            adj_batch = adj.unsqueeze(0).repeat(B * S, 1, 1)
        else:
            adj_batch = adj
        
        # Apply GAT layers
        x_gat, attn1 = self.gat1(x_flat, adj_batch)
        x_gat, attn2 = self.gat2(x_gat, adj_batch)
        
        # Reshape back and take last timestep
        # [B*S, N, 8] -> [B, S, N*8] -> [B, N*8] (last timestep)
        x_gat_out = x_gat.view(B, S, -1)[:, -1, :]
        
        # =================================================================
        # PATH B: LSTM Processing
        # =================================================================
        
        # Reshape for LSTM: [B, S, N, F] -> [B, S, N]
        x_lstm_in = x.view(B, S, N)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_lstm_in)
        
        # Take last timestep output: [B, 64]
        x_lstm_out = lstm_out[:, -1, :]
        
        # =================================================================
        # COMBINE AND PREDICT
        # =================================================================
        
        # Concatenate GAT and LSTM outputs
        combined = torch.cat([x_lstm_out, x_gat_out], dim=1)
        
        # Multi-task predictions
        pred_dir = self.head_dir(combined)
        pred_ret = self.head_ret(combined)
        pred_vol = self.head_vol(combined)
        
        return pred_dir, pred_ret, pred_vol
    
    def get_attention_weights(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Extract attention weights for interpretability analysis.
        
        Args:
            x: Input tensor [Batch, Sequence, Nodes, Features]
            adj: Adjacency matrix
            
        Returns:
            Tuple of (attn_layer1, attn_layer2) attention matrices
        """
        B, S, N, F = x.size()
        
        x_emb = self.embedding(x)
        x_flat = x_emb.view(B * S, N, -1)
        
        if adj.dim() == 2:
            adj_batch = adj.unsqueeze(0).repeat(B * S, 1, 1)
        else:
            adj_batch = adj
        
        _, attn1 = self.gat1(x_flat, adj_batch)
        x_gat, _ = self.gat1(x_flat, adj_batch)
        _, attn2 = self.gat2(x_gat, adj_batch)
        
        # Average attention over batch*sequence
        attn1_avg = attn1.view(B, S, N, N).mean(dim=(0, 1))
        attn2_avg = attn2.view(B, S, N, N).mean(dim=(0, 1))
        
        return attn1_avg, attn2_avg
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    import sys
    sys.path.append('../..')
    from configs.config import FEATURE_NODES, SEQ_LEN, DEVICE
    
    # Create model
    model = HybridGATLSTM(nfeat=1, n_nodes=len(FEATURE_NODES), dropout=0.2)
    model = model.to(DEVICE)
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, SEQ_LEN, len(FEATURE_NODES), 1).to(DEVICE)
    adj = torch.eye(len(FEATURE_NODES)).to(DEVICE)
    
    pred_dir, pred_ret, pred_vol = model(x, adj)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shapes:")
    print(f"  Direction: {pred_dir.shape}")
    print(f"  Return: {pred_ret.shape}")
    print(f"  Volatility: {pred_vol.shape}")
