"""
LSTM Baseline model for comparison.

This simple LSTM serves as an ablation baseline to demonstrate
the value added by the Graph Attention component.
"""

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    Simple LSTM baseline model for time series classification.
    
    This model processes the same input features as the hybrid model
    but without the graph attention component, serving as an
    ablation study baseline.
    
    Args:
        n_nodes: Number of input features (technical indicators)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        n_nodes: int = 11,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(SimpleLSTM, self).__init__()
        
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_nodes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Direction prediction head
        self.head_dir = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [Batch, Sequence, Nodes, Features]
               or [Batch, Sequence, Nodes]
               
        Returns:
            Direction logits [Batch, 1]
        """
        # Handle 4D input from GAT dataloader
        if x.dim() == 4:
            B, S, N, F = x.size()
            x = x.squeeze(-1)  # [B, S, N]
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last timestep
        x_out = lstm_out[:, -1, :]
        
        # Predict direction
        pred_dir = self.head_dir(x_out)
        
        return pred_dir
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the baseline model
    model = SimpleLSTM(n_nodes=11, hidden_dim=64, num_layers=2, dropout=0.2)
    
    print(f"LSTM Baseline: {model.count_parameters():,} parameters")
    
    # Test forward pass
    x = torch.randn(32, 30, 11, 1)  # Same format as GAT-LSTM input
    pred = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
