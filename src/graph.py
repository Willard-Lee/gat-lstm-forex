"""
Graph construction module for GAT-LSTM Forex Forecasting.

This module handles:
- Adjacency matrix construction from feature correlations
- Graph visualization utilities
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import FEATURE_NODES, CORRELATION_THRESHOLD, CORRELATION_METHOD, DEVICE


def build_adjacency_matrix(
    data: pd.DataFrame,
    features: List[str] = FEATURE_NODES,
    threshold: float = CORRELATION_THRESHOLD,
    method: str = CORRELATION_METHOD,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Build graph adjacency matrix from feature correlations.
    
    The adjacency matrix defines which feature nodes are connected
    in the graph. Edges are created between features that have
    correlation above the specified threshold.
    
    Args:
        data: DataFrame containing feature columns
        features: List of feature column names (graph nodes)
        threshold: Minimum correlation for edge creation
        method: Correlation method ('pearson' or 'spearman')
        add_self_loops: Whether to add self-connections
        
    Returns:
        Adjacency matrix as PyTorch tensor [N_nodes, N_nodes]
        
    Note:
        Uses absolute correlation values, so both positive and
        negative correlations above threshold create edges.
    """
    # Calculate correlation matrix
    corr_matrix = data[features].corr(method=method).abs()
    
    n_features = len(features)
    adj_matrix = np.zeros((n_features, n_features))
    
    edge_count = 0
    for i in range(n_features):
        for j in range(n_features):
            if i != j:  # Skip diagonal (handled by self-loops)
                if corr_matrix.iloc[i, j] > threshold:
                    adj_matrix[i, j] = 1
                    edge_count += 1
    
    # Add self-loops (nodes connect to themselves)
    if add_self_loops:
        np.fill_diagonal(adj_matrix, 1)
    
    print(f"✅ Graph built:")
    print(f"   Nodes: {n_features}")
    print(f"   Edges: {edge_count} (threshold > {threshold})")
    print(f"   Density: {edge_count / (n_features * (n_features - 1)):.2%}")
    
    return torch.FloatTensor(adj_matrix).to(DEVICE)


def get_edge_list(
    adj_matrix: torch.Tensor,
    features: List[str] = FEATURE_NODES
) -> List[tuple]:
    """
    Convert adjacency matrix to edge list representation.
    
    Args:
        adj_matrix: Adjacency matrix tensor
        features: List of feature names
        
    Returns:
        List of (source_node, target_node, weight) tuples
    """
    adj_np = adj_matrix.cpu().numpy()
    edges = []
    
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j and adj_np[i, j] > 0:
                edges.append((features[i], features[j], adj_np[i, j]))
    
    return edges


def visualize_graph(
    adj_matrix: torch.Tensor,
    features: List[str] = FEATURE_NODES,
    save_path: Optional[str] = None
):
    """
    Visualize the graph structure as a heatmap.
    
    Args:
        adj_matrix: Adjacency matrix tensor
        features: List of feature names
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        adj_matrix.cpu().numpy(),
        xticklabels=features,
        yticklabels=features,
        cmap='Blues',
        cbar=False,
        square=True,
        annot=True,
        fmt='.0f'
    )
    plt.title(f'Graph Adjacency Matrix\n(Correlation Threshold > {CORRELATION_THRESHOLD})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Graph visualization saved to {save_path}")
    
    plt.show()


def get_node_degrees(
    adj_matrix: torch.Tensor,
    features: List[str] = FEATURE_NODES
) -> dict:
    """
    Calculate the degree (number of connections) for each node.
    
    Args:
        adj_matrix: Adjacency matrix tensor
        features: List of feature names
        
    Returns:
        Dictionary mapping feature names to their degrees
    """
    adj_np = adj_matrix.cpu().numpy()
    
    # Subtract 1 to exclude self-loops from degree count
    degrees = adj_np.sum(axis=1) - 1  # Exclude self-loop
    
    return {features[i]: int(degrees[i]) for i in range(len(features))}


if __name__ == "__main__":
    # Test graph construction with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample correlation structure
    np.random.seed(42)
    n_samples = 500
    
    # Create correlated features
    base = np.random.randn(n_samples)
    sample_data = pd.DataFrame({
        'rsi_14': base + np.random.randn(n_samples) * 0.5,
        'macd': base * 0.8 + np.random.randn(n_samples) * 0.3,
        'macd_signal': base * 0.7 + np.random.randn(n_samples) * 0.4,
        'macd_hist': np.random.randn(n_samples),
        'ema_20': base * 0.6 + np.random.randn(n_samples) * 0.5,
        'log_return': np.random.randn(n_samples),
        'rolling_vol_14': abs(np.random.randn(n_samples)),
        'momentum_5': base * 0.5 + np.random.randn(n_samples) * 0.6,
        'rsi_momentum': np.random.randn(n_samples),
        'macd_momentum': np.random.randn(n_samples),
        'price_ema_dist': np.random.randn(n_samples)
    })
    
    # Build adjacency matrix
    adj = build_adjacency_matrix(sample_data)
    print(f"\nAdjacency matrix shape: {adj.shape}")
    
    # Get node degrees
    degrees = get_node_degrees(adj)
    print("\nNode degrees:")
    for node, degree in sorted(degrees.items(), key=lambda x: -x[1]):
        print(f"  {node}: {degree}")
