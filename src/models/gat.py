"""
Graph Attention Layer implementation for GAT-LSTM Forex Forecasting.

This module implements the Graph Attention Network layer that learns
to weight connections between technical indicators adaptively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with multi-head attention mechanism.

    This layer implements the attention mechanism from the GAT paper:
    "Graph Attention Networks" (Veličković et al., 2018)

    The layer learns to assign importance weights to different neighbors
    in the graph, allowing the model to focus on the most relevant
    technical indicators for prediction.

    Args:
        in_features: Size of input feature vectors
        out_features: Size of output feature vectors
        dropout: Dropout rate for attention coefficients
        alpha: Negative slope for LeakyReLU activation
        concat: Whether to apply ELU activation (True for intermediate layers)

    Input Shape:
        h: [Batch, Nodes, in_features]
        adj: [Batch, Nodes, Nodes] or [Nodes, Nodes]

    Output Shape:
        h_out: [Batch, Nodes, out_features]
        attention: [Batch, Nodes, Nodes] - attention weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Learnable weight matrix for feature transformation
        # W: [in_features, out_features]
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Learnable attention mechanism parameters
        # a: [2*out_features, 1] for computing attention scores
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU activation for attention mechanism
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Forward pass of the Graph Attention Layer.

        Args:
            h: Input features [Batch, Nodes, in_features]
            adj: Adjacency matrix [Batch, Nodes, Nodes] or [Nodes, Nodes]

        Returns:
            Tuple of (output_features, attention_weights)
        """
        # Linear transformation: h' = h @ W
        # Wh: [Batch, Nodes, out_features]
        Wh = torch.matmul(h, self.W)
        B, N, _ = Wh.size()

        # Prepare for attention computation
        # We need to compute attention for all pairs of nodes

        # Repeat Wh for all source nodes
        # [B, N, F] -> [B, N*N, F]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)

        # Repeat Wh for all target nodes
        # [B, N, F] -> [B, N*N, F]
        Wh_repeated_alternating = Wh.repeat(1, N, 1)

        # Concatenate source and target features
        # [B, N*N, 2*F]
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2
        )

        # Reshape to [B, N, N, 2*F]
        all_combinations_matrix = all_combinations_matrix.view(B, N, N, 2 * self.out_features)

        # Compute attention scores using learned attention mechanism
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # [B, N, N, 2*F] @ [2*F, 1] -> [B, N, N, 1] -> [B, N, N]
        e = self.leakyrelu(torch.matmul(all_combinations_matrix, self.a).squeeze(3))

        # Mask attention scores using adjacency matrix
        # Set attention to -infinity where there's no edge
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Normalize attention scores using softmax
        # attention: [B, N, N]
        attention = F.softmax(attention, dim=2)

        # Apply dropout to attention weights
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention to transformed features
        # h_out = attention @ Wh
        # [B, N, N] @ [B, N, F] -> [B, N, F]
        h_out = torch.matmul(attention, Wh)

        # Apply activation function if this is not the final layer
        if self.concat:
            return F.elu(h_out), attention
        else:
            return h_out, attention

    def __repr__(self):
        return (f'{self.__class__.__name__} ('
                f'{self.in_features} -> {self.out_features}, '
                f'dropout={self.dropout}, alpha={self.alpha})')


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer.

    Implements multiple attention heads in parallel and concatenates
    or averages their outputs. This allows the model to attend to
    different aspects of the graph structure simultaneously.

    Args:
        in_features: Size of input features
        out_features: Size of output features per head
        num_heads: Number of attention heads
        dropout: Dropout rate
        alpha: LeakyReLU negative slope
        concat_heads: Whether to concatenate (True) or average (False) head outputs

    Output:
        If concat_heads=True: [Batch, Nodes, out_features * num_heads]
        If concat_heads=False: [Batch, Nodes, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat_heads: bool = True
    ):
        super(MultiHeadGATLayer, self).__init__()

        self.num_heads = num_heads
        self.concat_heads = concat_heads

        # Create multiple attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(
                in_features=in_features,
                out_features=out_features,
                dropout=dropout,
                alpha=alpha,
                concat=True
            )
            for _ in range(num_heads)
        ])

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Forward pass through all attention heads.

        Returns:
            Tuple of (output_features, list_of_attention_weights)
        """
        # Process through each head
        head_outputs = []
        head_attentions = []

        for head in self.attention_heads:
            out, attn = head(h, adj)
            head_outputs.append(out)
            head_attentions.append(attn)

        # Combine outputs from all heads
        if self.concat_heads:
            # Concatenate: [B, N, F] * num_heads -> [B, N, F * num_heads]
            output = torch.cat(head_outputs, dim=2)
        else:
            # Average: [B, N, F] * num_heads -> [B, N, F]
            output = torch.mean(torch.stack(head_outputs), dim=0)

        # Average attention weights across heads for visualization
        avg_attention = torch.mean(torch.stack(head_attentions), dim=0)

        return output, avg_attention

    def __repr__(self):
        return (f'{self.__class__.__name__} ('
                f'{self.num_heads} heads, '
                f'concat={self.concat_heads})')


if __name__ == "__main__":
    # Test the GAT layer
    print("Testing Graph Attention Layer...")

    # Create sample data
    batch_size = 4
    num_nodes = 11
    in_features = 16
    out_features = 8

    # Random input features
    h = torch.randn(batch_size, num_nodes, in_features)

    # Random adjacency matrix (binary)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)

    # Add self-loops
    for i in range(num_nodes):
        adj[:, i, i] = 1

    # Test single-head GAT
    print("\n1. Single-head GAT:")
    gat = GraphAttentionLayer(in_features, out_features, dropout=0.2, alpha=0.2)
    out, attn = gat(h, adj)
    print(f"   Input shape: {h.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention shape: {attn.shape}")
    print(f"   Parameters: {sum(p.numel() for p in gat.parameters()):,}")

    # Test multi-head GAT
    print("\n2. Multi-head GAT (4 heads):")
    multi_gat = MultiHeadGATLayer(
        in_features, out_features,
        num_heads=4, dropout=0.2, concat_heads=True
    )
    out_multi, attn_multi = multi_gat(h, adj)
    print(f"   Input shape: {h.shape}")
    print(f"   Output shape: {out_multi.shape}")
    print(f"   Attention shape: {attn_multi.shape}")
    print(f"   Parameters: {sum(p.numel() for p in multi_gat.parameters()):,}")

    print("\n✅ GAT layer tests passed!")
