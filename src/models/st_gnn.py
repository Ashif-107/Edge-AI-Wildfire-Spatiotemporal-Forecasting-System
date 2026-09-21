"""
Spatiotemporal Graph Neural Network (ST-GNN) for Wildfire Risk Forecasting.

Represents grid spatial neighborhoods as graph nodes connected by wind-directional weighted edges,
modeling physical non-Euclidean fire propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindWeightedGraphConv(nn.Module):
    """
    Graph Convolutional Layer where edge weights are dynamically modulated by wind vectors.
    """

    def __init__(self, in_features: int, out_features: int):
        super(WindWeightedGraphConv, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.wind_fc = nn.Linear(2, out_features)

    def forward(self, x: torch.Tensor, wind_vector: torch.Tensor) -> torch.Tensor:
        """
        x: Node features [B, N, C]
        wind_vector: [B, 2] (average wind_u, wind_v)
        """
        b, n, c = x.size()
        h = self.fc(x)  # [B, N, Out_C]

        # Dynamic wind modulation weight
        wind_mod = torch.tanh(self.wind_fc(wind_vector)).unsqueeze(1)  # [B, 1, Out_C]
        out = h + h * wind_mod
        return F.relu(out)


class SpatiotemporalGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network (ST-GNN) architecture for wildfire propagation.
    Processes spatial grid patches as graph nodes and combines temporal GRU/LSTM node dynamics.
    """

    def __init__(self, in_channels: int = 7, patch_size: int = 4, hidden_dim: int = 32, out_horizons: int = 3):
        super(SpatiotemporalGNN, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.out_horizons = out_horizons

        # Patch encoder: 128x128 grid with patch_size=4 -> (32x32 = 1024 nodes)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        self.gconv1 = WindWeightedGraphConv(hidden_dim, hidden_dim)
        self.gconv2 = WindWeightedGraphConv(hidden_dim, hidden_dim)

        self.temporal_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Decoder mapping graph patches back to 128x128 resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 16, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_horizons, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, T=5, C=7, H=128, W=128]
        Output: [B, 3, H=128, W=128]
        """
        b, t, c, h, w = x.size()

        # Extract average wind vector per batch sample for graph edge modulation
        # Channel 1: wind_u, Channel 2: wind_v
        wind_vector = x[:, :, 1:3, :, :].mean(dim=[1, 3, 4])  # [B, 2]

        h_node = None

        for t_step in range(t):
            input_t = x[:, t_step, :, :, :]  # [B, C, H, W]
            patches = self.patch_encoder(input_t)  # [B, Hidden, H_p, W_p]
            b_p, c_p, h_p, w_p = patches.size()

            # Reshape to graph nodes [B, N=H_p*W_p, C]
            nodes = patches.view(b_p, c_p, -1).permute(0, 2, 1)

            # Wind-weighted Graph Convolutions
            g1 = self.gconv1(nodes, wind_vector)
            g2 = self.gconv2(g1, wind_vector)  # [B, N, Hidden]

            # Pool graph nodes to vector per sample for GRU temporal step
            graph_pooled = g2.mean(dim=1)  # [B, Hidden]

            h_node = self.temporal_gru(graph_pooled, h_node)

        # Broadcast temporal state back onto spatial grid & decode
        h_spatial = h_node.view(b, self.hidden_dim, 1, 1).expand(-1, -1, h_p, w_p)
        out_risk = self.decoder(h_spatial)  # [B, 3, 128, 128]

        return out_risk


if __name__ == "__main__":
    gnn_model = SpatiotemporalGNN(in_channels=7, patch_size=4, hidden_dim=32, out_horizons=3)
    dummy_input = torch.randn(2, 5, 7, 128, 128)
    output = gnn_model(dummy_input)
    print(f"ST-GNN Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
