"""Grid-graph spatiotemporal neural network for wildfire risk forecasting.

Input:  [B, T, C, H, W]
Output: [B, 3, H, W] probabilities for 24h, 48h, and 72h.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridGraphConv(nn.Module):
    """Message-passing layer over a regular 2-D grid graph."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features, bias=False)
        self.neighbor_linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.norm = nn.LayerNorm(out_features)

    def forward(self, nodes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, node_count, channels = nodes.shape
        if node_count != height * width:
            raise ValueError("Node count does not match grid dimensions.")

        grid = nodes.transpose(1, 2).reshape(batch, channels, height, width)
        kernel = torch.ones((1, 1, 3, 3), device=grid.device, dtype=grid.dtype)
        neighbor_sum = F.conv2d(
            grid, kernel.expand(channels, 1, 3, 3), padding=1, groups=channels
        )
        count_map = F.conv2d(
            torch.ones((batch, 1, height, width), device=grid.device, dtype=grid.dtype),
            kernel,
            padding=1,
        ).clamp_min(1.0)
        neighbor_mean = neighbor_sum / count_map
        neighbor_nodes = neighbor_mean.flatten(2).transpose(1, 2)

        output = (
            self.self_linear(nodes)
            + self.neighbor_linear(neighbor_nodes)
            + self.bias
        )
        return F.relu(self.norm(output))


class SpatiotemporalGNN(nn.Module):
    """Temporal grid-graph model with node-wise recurrent state."""

    def __init__(
        self,
        in_channels: int = 7,
        patch_size: int = 4,
        hidden_dim: int = 32,
        out_horizons: int = 3,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1.")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.out_horizons = out_horizons

        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gconv1 = GridGraphConv(hidden_dim, hidden_dim)
        self.gconv2 = GridGraphConv(hidden_dim, hidden_dim)
        self.temporal_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.wind_fc = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 16, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_horizons, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,T,C,H,W], got {tuple(x.shape)}.")
        batch, timesteps, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Input height and width must be divisible by patch_size.")

        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        hidden_nodes = None

        for time_index in range(timesteps):
            encoded = self.patch_encoder(x[:, time_index])
            nodes = encoded.flatten(2).transpose(1, 2)
            nodes = self.gconv1(nodes, patch_h, patch_w)
            nodes = self.gconv2(nodes, patch_h, patch_w)

            if hidden_nodes is None:
                hidden_nodes = torch.zeros_like(nodes)

            hidden_nodes = self.temporal_gru(
                nodes.reshape(batch * patch_h * patch_w, self.hidden_dim),
                hidden_nodes.reshape(batch * patch_h * patch_w, self.hidden_dim),
            ).reshape(batch, patch_h * patch_w, self.hidden_dim)

        wind_vector = x[:, :, 1:3, :, :].mean(dim=(1, 3, 4))
        wind_modulation = self.wind_fc(wind_vector).unsqueeze(1)
        hidden_nodes = hidden_nodes * (1.0 + wind_modulation)

        hidden_grid = hidden_nodes.transpose(1, 2).reshape(
            batch, self.hidden_dim, patch_h, patch_w
        )
        output = self.decoder(hidden_grid)
        if output.shape[-2:] != (height, width):
            output = F.interpolate(
                output, size=(height, width), mode="bilinear", align_corners=False
            )
            output = torch.sigmoid(output)
        return output


if __name__ == "__main__":
    model = SpatiotemporalGNN(in_channels=7, patch_size=4, hidden_dim=32, out_horizons=3)
    dummy_input = torch.randn(2, 5, 7, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"ST-GNN Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
