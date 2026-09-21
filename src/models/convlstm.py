"""
Stacked ConvLSTM2D Architecture for Wildfire Spatiotemporal Risk Forecasting.

Processes 5-day rolling tensor data cubes [Batch, T=5, C=7, H=128, W=128]
and outputs multi-horizon risk probability maps [Batch, 3, H=128, W=128] for 24h, 48h, and 72h horizons.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single Convolutional LSTM cell.
    Captures joint spatial convolutions and recurrent temporal state transitions.
    """

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor: torch.Tensor, cur_state: tuple = None):
        h_cur, c_cur = cur_state if cur_state is not None else self._init_hidden(input_tensor)

        combined = torch.cat([input_tensor, h_cur], dim=1)  # [B, in_C + hidden_dim, H, W]
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def _init_hidden(self, input_tensor: torch.Tensor):
        b, _, h, w = input_tensor.size()
        h_zeros = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device)
        c_zeros = torch.zeros(b, self.hidden_dim, h, w, device=input_tensor.device)
        return h_zeros, c_zeros


class ConvLSTM2D(nn.Module):
    """
    Stacked 2-Layer ConvLSTM2D network with Multi-Horizon (24h, 48h, 72h) Conv2D output head.
    """

    def __init__(self, in_channels: int = 7, hidden_dims: list = [48, 24], out_horizons: int = 3):
        super(ConvLSTM2D, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_horizons = out_horizons

        self.cell1 = ConvLSTMCell(in_channels=in_channels, hidden_dim=hidden_dims[0], kernel_size=3)
        self.cell2 = ConvLSTMCell(in_channels=hidden_dims[0], hidden_dim=hidden_dims[1], kernel_size=3)

        # Output head mapping final hidden state to 3 hazard probability maps (24h, 48h, 72h)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dims[1], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_horizons, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, T=5, C=7, H=128, W=128]
        Output: [B, 3, H=128, W=128] risk probability grids
        """
        b, t, c, h, w = x.size()

        h1, c1 = None, None
        h2, c2 = None, None

        for t_step in range(t):
            input_t = x[:, t_step, :, :, :]  # [B, C, H, W]
            h1, c1 = self.cell1(input_t, (h1, c1) if h1 is not None else None)
            h2, c2 = self.cell2(h1, (h2, c2) if h2 is not None else None)

        out_risk = self.head(h2)  # [B, 3, H, W]
        return out_risk


if __name__ == "__main__":
    model = ConvLSTM2D(in_channels=7, hidden_dims=[32, 16], out_horizons=3)
    dummy_input = torch.randn(2, 5, 7, 128, 128)
    output = model(dummy_input)
    print(f"ConvLSTM2D Input shape: {dummy_input.shape} -> Output shape: {output.shape}")
