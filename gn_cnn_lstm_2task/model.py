import torch
import torch.nn as nn


def _choose_num_groups(num_channels: int, preferred_groups: int) -> int:
    """Choose a valid GroupNorm num_groups for given channels.

    GroupNorm requires `num_channels % num_groups == 0`.
    We pick the largest divisor <= preferred_groups, and fall back to 1.
    """
    if num_channels <= 0:
        return 1
    if preferred_groups is None:
        return 1
    g_max = int(preferred_groups)
    if g_max <= 0:
        return 1
    g_max = min(g_max, num_channels)
    for g in range(g_max, 0, -1):
        if num_channels % g == 0:
            return g
    return 1

class GNCNNLSTM2Head(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 64,
        hidden: int = 128,
        bidirectional: bool = False,
        gn_groups: int = 8,
    ):
        super().__init__()
        g1 = _choose_num_groups(32, gn_groups)
        g2 = _choose_num_groups(int(cnn_channels), gn_groups)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=g1, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, cnn_channels, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=g2, num_channels=cnn_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head_cls = nn.Linear(out_dim, 1)
        self.head_reg = nn.Linear(out_dim, 1)

    def forward(self, X: torch.Tensor):
        B, T, F = X.shape
        x = X.reshape(B * T, 1, F)
        x = self.cnn(x).squeeze(-1)
        x = x.reshape(B, T, -1)
        out, _ = self.lstm(x)
        logits_cls = self.head_cls(out).squeeze(-1)
        logits_reg = self.head_reg(out).squeeze(-1)
        return logits_cls, logits_reg