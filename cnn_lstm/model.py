import torch.nn as nn
import torch

class CNNLSTM(nn.Module):
    def __init__(self, input_dim: int, cnn_channels: int = 64, hidden: int = 128, bidirectional: bool = False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
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
        self.head = nn.Linear(out_dim, 1)

    def forward(self, X: torch.Tensor):
        # X: [B, T, F]
        B, T, F = X.shape
        x = X.reshape(B * T, 1, F)          # [B*T, 1, F]
        x = self.cnn(x).squeeze(-1)         # [B*T, C]
        x = x.reshape(B, T, -1)             # [B, T, C]
        out, _ = self.lstm(x)               # [B, T, H]
        logits = self.head(out).squeeze(-1) # [B, T]
        return logits