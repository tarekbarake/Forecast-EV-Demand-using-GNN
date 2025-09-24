import torch
import torch.nn as nn
from typing import Tuple

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # Supports H shaped [N, F] or [B, N, F]; A_hat is [N, N]
        H = self.dropout(H)
        if H.dim() == 2:
            H = A_hat @ H                  # [N, F]
            H = self.lin(H)                # [N, out]
        else:
            # Broadcast A_hat over batch: result [B, N, F]
            H = torch.matmul(A_hat, H)     # [B, N, F]
            H = self.lin(H)                # [B, N, out]
        return self.act(H)

class TemporalConv(nn.Module):
    def __init__(self, in_feat: int, hidden: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_feat, out_channels=hidden, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        x = x.permute(0, 2, 3, 1)      # [B, N, F, T]
        x = x.reshape(B*N, F, T)       # [B*N, F, T]
        x = self.conv(x)               # [B*N, H, T]
        x = self.act(x)
        x = x[:, :, -1]                # [B*N, H]
        x = x.reshape(B, N, -1)        # [B, N, H]
        return x

class STGCN(nn.Module):
    def __init__(self, in_features: int, gcn_hidden: int, gcn_layers: int, temporal_hidden: int, dropout: float):
        super().__init__()
        self.temporal = TemporalConv(in_features, temporal_hidden)
        gcn = []
        gcn.append(GCNLayer(temporal_hidden, gcn_hidden, dropout))
        for _ in range(gcn_layers - 1):
            gcn.append(GCNLayer(gcn_hidden, gcn_hidden, dropout))
        self.gcn = nn.ModuleList(gcn)
        self.head_mean = nn.Linear(gcn_hidden, 1)
        self.head_logvar = nn.Linear(gcn_hidden, 1)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, N, F]
        out = self.temporal(x)  # [B, N, Ht]
        for layer in self.gcn:
            out = layer(out, A_hat)   # keep [B, N, *] shape
        mean = self.head_mean(out).squeeze(-1)     # [B, N]
        logvar = self.head_logvar(out).squeeze(-1) # [B, N]
        return mean, logvar

def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar).clamp_min(1e-6)
    return 0.5 * (torch.log(var) + (target - mean)**2 / var)
