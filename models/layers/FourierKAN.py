import torch
from torch import nn
from typing import Optional


class FourierKAN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        grid_size: int = 5,
        width: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.width = width or (2 * in_features + 1)

        k = grid_size + 1
        self.a = nn.Parameter(torch.empty(self.width, in_features, k))
        self.b = nn.Parameter(torch.empty(self.width, in_features, k))
        self.W_out = nn.Parameter(torch.empty(out_features, self.width))
        self.bias_out = nn.Parameter(torch.zeros(out_features))

        nn.init.normal_(self.a, mean=0.0, std=0.02)
        nn.init.normal_(self.b, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.W_out)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, dim = x.shape
        k = torch.arange(self.grid_size + 1, device=x.device, dtype=x.dtype)
        x_e = x.unsqueeze(1).unsqueeze(-1)
        k_e = k.view(1, 1, 1, -1)
        ang = x_e * k_e
        cos_kx = torch.cos(ang)
        sin_kx = torch.sin(ang)
        a = self.a.unsqueeze(0)
        b = self.b.unsqueeze(0)
        phi = (a * cos_kx.unsqueeze(1) + b * sin_kx.unsqueeze(1)).sum(dim=-1)
        s = phi.sum(dim=-1)
        y = self.act(s)
        out = torch.matmul(y, self.W_out.t()) + self.bias_out
        return out
