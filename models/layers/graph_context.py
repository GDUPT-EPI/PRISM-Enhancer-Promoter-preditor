import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import (
    GCN_PROTOS_PER_CELL,
    GCN_HIDDEN_DIM,
    GCN_LAYERS,
    GCN_SIM_TAU,
    GCN_CENTER_LOSS_W,
    GCN_MARGIN,
)


class GraphContext(nn.Module):
    def __init__(self, num_cells: int, d_spec: int):
        super().__init__()
        self.num_cells = num_cells
        self.d_spec = d_spec
        self.num_protos = GCN_PROTOS_PER_CELL
        self.tau = GCN_SIM_TAU
        self.prototypes = nn.Parameter(torch.randn(num_cells, self.num_protos, d_spec) * 0.02)
        layers = []
        in_dim = d_spec
        for i in range(max(1, GCN_LAYERS)):
            out_dim = GCN_HIDDEN_DIM if i < GCN_LAYERS - 1 else d_spec
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            in_dim = out_dim
        self.gcns = nn.ModuleList(layers)

    def _build_adjacency(self, P: torch.Tensor) -> torch.Tensor:
        N = P.size(0)
        W = torch.zeros(N, N, device=P.device, dtype=P.dtype)
        idx = torch.arange(N, device=P.device)
        cell_ids = (idx // self.num_protos)
        for c in range(self.num_cells):
            mask = (cell_ids == c)
            if mask.sum() <= 1:
                continue
            Pc = P[mask]
            d = torch.cdist(Pc, Pc, p=2)
            S = torch.exp(-(d.pow(2)) / max(self.tau, 1e-6))
            W[mask][:, mask] = S
        W = W + torch.eye(N, device=P.device, dtype=P.dtype)
        d = W.sum(dim=-1)
        d_inv_sqrt = torch.pow(d.clamp(min=1e-6), -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        Ahat = D_inv_sqrt @ W @ D_inv_sqrt
        return Ahat

    def _laplacian(self, W: torch.Tensor) -> torch.Tensor:
        d = W.sum(dim=-1)
        D = torch.diag(d)
        L = D - W
        d_inv_sqrt = torch.pow(d.clamp(min=1e-6), -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        Lsym = D_inv_sqrt @ L @ D_inv_sqrt
        return Lsym

    def _gcn(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Ahat = self._build_adjacency(P)
        H = P
        for i, layer in enumerate(self.gcns):
            H = Ahat @ H
            H = layer(H)
            if i < len(self.gcns) - 1:
                H = F.relu(H)
        Lsym = self._laplacian(Ahat)
        return H, Lsym

    def forward(self, zF: torch.Tensor, cell_labels: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = zF.size(0)
        P = self.prototypes.view(self.num_cells * self.num_protos, self.d_spec)
        H, Lsym = self._gcn(P)
        Hc = H.view(self.num_cells, self.num_protos, self.d_spec)
        mu = Hc.mean(dim=1)
        if cell_labels is None:
            mu_b = mu.mean(dim=0, keepdim=True).expand(B, -1)
        else:
            mu_b = mu.index_select(0, cell_labels).view(B, self.d_spec)
        smooth = torch.trace(H.t() @ Lsym @ H) / max(H.size(0), 1)
        if cell_labels is None:
            center = torch.tensor(0.0, device=zF.device, dtype=zF.dtype)
        else:
            center = F.mse_loss(zF, mu_b)
        dists = torch.cdist(mu, mu, p=2)
        m = GCN_MARGIN
        margin_mask = ~torch.eye(self.num_cells, dtype=torch.bool, device=mu.device)
        margins = (m - dists)[margin_mask]
        margin = torch.relu(margins).mean() if margins.numel() > 0 else torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
        return mu_b, smooth, center, margin

