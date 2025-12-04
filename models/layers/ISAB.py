"""
Induced Set Attention Block (ISAB)

ISAB uses a set of learnable inducing points to reduce attention complexity
from O(n^2) (SAB) to O(m·n), where m is the number of inducing points and
n is the set size. It first attends from inducing points to the input set,
then attends from the input set to the induced representation.

Design goals:
- Standalone, minimal dependency
- Clear typed interface and maintainable structure
- Mask-aware for padded tokens

Usage:
    x_out = ISAB(d_model=64, num_heads=8, num_inducing=32)(x, key_padding_mask)

Args:
    d_model: Feature dimension
    num_heads: Attention heads
    num_inducing: Number of inducing points (m)
    dropout: Dropout probability
"""

from typing import Optional

import torch
from torch import nn


class ISAB(nn.Module):
    """Induced Set Attention Block

    Implements two-stage attention:
    1) H = MHA(I, X, X), queries are inducing points I, keys/values are X
    2) Y = MHA(X, H, H), queries are X, keys/values are induced H

    Shapes:
        X: [B, L, D], key_padding_mask: [B, L] with True for padding
        Returns: [B, L, D]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model  # feature dim
        self.num_heads = num_heads  # heads
        self.num_inducing = num_inducing  # m

        # Learnable inducing points [m, D]
        self.inducing = nn.Parameter(torch.empty(num_inducing, d_model))  # inducing points
        nn.init.xavier_uniform_(self.inducing)  # init

        # Multihead attention (batch_first)
        self.mha_I = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)  # I->X
        self.mha_X = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)  # X->H

        # Feedforward and norms
        self.ffn = nn.Sequential(  # FFN
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm_in = nn.LayerNorm(d_model)  # pre-norm 1
        self.norm_mid = nn.LayerNorm(d_model)  # pre-norm 2
        self.norm_out = nn.LayerNorm(d_model)  # post-norm
        self.drop = nn.Dropout(dropout)  # dropout

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input set `[B, L, D]`
            key_padding_mask: Optional boolean mask `[B, L]`, True marks padding

        Returns:
            Tensor of shape `[B, L, D]`
        """
        B, L, D = x.shape  # shapes

        # Stage 1: H = MHA(I, X, X)
        x_norm = self.norm_in(x)  # [B, L, D]
        I = self.inducing.unsqueeze(0).expand(B, self.num_inducing, D)  # [B, m, D]
        H, _ = self.mha_I(query=I, key=x_norm, value=x_norm, key_padding_mask=key_padding_mask)  # [B, m, D]

        # Stage 2: Y = MHA(X, H, H)
        x_mid = self.norm_mid(x_norm)  # [B, L, D]
        Y, _ = self.mha_X(query=x_mid, key=H, value=H)  # [B, L, D]

        # Residual + FFN
        x = x + self.drop(Y)  # residual 1
        x = x + self.drop(self.ffn(self.norm_out(x)))  # residual 2

        return x  # [B, L, D]

