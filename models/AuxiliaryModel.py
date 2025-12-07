#!/usr/bin/env python3
"""
旁路解耦模型 AuxiliaryModel

目标：
- 结构与主干网络保持一致的双塔嵌入与CNN（确保词表与卷积参数一致）
- 下游接入 LCWnetFootprint（提取样本级向量）→ 4层RoPE自注意 → FourierKAN
- 输出分布特征 M = [G, F, I] 以支撑解耦对抗训练

说明：
- 代码注释使用中文；绘图在训练脚本中以英文标签保存，避免中文字体问题
- 所有超参数从 config.py 集中管理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from config import (
    DNA_EMBEDDING_VOCAB_SIZE,
    DNA_EMBEDDING_DIM,
    DNA_EMBEDDING_PADDING_IDX,
    DNA_EMBEDDING_INIT_STD,
    EMBEDDING_DIM,
    CNN_KERNEL_SIZE,
    POOL_KERNEL_SIZE,
    CNN_DROPOUT,
    OUT_CHANNELS,
    TRANSFORMER_HEADS,
)

from models.pleat.embedding import create_dna_embedding_layer
from models.layers.attn import RoPEAttention
from models.layers.FourierKAN import FourierKAN
from models.layers.footprint import FootprintExpert


class SequencePooling(nn.Module):
    """序列池化（注意力权重归一化求和）"""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        xn = self.norm(x)
        w = self.proj(xn).squeeze(-1)
        if key_padding_mask is not None:
            w = w.masked_fill(key_padding_mask, -1e9)
        w = torch.softmax(w, dim=-1)
        if key_padding_mask is not None:
            w = w.masked_fill(key_padding_mask, 0.0)
        denom = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        w = w / denom
        return (x * w.unsqueeze(-1)).sum(dim=1)


class GradientReversalFn(torch.autograd.Function):
    """梯度反转层（GRL）实现：前向恒等、反向乘以 -alpha"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """梯度反转模块包装"""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.alpha)


class AuxiliaryModel(nn.Module):
    """
    旁路解耦模型

    流程：
    E/P → 词表嵌入 → CNN → LCWnetFootprint → 4层RoPE自注意 → 序列池化 → FourierKAN
    同时从FootprintExpert中得到样本级向量并投影为 [G, F, I] 用于对抗解耦训练。

    Args:
        num_cell_types: 细胞系类别数（用于特性分类器与域判别器）
        rope_layers: RoPE自注意层数（默认4层）
        rope_heads: 注意力头数（默认与主干一致）
        grl_alpha: 梯度反转系数
    """
    def __init__(
        self,
        num_cell_types: int,
        rope_layers: int = None,
        rope_heads: int = None,
        grl_alpha: float = 1.0,
    ):
        super().__init__()

        # ============= 双塔嵌入（与主干一致） =============
        self.enh_embedding = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD,
        )
        self.pr_embedding = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD,
        )

        # ============= 双塔CNN（与主干一致） =============
        self.enh_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT),
        )
        self.pr_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT),
        )

        # ============= Footprint专家（样本级向量 + 三元子空间） =============
        self.fp_enh = FootprintExpert(d_model=OUT_CHANNELS)
        self.fp_pr = FootprintExpert(d_model=OUT_CHANNELS)

        # ============= RoPE自注意堆叠（层数/头数可配置） =============
        from config import BYPASS_ROPE_LAYERS, BYPASS_ROPE_HEADS
        rope_layers = rope_layers or BYPASS_ROPE_LAYERS
        rope_heads = rope_heads or BYPASS_ROPE_HEADS
        self.rope_layers = rope_layers
        self.enh_attn = nn.ModuleList([RoPEAttention(d_model=OUT_CHANNELS, num_heads=rope_heads) for _ in range(rope_layers)])
        self.pr_attn = nn.ModuleList([RoPEAttention(d_model=OUT_CHANNELS, num_heads=rope_heads) for _ in range(rope_layers)])
        self.enh_norms = nn.ModuleList([nn.LayerNorm(OUT_CHANNELS) for _ in range(rope_layers)])
        self.pr_norms = nn.ModuleList([nn.LayerNorm(OUT_CHANNELS) for _ in range(rope_layers)])

        # ============= 序列池化与分类头（FourierKAN） =============
        self.seq_pool = SequencePooling(d_model=OUT_CHANNELS)
        self.classifier = FourierKAN(in_features=2 * OUT_CHANNELS, out_features=1, grid_size=5)

        # ============= 对抗解耦模块 =============
        # 特性分类器 D_cls: z_F -> cell_type
        self.num_cell_types = num_cell_types
        self.spec_head = nn.Sequential(
            nn.LayerNorm(max(8, OUT_CHANNELS // 8)),
            nn.Linear(max(8, OUT_CHANNELS // 8), num_cell_types)
        )
        # 域判别器 D_adv: GRL(z_G) -> cell_type
        self.grl = GradientReversal(alpha=grl_alpha)
        self.adv_head = nn.Sequential(
            nn.LayerNorm(OUT_CHANNELS),
            nn.Linear(OUT_CHANNELS, num_cell_types)
        )

    # ---------------------- 工具函数 ----------------------
    def _build_pad_mask(self, ids: torch.Tensor, L_out: int, L_orig: int) -> torch.Tensor:
        """根据输入ID的padding区域，推导卷积+池化后的序列mask"""
        pad = (ids == DNA_EMBEDDING_PADDING_IDX)
        pad_mask = torch.zeros(ids.size(0), L_out, dtype=torch.bool, device=ids.device)
        for j in range(L_out):
            s = j * POOL_KERNEL_SIZE
            e = min(s + POOL_KERNEL_SIZE + CNN_KERNEL_SIZE - 2, L_orig - 1)
            pad_mask[:, j] = pad[:, s:e+1].all(dim=-1)
        return pad_mask

    # ---------------------- 前向传播 ----------------------
    def forward(
        self,
        enhancer_ids: torch.Tensor,
        promoter_ids: torch.Tensor,
        cell_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            enhancer_ids: 增强子token索引 [B, L_en]
            promoter_ids: 启动子token索引 [B, L_pr]
            cell_labels: 细胞系标签索引 [B]（用于训练时的分类损失）

        Returns:
            (pred_prob, extras): 预测概率与附加输出（z向量、分类logits、mask等）
        """
        K = CNN_KERNEL_SIZE
        P = POOL_KERNEL_SIZE
        min_len = K + P - 1
        if enhancer_ids.size(1) < min_len:
            enhancer_ids = F.pad(enhancer_ids, (0, min_len - enhancer_ids.size(1)), value=DNA_EMBEDDING_PADDING_IDX)
        if promoter_ids.size(1) < min_len:
            promoter_ids = F.pad(promoter_ids, (0, min_len - promoter_ids.size(1)), value=DNA_EMBEDDING_PADDING_IDX)

        # 嵌入
        embed_en = self.enh_embedding(enhancer_ids)  # [B,L,D]
        embed_pr = self.pr_embedding(promoter_ids)   # [B,L,D]

        # CNN
        enh = self.enh_cnn(embed_en.permute(0, 2, 1)).permute(2, 0, 1)  # [L',B,D]→[B,L',D]
        pr = self.pr_cnn(embed_pr.permute(0, 2, 1)).permute(2, 0, 1)
        enh = enh.permute(1, 0, 2)
        pr = pr.permute(1, 0, 2)

        B = enhancer_ids.size(0)
        L_en_orig = enhancer_ids.size(1)
        L_pr_orig = promoter_ids.size(1)
        L_en = enh.size(1)
        L_pr = pr.size(1)

        # 注意力mask
        enh_pad_mask = self._build_pad_mask(enhancer_ids, L_en, L_en_orig)
        pr_pad_mask = self._build_pad_mask(promoter_ids, L_pr, L_pr_orig)
        enh_attn_mask = torch.zeros(B, 1, L_en, L_en, device=enhancer_ids.device, dtype=torch.float32)
        pr_attn_mask = torch.zeros(B, 1, L_pr, L_pr, device=promoter_ids.device, dtype=torch.float32)
        if enh_pad_mask.any():
            for b in range(B):
                cols = enh_pad_mask[b]
                if cols.any():
                    enh_attn_mask[b, 0, :, cols] = float('-inf')
        if pr_pad_mask.any():
            for b in range(B):
                cols = pr_pad_mask[b]
                if cols.any():
                    pr_attn_mask[b, 0, :, cols] = float('-inf')

        # Footprint专家：得到序列输出与样本级向量
        enh_seq, enh_vec, zG_e, zF_e, zI_e = self.fp_enh(enh)
        pr_seq, pr_vec, zG_p, zF_p, zI_p = self.fp_pr(pr)

        # 合并两个塔的子空间（简单平均）
        zG = 0.5 * (zG_e + zG_p)
        zF = 0.5 * (zF_e + zF_p)
        zI = 0.5 * (zI_e + zI_p)

        # RoPE自注意堆叠（预归一化）
        xe = enh_seq
        xp = pr_seq
        for i in range(self.rope_layers):
            xe = self.enh_norms[i](xe)
            xp = self.pr_norms[i](xp)
            xe = self.enh_attn[i](xe, attention_mask=enh_attn_mask)
            xp = self.pr_attn[i](xp, attention_mask=pr_attn_mask)

        # 序列池化
        he = self.seq_pool(xe, key_padding_mask=enh_pad_mask)
        hp = self.seq_pool(xp, key_padding_mask=pr_pad_mask)
        h = torch.cat([he, hp], dim=-1)

        # KAN分类头（用于可选的监督信号）
        logits = self.classifier(h)
        pred_prob = torch.sigmoid(logits)

        # 特性分类器（F）
        spec_logits = self.spec_head(zF)
        # 域判别器（G，通过GRL）
        adv_logits = self.adv_head(self.grl(zG))

        extras = {
            'zG': zG, 'zF': zF, 'zI': zI,
            'spec_logits': spec_logits,
            'adv_logits': adv_logits,
            'enh_pad_mask': enh_pad_mask,
            'pr_pad_mask': pr_pad_mask,
        }

        if cell_labels is not None:
            # 分类损失与准确率（供训练脚本使用）
            spec_loss = F.cross_entropy(spec_logits, cell_labels)
            with torch.no_grad():
                spec_acc = (spec_logits.argmax(dim=-1) == cell_labels).float().mean()
            adv_loss = F.cross_entropy(adv_logits, cell_labels)
            with torch.no_grad():
                adv_acc = (adv_logits.argmax(dim=-1) == cell_labels).float().mean()
            extras['spec_loss'] = spec_loss
            extras['spec_acc'] = spec_acc
            extras['adv_loss'] = adv_loss
            extras['adv_acc'] = adv_acc

        return pred_prob, extras
