"""
PRISM 组件：主干与细胞系分类专家
"""

from torch import nn
import torch
import torch.nn.functional as F
from config import *
from models.pleat.embedding import create_dna_embedding_layer
from models.pleat.RoPE import RoPEConfig
from models.layers.footprint import FootprintConfig, FootprintExpert
from models.layers.attn import *
from models.layers.FourierKAN import FourierKAN
from models.layers.masking import SegmentMaskBuilder
from models.pleat.adaptive_immax import AdaptiveIMMAXLoss
from models.layers.footprint import FootprintExpert
from models.EPIModel import CBATTransformerEncoderLayer
from config import *


class AttnPool1d(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.v = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.v, mean=0.0, std=0.02)
        self.drop = nn.Dropout(CNN_DROPOUT)
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        h = torch.tanh(self.proj(self.drop(x)))
        s = (h * self.v).sum(-1)
        s = s.masked_fill(mask, -1e9)
        w = torch.softmax(s, dim=-1)
        w = w.masked_fill(mask, 0.0)
        norm = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        w = w / norm
        return (x * w.unsqueeze(-1)).sum(dim=1)

class AttnPool1dWindow(nn.Module):
    def __init__(self, d: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.attn = AttnPool1d(d)
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, C, L = x.shape
        step = self.stride
        win = self.kernel_size
        if L < win:
            return x.new_zeros(B, C, 0)
        L_pool = 1 + (L - win) // step
        outs = []
        for j in range(L_pool):
            start = j * step
            end = start + win
            xw = x[:, :, start:end].permute(0, 2, 1)
            mw = mask[:, start:end]
            ow = self.attn(xw, mw)
            outs.append(ow.unsqueeze(-1))
        if len(outs) == 0:
            return x.new_zeros(B, C, 0)
        return torch.cat(outs, dim=-1)

class FusionGate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
            nn.Sigmoid(),
        )
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f1, f2, f1 - f2, f1 * f2], dim=-1)
        g = self.net(x)
        return g * f1 + (1.0 - g) * f2

# 移除旧版PRISMModel（MLM预训练）及其相关函数，统一由PRISMBackbone和CellClassificationExpert组成


class PRISMBackbone(nn.Module):
    def __init__(self, num_classes: int = None):
        super().__init__()
        self.num_transformer_layers = TRANSFORMER_LAYERS
        TRANSFORMER_DROPOUT = RoPEConfig.ROPE_DROPOUT
        img_size = PRISM_IMG_SIZE

        self.enh_embedding = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )
        self.pr_embedding = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )
        self.enh_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        self.pr_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        self.pre_enh_attn = RoPEAttention(
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )
        self.pre_pr_attn = RoPEAttention(
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )
        self.cbat_layers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,
                img_size=img_size
            ) for _ in range(self.num_transformer_layers)
        ])
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )
        self.post_cbat = CBAT(
            d_model=OUT_CHANNELS,
            num_heads=TRANSFORMER_HEADS,
            img_size=img_size,
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,
            dropout=TRANSFORMER_DROPOUT,
        )
        self.attn_pool_en = AttnPool1d(OUT_CHANNELS)
        self.attn_pool_pr = AttnPool1d(OUT_CHANNELS)
        self.classifier = FourierKAN(
            in_features=OUT_CHANNELS * 2,
            out_features=1,
            grid_size=5,
            width=2 * (OUT_CHANNELS * 2) + 1,
        )
        self.cell_inject_proj = nn.Sequential(
            nn.LayerNorm(OUT_CHANNELS * 2),
            nn.Linear(OUT_CHANNELS * 2, OUT_CHANNELS * 2),
            nn.GELU()
        )
        self.cell_alpha = nn.Parameter(torch.tensor(0.0))
        self.criterion = AdaptiveIMMAXLoss()
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.enhancer_footprint = FootprintExpert(d_model=OUT_CHANNELS)
        self.promoter_footprint = FootprintExpert(d_model=OUT_CHANNELS)
        self.fusion_gate = FusionGate(OUT_CHANNELS)
        self.inj_proj = nn.Sequential(
            nn.LayerNorm(OUT_CHANNELS),
            nn.Linear(OUT_CHANNELS, OUT_CHANNELS),
            nn.GELU()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, enhancer_ids, promoter_ids, cell_vec=None):
        K = CNN_KERNEL_SIZE
        P = POOL_KERNEL_SIZE
        min_required_length = K + P - 1
        if enhancer_ids.size(1) < min_required_length:
            enhancer_ids = F.pad(enhancer_ids, (0, min_required_length - enhancer_ids.size(1)), value=DNA_EMBEDDING_PADDING_IDX)
        if promoter_ids.size(1) < min_required_length:
            promoter_ids = F.pad(promoter_ids, (0, min_required_length - promoter_ids.size(1)), value=DNA_EMBEDDING_PADDING_IDX)

        embed_en = self.enh_embedding(enhancer_ids)
        embed_pr = self.pr_embedding(promoter_ids)
        enh = self.enh_cnn(embed_en.permute(0, 2, 1))
        pr = self.pr_cnn(embed_pr.permute(0, 2, 1))
        enh = enh.permute(2, 0, 1)
        pr = pr.permute(2, 0, 1)

        enh_pre = self.pre_enh_attn(enh.permute(1, 0, 2))
        enh_for_fp = enh_pre
        _, fp1_vec, _, _ = self.enhancer_footprint(enh_for_fp)
        enh = enh_pre.permute(1, 0, 2)
        pr_pre = self.pre_pr_attn(pr.permute(1, 0, 2))
        pr = pr_pre.permute(1, 0, 2)

        att1, _ = self.cross_attn_1(enh, pr, pr)
        enh = enh + att1
        enh_cross = enh.permute(1, 0, 2)
        _, fp2_vec, _, _ = self.enhancer_footprint(enh_cross)
        fused = self.fusion_gate(fp1_vec, fp2_vec)
        proj = self.inj_proj(fused).unsqueeze(0).expand(enh.shape[0], -1, -1)
        enh = enh + self.alpha * proj
        residual_proj = proj

        total_adaptive_loss = 0.0
        for layer in self.cbat_layers:
            enh, layer_loss = layer(enh)
            total_adaptive_loss += layer_loss

        att2, _ = self.cross_attn_2(enh, pr, pr)
        enh = enh + att2 + self.alpha * residual_proj

        post_out, post_loss = self.post_cbat(enh.permute(1, 0, 2), return_loss=True)
        total_adaptive_loss += post_loss
        enh = post_out.permute(1, 0, 2)

        enh_pooled = nn.AdaptiveAvgPool1d(1)(enh.permute(1, 2, 0)).squeeze(-1)
        pr_pooled = nn.AdaptiveAvgPool1d(1)(pr.permute(1, 2, 0)).squeeze(-1)
        combined = torch.cat([enh_pooled, pr_pooled], dim=1)
        combined = F.layer_norm(combined, combined.shape[-1:])
        if cell_vec is not None:
            inj = self.cell_inject_proj(cell_vec)
            combined = combined + self.cell_alpha * F.layer_norm(inj, inj.shape[-1:])
        result = self.classifier(combined)
        return torch.sigmoid(result), total_adaptive_loss

    def compute_loss(self, outputs, labels, adaptive_loss=0.0):
        loss = self.criterion(outputs, labels)
        return loss + adaptive_loss


class CellClassificationExpert(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.enh_embedding = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )
        self.pr_embedding = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )
        self.enh_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        self.pr_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        self.pre_enh_attn = RoPEAttention(d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=RoPEConfig.ROPE_DROPOUT)
        self.pre_pr_attn = RoPEAttention(d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=RoPEConfig.ROPE_DROPOUT)
        self.enhancer_footprint = FootprintExpert(d_model=OUT_CHANNELS)
        self.promoter_footprint = FootprintExpert(d_model=OUT_CHANNELS)
        self.fusion_gate = FusionGate(OUT_CHANNELS)
        self.inj_proj = nn.Sequential(
            nn.LayerNorm(OUT_CHANNELS),
            nn.Linear(OUT_CHANNELS, OUT_CHANNELS),
            nn.GELU()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        img_size = PRISM_IMG_SIZE
        self.transformers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=RoPEConfig.ROPE_DROPOUT,
                img_size=img_size
            ) for _ in range(4)
        ])
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = FourierKAN(
            in_features=OUT_CHANNELS * 2,
            out_features=num_classes,
            grid_size=5,
            width=2 * (OUT_CHANNELS * 2) + 1,
        )

    def forward(self, enh_ids, pr_ids):
        min_required_length = 59
        if enh_ids.size(1) < min_required_length:
            padding_size = min_required_length - enh_ids.size(1)
            enh_ids = F.pad(enh_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)
        if pr_ids.size(1) < min_required_length:
            padding_size = min_required_length - pr_ids.size(1)
            pr_ids = F.pad(pr_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)
        embed_en = self.enh_embedding(enh_ids)
        embed_pr = self.pr_embedding(pr_ids)
        enh = self.enh_cnn(embed_en.permute(0, 2, 1))
        pr = self.pr_cnn(embed_pr.permute(0, 2, 1))
        enh = enh.permute(2, 0, 1)
        pr = pr.permute(2, 0, 1)
        enh_pre = self.pre_enh_attn(enh.permute(1, 0, 2))
        enh_for_fp = enh_pre
        fp1_seq, fp1_vec, _, _ = self.enhancer_footprint(enh_for_fp)
        enh = enh_pre.permute(1, 0, 2)
        pr_pre = self.pre_pr_attn(pr.permute(1, 0, 2))
        pr = pr_pre.permute(1, 0, 2)
        att1, _ = self.cross_attn(enh, pr, pr)
        enh = enh + att1
        enh_cross = enh.permute(1, 0, 2)
        fp2_seq, fp2_vec, _, _ = self.enhancer_footprint(enh_cross)
        fused = self.fusion_gate(fp1_vec, fp2_vec)
        proj = self.inj_proj(fused).unsqueeze(0).expand(enh.shape[0], -1, -1)
        enh = enh + self.alpha * proj
        residual_proj = proj
        total_loss = 0.0
        for layer in self.transformers:
            enh, layer_loss = layer(enh)
            total_loss += layer_loss
        att2, _ = self.cross_attn(enh, pr, pr)
        enh = enh + att2 + self.alpha * residual_proj
        enh_pooled = self.pool(enh.permute(1, 2, 0)).squeeze(-1)
        pr_pooled = self.pool(pr.permute(1, 2, 0)).squeeze(-1)
        combined = torch.cat([enh_pooled, pr_pooled], dim=1)
        logits = self.classifier(combined)
        return logits, combined
