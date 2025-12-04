"""
PRISM组件

概述:
- 主干网络(`PRISMBackbone`)用于EP互作概率预测

说明:
- 已移除专家网络（CellClassificationExpert）的具体实现，后续重构留空
"""

from torch import nn
import torch
import torch.nn.functional as F
from config import *
from models.pleat.embedding import create_dna_embedding_layer
from models.pleat.RoPE import RoPEConfig
from models.layers.footprint import FootprintExpert
from models.layers.attn import *
from models.layers.FourierKAN import FourierKAN
from models.pleat.masking import SegmentMaskBuilder
from models.pleat.adaptive_immax import AdaptiveIMMAXLoss
from models.pleat.SpeculationPenalty import SpeculationPenaltyLoss
from models.layers.footprint import FootprintExpert
from typing import Optional, Tuple


class CBATTransformerEncoderLayer(nn.Module):
    """
    使用CBAT模块的Transformer编码器层
    替换标准的多头自注意力为CBAT注意力机制
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, img_size=32):
        super(CBATTransformerEncoderLayer, self).__init__()
        self.self_attn = CBAT(
            d_model=d_model,  # 模型维度
            num_heads=nhead,  # 注意力头数
            img_size=img_size,  # CBAT图像大小
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,  # RoPE最大序列长度
            dropout=dropout  # Dropout比例
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 前馈层1
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 前馈层2
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化1
        self.norm2 = nn.LayerNorm(d_model)  # 层归一化2
        self.dropout1 = nn.Dropout(dropout)  # 残差Dropout1
        self.dropout2 = nn.Dropout(dropout)  # 残差Dropout2
        self.activation = nn.ReLU()  # 激活函数
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):  # 编码器层前向
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (seq_len, batch, d_model)
            src_mask: 注意力掩码
            src_key_padding_mask: key填充掩码
            
        Returns:
            返回 (输出张量, adaptive_loss)
            输出张量形状 (seq_len, batch, d_model)
        """
        # 保存原始输入用于残差连接
        residual = x  # 残差1
        
        # 自注意力计算 (使用CBAT)
        # 转换维度以适配CBAT: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        x_transposed = x.permute(1, 0, 2)  # 转换到(batch, seq, d)
        
        # CBAT返回(output, adaptive_loss)
        attn_output, adaptive_loss = self.self_attn(x_transposed, attention_mask=src_mask, return_loss=True)  # CBAT输出
            
        # 转回原始维度: (batch, seq_len, d_model) -> (seq_len, batch, d_model)
        attn_output = attn_output.permute(1, 0, 2)  # 转回(seq, batch, d)
        
        # 残差连接和层归一化
        x = residual + self.dropout1(attn_output)  # 残差加权
        x = self.norm1(x)  # 归一化1
        
        # 保存用于第二个残差连接
        residual = x  # 残差2
        
        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))  # 前馈输出
        
        # 残差连接和层归一化
        x = residual + self.dropout2(ff_output)  # 残差加权
        x = self.norm2(x)  # 归一化2
        
        return x, adaptive_loss

class AttnPool1d(nn.Module):
    def __init__(self, d: int):  # 1D注意力池化
        super().__init__()
        self.proj = nn.Linear(d, d)  # 投影层
        self.v = nn.Parameter(torch.zeros(d))  # 注意力向量
        nn.init.normal_(self.v, mean=0.0, std=0.02)  # 初始化
        self.drop = nn.Dropout(CNN_DROPOUT)  # Dropout
    def forward(self, x: torch.Tensor, mask: torch.Tensor):  # 前向
        h = torch.tanh(self.proj(self.drop(x)))  # 非线性投影
        s = (h * self.v).sum(-1)  # 注意力分数
        s = s.masked_fill(mask, -1e9)  # 掩码填充
        w = torch.softmax(s, dim=-1)  # 权重
        w = w.masked_fill(mask, 0.0)  # 掩码权重置零
        norm = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # 归一化因子
        w = w / norm  # 归一化
        return (x * w.unsqueeze(-1)).sum(dim=1)  # 加权求和

class AttnPool1dWindow(nn.Module):
    def __init__(self, d: int, kernel_size: int, stride: int):  # 滑窗注意力池化
        super().__init__()
        self.kernel_size = kernel_size  # 窗长
        self.stride = stride  # 步长
        self.attn = AttnPool1d(d)  # 注意力池化
    def forward(self, x: torch.Tensor, mask: torch.Tensor):  # 前向
        B, C, L = x.shape  # 维度
        step = self.stride  # 步长
        win = self.kernel_size  # 窗长
        if L < win:
            return x.new_zeros(B, C, 0)  # 长度不足返回空
        L_pool = 1 + (L - win) // step  # 片段数
        outs = []  # 输出列表
        for j in range(L_pool):  # 遍历窗口
            start = j * step  # 起点
            end = start + win  # 终点
            xw = x[:, :, start:end].permute(0, 2, 1)  # 片段
            mw = mask[:, start:end]  # 掩码片段
            ow = self.attn(xw, mw)  # 注意力池化
            outs.append(ow.unsqueeze(-1))  # 追加
        if len(outs) == 0:
            return x.new_zeros(B, C, 0)  # 无输出返回空
        return torch.cat(outs, dim=-1)  # 拼接

class FusionGate(nn.Module):
    def __init__(self, d: int):  # 双分支融合门控
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d, 2 * d),  # 组合特征
            nn.GELU(),  # 激活
            nn.Linear(2 * d, d),  # 回投影
            nn.Sigmoid(),  # 门控权重
        )
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:  # 前向
        x = torch.cat([f1, f2, f1 - f2, f1 * f2], dim=-1)  # 组合
        g = self.net(x)  # 门控
        return g * f1 + (1.0 - g) * f2  # 融合

# 统一由PRISMBackbone和CellClassificationExpert组成

class PRISMBackbone(nn.Module):
    """PRISM主干网络
    
    编码E/P序列、建模跨序列注意力、融合足迹特征，输出互作概率。
    
    Args:
        num_classes: 预留参数，当前未使用。
    """
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
        )  # 增强子预注意力
        self.pre_pr_attn = RoPEAttention(
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )  # 启动子预注意力
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )  # 第一次跨序列注意力
        self.cbat_layers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,
                img_size=img_size
            ) for _ in range(self.num_transformer_layers)
        ])  # 增强子CBAT层
        self.pr_cbat_layers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,
                img_size=img_size
            ) for _ in range(self.num_transformer_layers)
        ])  # 启动子CBAT层
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )  # 第二次跨序列注意力
        self.post_cbat = CBAT(
            d_model=OUT_CHANNELS,
            num_heads=TRANSFORMER_HEADS,
            img_size=img_size,
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,
            dropout=TRANSFORMER_DROPOUT,
        )  # 末端CBAT
        self.attn_pool_en = AttnPool1d(OUT_CHANNELS)  # 增强子池化
        self.attn_pool_pr = AttnPool1d(OUT_CHANNELS)  # 启动子池化
        self.classifier = FourierKAN(
            in_features=OUT_CHANNELS * 2,
            out_features=1,
            grid_size=5,
            width=2 * (OUT_CHANNELS * 2) + 1,
        )  # KAN分类头
        self.criterion = AdaptiveIMMAXLoss()  # 基础损失
        self.spec_penalty = SpeculationPenaltyLoss()  # 投机惩罚
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.enhancer_footprint = FootprintExpert(d_model=OUT_CHANNELS)  # 增强子足迹
        self.promoter_footprint = FootprintExpert(d_model=OUT_CHANNELS)  # 启动子足迹
        self.fusion_gate = FusionGate(OUT_CHANNELS)
        self.inj_proj = nn.Sequential(
            nn.LayerNorm(OUT_CHANNELS),  # 归一化
            nn.Linear(OUT_CHANNELS, OUT_CHANNELS),  # 投影
            nn.GELU()  # 激活
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        enhancer_ids: torch.Tensor,
        promoter_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        编码E/P序列、进行跨序列交互与CBAT增强，输出互作概率。

        Args:
            enhancer_ids: 增强子序列ID，形状 `[B, L_en]`
            promoter_ids: 启动子序列ID，形状 `[B, L_pr]`

        Returns:
            (prob, adaptive_loss): 概率与自适应注意力损失
        """
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

        B_en = enhancer_ids.size(0)
        L_en_orig = enhancer_ids.size(1)
        L_en = enh.size(0)
        pad_en = (enhancer_ids == DNA_EMBEDDING_PADDING_IDX)
        enh_pad_mask = torch.zeros(B_en, L_en, dtype=torch.bool, device=enhancer_ids.device)
        for j in range(L_en):
            s = j * POOL_KERNEL_SIZE
            e = min(s + POOL_KERNEL_SIZE + CNN_KERNEL_SIZE - 2, L_en_orig - 1)
            enh_pad_mask[:, j] = pad_en[:, s:e+1].all(dim=-1)
        enh_attn_mask = torch.zeros(B_en, 1, L_en, L_en, device=enhancer_ids.device, dtype=torch.float32)
        if enh_pad_mask.any():
            mask_cols = enh_pad_mask
            for b in range(B_en):
                cols = mask_cols[b]
                if cols.any():
                    enh_attn_mask[b, 0, :, cols] = float('-inf')
        enh_pre = self.pre_enh_attn(enh.permute(1, 0, 2), attention_mask=enh_attn_mask)
        enh_for_fp = enh_pre
        _, fp1_vec, _, _ = self.enhancer_footprint(enh_for_fp)
        enh = enh_pre.permute(1, 0, 2)
        B_pr = promoter_ids.size(0)
        L_pr_orig = promoter_ids.size(1)
        L_pr = pr.size(0)
        pad_pr = (promoter_ids == DNA_EMBEDDING_PADDING_IDX)
        pr_pad_mask = torch.zeros(B_pr, L_pr, dtype=torch.bool, device=promoter_ids.device)
        for j in range(L_pr):
            s = j * POOL_KERNEL_SIZE
            e = min(s + POOL_KERNEL_SIZE + CNN_KERNEL_SIZE - 2, L_pr_orig - 1)
            pr_pad_mask[:, j] = pad_pr[:, s:e+1].all(dim=-1)
        pr_attn_mask = torch.zeros(B_pr, 1, L_pr, L_pr, device=promoter_ids.device, dtype=torch.float32)
        if pr_pad_mask.any():
            for b in range(B_pr):
                cols = pr_pad_mask[b]
                if cols.any():
                    pr_attn_mask[b, 0, :, cols] = float('-inf')
        pr_pre = self.pre_pr_attn(pr.permute(1, 0, 2), attention_mask=pr_attn_mask)
        pr = pr_pre.permute(1, 0, 2)

        att1, _ = self.cross_attn_1(enh, pr, pr, key_padding_mask=pr_pad_mask)
        enh = enh + att1
        enh_cross = enh.permute(1, 0, 2)
        _, fp2_vec, _, _ = self.enhancer_footprint(enh_cross)
        fused = self.fusion_gate(fp1_vec, fp2_vec)
        proj = self.inj_proj(fused).unsqueeze(0).expand(enh.shape[0], -1, -1)
        enh = enh + self.alpha * proj
        residual_proj = proj

        total_adaptive_loss = 0.0
        for layer in self.cbat_layers:
            enh, layer_loss = layer(enh, src_mask=enh_attn_mask)
            total_adaptive_loss += layer_loss
        for layer in self.pr_cbat_layers:
            pr, layer_loss_pr = layer(pr, src_mask=pr_attn_mask)
            total_adaptive_loss += layer_loss_pr

        att2, _ = self.cross_attn_2(enh, pr, pr, key_padding_mask=pr_pad_mask)
        enh = enh + att2

        post_out, post_loss = self.post_cbat(enh.permute(1, 0, 2), return_loss=True)
        total_adaptive_loss += post_loss
        enh = post_out.permute(1, 0, 2)

        x_en = enh.permute(1, 0, 2)
        w_en = (~enh_pad_mask).float()
        s_en = (x_en * w_en.unsqueeze(-1)).sum(dim=1)
        d_en = w_en.sum(dim=1).clamp(min=1e-6)
        enh_pooled = s_en / d_en.unsqueeze(-1)
        x_pr = pr.permute(1, 0, 2)
        w_pr = (~pr_pad_mask).float()
        s_pr = (x_pr * w_pr.unsqueeze(-1)).sum(dim=1)
        d_pr = w_pr.sum(dim=1).clamp(min=1e-6)
        pr_pooled = s_pr / d_pr.unsqueeze(-1)
        combined = torch.cat([enh_pooled, pr_pooled], dim=1)  # 拼接池化向量
        combined = F.layer_norm(combined, combined.shape[-1:])  # 归一化
        result = self.classifier(combined)  # KAN分类
        return torch.sigmoid(result), total_adaptive_loss

    def compute_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        adaptive_loss: torch.Tensor | float = 0.0,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """计算总损失

        组成：`AdaptiveIMMAX`基础损失 + 自适应注意力损失 + 投机惩罚损失。

        Args:
            outputs: 预测概率 `[B]` 或 `[B,1]`
            labels: 二分类标签 `[B]`
            adaptive_loss: 自适应注意力损失
            return_details: 返回损失细节

        Returns:
            总损失或(损失, 细节)
        """
        base_loss = self.criterion(outputs, labels)
        penalty_loss = self.spec_penalty(outputs, labels)
        total = base_loss + adaptive_loss + penalty_loss
        if return_details:
            return total, {
                'base': float(base_loss.detach().item()),
                'adaptive': float((adaptive_loss.detach().item() if isinstance(adaptive_loss, torch.Tensor) else adaptive_loss)),
                'penalty': float(penalty_loss.detach().item()),
            }
        return total
