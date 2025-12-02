from torch import nn
import torch
import numpy as np
import math
from config import (
    LEARNING_RATE, EMBEDDING_DIM, CNN_KERNEL_SIZE, POOL_KERNEL_SIZE, OUT_CHANNELS,
    TRANSFORMER_LAYERS, TRANSFORMER_HEADS, TRANSFORMER_FF_DIM,
    CNN_DROPOUT, CLASSIFIER_HIDDEN_SIZE, CLASSIFIER_DROPOUT,
    WEIGHT_DECAY, DNA_EMBEDDING_VOCAB_SIZE, DNA_EMBEDDING_DIM,
    DNA_EMBEDDING_PADDING_IDX, DNA_EMBEDDING_INIT_STD
)
from models.pleat.embedding import create_dna_embedding_layer
from models.pleat.RoPE import RoPEConfig
from models.pleat.adaptive_immax import AdaptiveIMMAXLoss  
from models.layers.attn import *
from models.layers.FourierKAN import FourierKAN


class CBATTransformerEncoderLayer(nn.Module):
    """
    使用CBAT模块的Transformer编码器层
    替换标准的多头自注意力为CBAT注意力机制
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, img_size=32):
        super(CBATTransformerEncoderLayer, self).__init__()
        
        # 使用CBAT模块替换原有的注意力机制
        self.self_attn = CBAT(
            d_model=d_model,
            num_heads=nhead,
            img_size=img_size,
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,
            dropout=dropout
        )
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = nn.ReLU()
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
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
        residual = x
        
        # 自注意力计算 (使用CBAT)
        # 转换维度以适配CBAT: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        x_transposed = x.permute(1, 0, 2)
        
        # CBAT返回(output, adaptive_loss)
        attn_output, adaptive_loss = self.self_attn(x_transposed, attention_mask=src_mask, return_loss=True)
            
        # 转回原始维度: (batch, seq_len, d_model) -> (seq_len, batch, d_model)
        attn_output = attn_output.permute(1, 0, 2)
        
        # 残差连接和层归一化
        x = residual + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 保存用于第二个残差连接
        residual = x
        
        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # 残差连接和层归一化
        x = residual + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, adaptive_loss


# Backbone
class EPIModel(nn.Module):
    def __init__(self):
        super(EPIModel, self).__init__()
        
        # Transformer配置
        self.num_transformer_layers = TRANSFORMER_LAYERS

        # DNA嵌入层 - 使用6-mer overlapping tokenization
        self.embedding_en = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )
        self.embedding_pr = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )
        
        # CNN特征提取 - 使用集中配置的dropout
        self.enhancer_sequential = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        self.promoter_sequential = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        
        # Transformer编码器 - 使用CBAT模块
        TRANSFORMER_DROPOUT = RoPEConfig.ROPE_DROPOUT  # 获取dropout
        
        # 计算合适的img_size，使序列长度可以reshape为正方形
        # 根据实际序列长度动态计算img_size
        # 这里使用一个保守的估计，确保序列长度可以reshape为img_size*img_size
        # 假设经过CNN后的序列长度为seq_len，我们需要找到最接近的平方数
        # 这里使用一个合理的默认值，实际使用时可能需要根据序列长度调整
        # 为了避免错误，我们将img_size设置为较小的值，并在CBAT中处理不匹配的情况
        img_size = 16  # 16*16=256，可以容纳大多数序列长度
        
        # Enhancer CBAT Transformer layers
        self.enhancer_transformer_layers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS, 
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,
                img_size=img_size
            ) for _ in range(self.num_transformer_layers)
        ])

        # Promoter CBAT Transformer layers
        self.promoter_transformer_layers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,
                img_size=img_size
            ) for _ in range(self.num_transformer_layers)
        ])
        
        # 早期自注意力机制 (Pre-CBAT) - 使用RoPE
        self.pre_enhancer_self_attn = RoPEAttention(
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )
        self.pre_promoter_self_attn = RoPEAttention(
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )
        
        # ============================================================================
        # LCWnet Footprint模块 (待集成)
        # ============================================================================
        # from models.layers.footprint import LCWnetFootprint
        # 
        # # 为Enhancer分支添加LCWnet时频特征提取
        # self.enhancer_footprint = LCWnetFootprint(
        #     d_model=OUT_CHANNELS,
        #     scales=np.logspace(0, 2, 32).tolist(),  # 32个对数均匀分布的尺度
        #     k_max=0.5,                               # 频率支撑上限
        #     hidden_dim=64,                           # 母小波MLP隐藏维度
        #     num_layers=3,                            # 母小波MLP层数
        #     fusion_type='attention',                 # 多尺度融合方式
        # )
        # 
        # 集成位置说明:
        # - 输入1: self.pre_enhancer_self_attn的输出 (早期自注意力后的Enhancer特征)
        # - 输入2: cross_attention_1的输出 (交叉注意力后的Enhancer特征)
        # - 通过门控机制融合LCWnet时频特征与原始特征
        # - 输出传递给后续的CBAT Transformer layers
        # 
        # 前向传播集成示例 (在forward方法中):
        # ```python
        # # 早期自注意力后，保存用于LCWnet
        # enhancers_pre_attn = self.pre_enhancer_self_attn(enhancers_output)
        # 
        # # ... CBAT Transformer layers ...
        # 
        # # 交叉注意力后
        # enhancers_attended_1, _ = self.cross_attention_1(...)
        # 
        # # LCWnet时频特征提取 (使用早期特征作为输入，交叉注意力特征作为残差)
        # enhancers_footprint = self.enhancer_footprint(
        #     x=enhancers_pre_attn,           # 输入: 早期自注意力特征
        #     residual=enhancers_attended_1   # 残差: 交叉注意力特征
        # )
        # 
        # # 门控融合已在LCWnetFootprint内部完成
        # # 将footprint输出传递给后续模块
        # enhancers_output = enhancers_footprint
        # ```
        # ============================================================================
        
        # 交叉注意力机制 1 (CBAT后) - 使用集中配置
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )
        
        # 交叉注意力机制 2 (原有的) - 使用集中配置
        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )
        
        # 晚期自注意力增强 (Post-CrossAttn) - 使用CBAT模块
        # 这里的img_size可能需要根据合并后的序列长度调整，或者保持一致
        self.post_enhancer_cbat = CBAT(
            d_model=OUT_CHANNELS,
            num_heads=TRANSFORMER_HEADS,
            img_size=img_size, # 假设维度保持一致
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,
            dropout=TRANSFORMER_DROPOUT
        )
        self.post_promoter_cbat = CBAT(
            d_model=OUT_CHANNELS,
            num_heads=TRANSFORMER_HEADS,
            img_size=img_size,
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,
            dropout=TRANSFORMER_DROPOUT
        )
        
        # Pooling层 - 将序列降维为固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        
        # 使用FourierKAN替代末端MLP作为分类器
        self.classifier = FourierKAN(
            in_features=OUT_CHANNELS * 2,
            out_features=1,
            grid_size=5,
            width=2 * (OUT_CHANNELS * 2) + 1,
        )
        
        # 损失函数
        self.criterion = AdaptiveIMMAXLoss(
        alpha_init=0.5,           # 初始α=0.5，训练中会自动调整
        alpha_momentum=0.9,       # 高动量保证α稳定更新
        eps=1e-8,                 # 数值稳定性
        margin_clip=10.0          # 防止极端margin值导致数值溢出
        )

        # 优化器使用集中配置的weight_decay
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    
    def forward(self, enhancer_ids, promoter_ids, enhancer_features, promoter_features):
        K = CNN_KERNEL_SIZE
        P = POOL_KERNEL_SIZE
        min_required_length = K + P - 1
        
        if enhancer_ids.size(1) < min_required_length:
            padding_size = min_required_length - enhancer_ids.size(1)
            enhancer_ids = torch.nn.functional.pad(enhancer_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)
        if promoter_ids.size(1) < min_required_length:
            padding_size = min_required_length - promoter_ids.size(1)
            promoter_ids = torch.nn.functional.pad(promoter_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)
        
        # DNA嵌入 - 使用6-mer overlapping tokenization
        enhancer_embedding = self.embedding_en(enhancer_ids)
        promoter_embedding = self.embedding_pr(promoter_ids)
        
        # CNN特征提取
        enhancers_output = self.enhancer_sequential(enhancer_embedding.permute(0, 2, 1))
        promoters_output = self.promoter_sequential(promoter_embedding.permute(0, 2, 1))
        
        # 转换为Transformer格式 (seq_len, batch, features)
        enhancers_output = enhancers_output.permute(2, 0, 1)
        promoters_output = promoters_output.permute(2, 0, 1)
        
        # 构造padding掩码（映射到CNN/Pool后的长度）
        B = enhancer_ids.size(0)
        L_en_orig = enhancer_ids.size(1)
        L_pr_orig = promoter_ids.size(1)
        L_en_cnn = enhancers_output.size(0)
        L_pr_cnn = promoters_output.size(0)
        pad_en = (enhancer_ids == DNA_EMBEDDING_PADDING_IDX)
        pad_pr = (promoter_ids == DNA_EMBEDDING_PADDING_IDX)
        enh_pad_mask = torch.zeros(B, L_en_cnn, dtype=torch.bool, device=enhancer_ids.device)
        pr_pad_mask = torch.zeros(B, L_pr_cnn, dtype=torch.bool, device=promoter_ids.device)
        for j in range(L_en_cnn):
            s = j * P
            e = min(s + P + K - 2, L_en_orig - 1)
            enh_pad_mask[:, j] = pad_en[:, s:e+1].all(dim=-1)
        for j in range(L_pr_cnn):
            s = j * P
            e = min(s + P + K - 2, L_pr_orig - 1)
            pr_pad_mask[:, j] = pad_pr[:, s:e+1].all(dim=-1)

        # 早期自注意力 (Pre-CBAT) - 使用RoPEAttention并传入padding掩码
        # RoPEAttention需要 (batch, seq_len, d_model) 格式
        enhancers_output = enhancers_output.permute(1, 0, 2)
        promoters_output = promoters_output.permute(1, 0, 2)
        enh_attn_mask = torch.zeros(B, 1, L_en_cnn, L_en_cnn, device=enhancer_ids.device, dtype=torch.float32)
        pr_attn_mask = torch.zeros(B, 1, L_pr_cnn, L_pr_cnn, device=promoter_ids.device, dtype=torch.float32)
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
        enhancers_output = self.pre_enhancer_self_attn(enhancers_output, attention_mask=enh_attn_mask)
        promoters_output = self.pre_promoter_self_attn(promoters_output, attention_mask=pr_attn_mask)
        
        # ============================================================================
        # LCWnet Footprint集成点 1: 保存早期自注意力输出
        # ============================================================================
        # enhancers_pre_attn_for_footprint = enhancers_output.clone()  # [B, L, D]
        # 这个特征将作为LCWnet的主输入，用于提取时频特征
        # ============================================================================
        
        # 转回 (seq_len, batch, features) 供后续模块使用
        enhancers_output = enhancers_output.permute(1, 0, 2)
        promoters_output = promoters_output.permute(1, 0, 2)
        
        # 初始化adaptive_loss累积
        total_adaptive_loss = 0.0
        
        # Transformer编码 - 使用CBAT模块
        # CBAT模块内部会自动处理位置编码和自适应注意力
        for layer in self.enhancer_transformer_layers:
            enhancers_output, layer_adaptive_loss = layer(enhancers_output, src_mask=enh_attn_mask)
            total_adaptive_loss += layer_adaptive_loss

        for layer in self.promoter_transformer_layers:
            promoters_output, layer_adaptive_loss = layer(promoters_output, src_mask=pr_attn_mask)
            total_adaptive_loss += layer_adaptive_loss
        
        # 交叉注意力 1: 增强子查询启动子 (新增)
        # 注意: MultiheadAttention 默认 batch_first=False，即 (seq_len, batch, embed_dim)
        # query=enhancers, key=promoters, value=promoters
        enhancers_attended_1, _ = self.cross_attention_1(
            enhancers_output, promoters_output, promoters_output, key_padding_mask=pr_pad_mask
        )
        
        # 残差连接 1
        enhancers_output = enhancers_output + enhancers_attended_1
        
        # ============================================================================
        # LCWnet Footprint集成点 2: 交叉注意力后，应用时频特征提取
        # ============================================================================
        # # 转换为 (batch, seq_len, d_model) 格式
        # enhancers_cross_attn = enhancers_output.permute(1, 0, 2)  # [B, L, D]
        # 
        # # 应用LCWnet时频特征提取
        # # 输入: 早期自注意力特征 (用于CWT分析)
        # # 残差: 交叉注意力特征 (用于门控融合)
        # enhancers_with_footprint = self.enhancer_footprint(
        #     x=enhancers_pre_attn_for_footprint,  # 早期特征作为CWT输入
        #     residual=enhancers_cross_attn         # 交叉注意力特征作为残差
        # )  # [B, L, D]
        # 
        # # 转回 (seq_len, batch, d_model) 格式
        # enhancers_output = enhancers_with_footprint.permute(1, 0, 2)
        # 
        # # 此时enhancers_output已经融合了:
        # # 1. 早期自注意力的局部模式
        # # 2. 交叉注意力的E-P交互信息
        # # 3. LCWnet提取的时频域特征 (DNA序列的周期性、频率特性)
        # ============================================================================
        
        # 交叉注意力 2: 增强子查询启动子 (原有)
        enhancers_attended_2, _ = self.cross_attention_2(
            enhancers_output, promoters_output, promoters_output, key_padding_mask=pr_pad_mask
        )
        
        # 残差连接 2 (原有代码没有显式残差，这里按照需求添加与新的交叉连接残差连接)
        # 这里的需求理解为：将经过CBAT以后的交叉注意力(L233-236，即现在的cross_attention_1结果) 与 新的交叉连接(cross_attention_2) 残差连接
        # cross_attention_1的结果已经加到了enhancers_output中
        enhancers_final = enhancers_output + enhancers_attended_2
        
        # 晚期自注意力增强 -> 替换为CBAT模块
        # CBAT需要 (batch, seq_len, d_model)
        enhancers_final_transposed = enhancers_final.permute(1, 0, 2)
        promoters_output_transposed = promoters_output.permute(1, 0, 2)
        
        enhancers_final_cbat, loss_e = self.post_enhancer_cbat(enhancers_final_transposed, return_loss=True)
        promoters_final_cbat, loss_p = self.post_promoter_cbat(promoters_output_transposed, return_loss=True)
        
        total_adaptive_loss += loss_e
        total_adaptive_loss += loss_p
        
        # 转回 (seq_len, batch, d_model)
        enhancers_final = enhancers_final_cbat.permute(1, 0, 2)
        promoters_final = promoters_final_cbat.permute(1, 0, 2)
        
        # 全局池化降维为固定大小 (seq_len, batch, 64) -> (batch, 64)
        x_en = enhancers_final.permute(1, 0, 2)
        w_en = (~enh_pad_mask).float()
        s_en = (x_en * w_en.unsqueeze(-1)).sum(dim=1)
        d_en = w_en.sum(dim=1).clamp(min=1e-6)
        enhancers_pooled = s_en / d_en.unsqueeze(-1)
        x_pr = promoters_final.permute(1, 0, 2)
        w_pr = (~pr_pad_mask).float()
        s_pr = (x_pr * w_pr.unsqueeze(-1)).sum(dim=1)
        d_pr = w_pr.sum(dim=1).clamp(min=1e-6)
        promoters_pooled = s_pr / d_pr.unsqueeze(-1)
        
        # 拼接增强子和启动子表示
        combined = torch.cat([enhancers_pooled, promoters_pooled], dim=1)  # (batch, 128)
        
        # 分类（FourierKAN）
        result = self.classifier(combined)
        
        # CBAT模块总是返回adaptive_loss
        return torch.sigmoid(result), combined, total_adaptive_loss
    
    def get_loss_alpha(self):
        """获取当前IMMAX损失的α值，用于监控训练"""
        return self.criterion.get_alpha()
    
    def compute_loss(self, outputs, labels, adaptive_loss=0.0, return_details=False):
        """
        统一的损失计算接口
        
        Args:
            outputs: 模型输出（已sigmoid），形状 (batch, 1) 或 (batch,)
            labels: 真实标签，形状 (batch,)
            adaptive_loss: CBAT模块返回的自适应损失
            return_details: 是否返回详细信息
            
        Returns:
            loss: 总损失
            如果return_details=True，返回字典包含各项损失和α值
        """
        # 计算IMMAX损失
        immax_loss, alpha = self.criterion(outputs, labels, return_alpha=True)
        
        # 总损失
        total_loss = immax_loss + adaptive_loss
        
        if return_details:
            return total_loss, {
                'total_loss': total_loss.item(),
                'immax_loss': immax_loss.item(),
                'adaptive_loss': adaptive_loss if isinstance(adaptive_loss, float) else adaptive_loss.item(),
                'alpha': alpha
            }
        
        return total_loss
