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
        
        # 交叉注意力机制 - 使用集中配置
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )
        
        # 自注意力 - 使用集中配置
        self.self_attention = nn.MultiheadAttention(
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
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
        min_required_length = 59
        
        # 填充短序列
        if enhancer_ids.size(1) < min_required_length:
            padding_size = min_required_length - enhancer_ids.size(1)
            enhancer_ids = torch.nn.functional.pad(enhancer_ids, (0, padding_size), value=0)
        if promoter_ids.size(1) < min_required_length:
            padding_size = min_required_length - promoter_ids.size(1)
            promoter_ids = torch.nn.functional.pad(promoter_ids, (0, padding_size), value=0)
        
        # DNA嵌入 - 使用6-mer overlapping tokenization
        enhancer_embedding = self.embedding_en(enhancer_ids)
        promoter_embedding = self.embedding_pr(promoter_ids)
        
        # CNN特征提取
        enhancers_output = self.enhancer_sequential(enhancer_embedding.permute(0, 2, 1))
        promoters_output = self.promoter_sequential(promoter_embedding.permute(0, 2, 1))
        
        # 转换为Transformer格式 (seq_len, batch, features)
        enhancers_output = enhancers_output.permute(2, 0, 1)
        promoters_output = promoters_output.permute(2, 0, 1)
        
        # 初始化adaptive_loss累积
        total_adaptive_loss = 0.0
        
        # Transformer编码 - 使用CBAT模块
        # CBAT模块内部会自动处理位置编码和自适应注意力
        for layer in self.enhancer_transformer_layers:
            enhancers_output, layer_adaptive_loss = layer(enhancers_output)
            total_adaptive_loss += layer_adaptive_loss

        for layer in self.promoter_transformer_layers:
            promoters_output, layer_adaptive_loss = layer(promoters_output)
            total_adaptive_loss += layer_adaptive_loss
        
        # 交叉注意力: 让增强子关注启动子
        enhancers_attended, _ = self.cross_attention(
            enhancers_output, promoters_output, promoters_output
        )
        
        # 自注意力增强
        enhancers_final, _ = self.self_attention(
            enhancers_attended, enhancers_attended, enhancers_attended
        )
        promoters_final, _ = self.self_attention(
            promoters_output, promoters_output, promoters_output
        )
        
        # 全局池化降维为固定大小 (seq_len, batch, 64) -> (batch, 64)
        enhancers_pooled = self.adaptive_pool(enhancers_final.permute(1, 2, 0)).squeeze(-1)
        promoters_pooled = self.adaptive_pool(promoters_final.permute(1, 2, 0)).squeeze(-1)
        
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
