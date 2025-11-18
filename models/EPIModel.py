from torch import nn
import torch
import numpy as np
import math
from config import (
    LEARNING_RATE, NUMBER_WORDS, EMBEDDING_DIM, CNN_KERNEL_SIZE, POOL_KERNEL_SIZE, OUT_CHANNELS,
    TRANSFORMER_LAYERS, TRANSFORMER_HEADS, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT,
    CNN_DROPOUT, CLASSIFIER_HIDDEN_SIZE, CLASSIFIER_DROPOUT,
    POS_ENCODING_MAX_LEN, WEIGHT_DECAY, DNA_EMBEDDING_VOCAB_SIZE, DNA_EMBEDDING_DIM,
    DNA_EMBEDDING_PADDING_IDX, DNA_EMBEDDING_INIT_STD
)
from models.pleat.embedding import create_dna_embedding_layer
from models.pleat.RoPE import RoPEAttention


class RoPETransformerEncoderLayer(nn.Module):
    """
    集成RoPE的Transformer编码器层
    替换标准的多头自注意力为RoPE注意力
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(RoPETransformerEncoderLayer, self).__init__()
        
        # 使用RoPE注意力替代标准多头注意力
        self.self_attn = RoPEAttention(
            d_model=d_model,
            num_heads=nhead,
            max_seq_len=POS_ENCODING_MAX_LEN,
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
            输出张量，形状 (seq_len, batch, d_model)
        """
        # 保存原始输入用于残差连接
        residual = x
        
        # 自注意力计算 (使用RoPE)
        # 转换维度以适配RoPEAttention: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        x_transposed = x.permute(1, 0, 2)
        attn_output = self.self_attn(x_transposed, attention_mask=src_mask)
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
        
        return x


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
        
        # Transformer编码器 - 使用RoPE位置编码 (不再需要单独的SinPE)

        # Enhancer RoPE Transformer layers
        self.enhancer_transformer_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS, 
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT
            ) for _ in range(self.num_transformer_layers)
        ])

        # Promoter RoPE Transformer layers
        self.promoter_transformer_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT
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
        
        # 固定大小的分类头 - 使用集中配置的参数
        self.fc = nn.Sequential(
            nn.Linear(OUT_CHANNELS * 2, CLASSIFIER_HIDDEN_SIZE),  # 增强子+启动子各OUT_CHANNELS维
            nn.BatchNorm1d(CLASSIFIER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=CLASSIFIER_DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN_SIZE, 1)
        )
        
        self.criterion = nn.BCELoss()
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
        
        # Transformer编码 - 使用RoPE位置编码 (不再需要外部位置编码)
        # RoPE注意力层内部会自动处理位置编码
        for layer in self.enhancer_transformer_layers:
            enhancers_output = layer(enhancers_output)

        for layer in self.promoter_transformer_layers:
            promoters_output = layer(promoters_output)
        
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
        
        # 分类
        result = self.fc(combined)
        
        return torch.sigmoid(result), combined