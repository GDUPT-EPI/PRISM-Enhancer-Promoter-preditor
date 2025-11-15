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
from models.pleat.SinPE import SinPE


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
        
        # 固定的位置编码
        self.pos_encoder = SinPE(d_model=OUT_CHANNELS)
        
        # Transformer编码器 - 使用集中配置参数

        # Enhancer Transformer layers
        self.enhancer_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS, dim_feedforward=TRANSFORMER_FF_DIM, 
                dropout=TRANSFORMER_DROPOUT, batch_first=False
            ) for _ in range(self.num_transformer_layers)
        ])

        # Promoter Transformer layers
        self.promoter_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS, dim_feedforward=TRANSFORMER_FF_DIM,
                dropout=TRANSFORMER_DROPOUT, batch_first=False
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
        
        # 添加固定位置编码
        enhancers_output = self.pos_encoder(enhancers_output)
        promoters_output = self.pos_encoder(promoters_output)
        
        # Transformer编码 - 简单残差连接
        for layer in self.enhancer_transformer_layers:
            residual = enhancers_output
            enhancers_output = layer(enhancers_output)
            enhancers_output = residual + enhancers_output

        for layer in self.promoter_transformer_layers:
            residual = promoters_output
            promoters_output = layer(promoters_output)
            promoters_output = residual + promoters_output
        
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