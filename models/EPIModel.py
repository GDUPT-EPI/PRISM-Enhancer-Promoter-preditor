from torch import nn
import torch
import numpy as np
import math
from config import (
    LEARNING_RATE, NUMBER_WORDS, EMBEDDING_DIM, CNN_KERNEL_SIZE, POOL_KERNEL_SIZE, OUT_CHANNELS,
    TRANSFORMER_LAYERS, TRANSFORMER_HEADS, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT,
    STOCHASTIC_DEPTH_RATE, CNN_DROPOUT, CLASSIFIER_HIDDEN_SIZE, CLASSIFIER_DROPOUT,
    POS_ENCODING_MAX_LEN, WEIGHT_DECAY
)
from models.pleat.skipnet import StochasticDepth


# 位置编码
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=POS_ENCODING_MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :, :]

# Backbone
class EPIModel(nn.Module):
    def __init__(self):
        super(EPIModel, self).__init__()
        
        # Stochastic Depth 配置 - 使用集中配置
        self.num_transformer_layers = TRANSFORMER_LAYERS
        self.stochastic_depth_rate = STOCHASTIC_DEPTH_RATE

        # Embedding layers - 随机初始化，不使用预训练权重
        self.embedding_en = nn.Embedding(NUMBER_WORDS, EMBEDDING_DIM)
        self.embedding_pr = nn.Embedding(NUMBER_WORDS, EMBEDDING_DIM)
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.embedding_en.weight)
        nn.init.xavier_uniform_(self.embedding_pr.weight)
        self.embedding_en.requires_grad = True
        self.embedding_pr.requires_grad = True
        
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
        self.pos_encoder = FixedPositionalEncoding(d_model=OUT_CHANNELS)
        
        # Transformer编码器 - 使用集中配置参数
        # 计算线性递增的drop概率
        total_layers = self.num_transformer_layers * 2  # enhancer + promoter
        drop_probs = [i / (total_layers - 1) * self.stochastic_depth_rate 
                    for i in range(total_layers)]

        # Enhancer Transformer layers
        self.enhancer_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS, dim_feedforward=TRANSFORMER_FF_DIM, 
                dropout=TRANSFORMER_DROPOUT, batch_first=False
            ) for _ in range(self.num_transformer_layers)
        ])
        self.enhancer_stochastic_depth = nn.ModuleList([
            StochasticDepth(drop_probs[i]) 
            for i in range(self.num_transformer_layers)
        ])

        # Promoter Transformer layers
        self.promoter_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS, dim_feedforward=TRANSFORMER_FF_DIM,
                dropout=TRANSFORMER_DROPOUT, batch_first=False
            ) for _ in range(self.num_transformer_layers)
        ])
        self.promoter_stochastic_depth = nn.ModuleList([
            StochasticDepth(drop_probs[self.num_transformer_layers + i]) 
            for i in range(self.num_transformer_layers)
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
        
        # Embedding
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
        
        # Transformer编码
        # enhancers_output = self.enhancer_transformer(enhancers_output)
        # promoters_output = self.promoter_transformer(promoters_output)
        # Transformer编码 - 带Stochastic Depth
        for layer, sd in zip(self.enhancer_transformer_layers, self.enhancer_stochastic_depth):
            residual = enhancers_output
            enhancers_output = layer(enhancers_output)
            enhancers_output = residual + sd(enhancers_output - residual)

        for layer, sd in zip(self.promoter_transformer_layers, self.promoter_stochastic_depth):
            residual = promoters_output
            promoters_output = layer(promoters_output)
            promoters_output = residual + sd(promoters_output - residual)
        
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