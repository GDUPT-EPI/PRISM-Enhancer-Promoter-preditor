from torch import nn
import torch
import numpy as np
import math
from config import EMBEDDING_MATRIX_PATH, LEARNING_RATE
from layers.skipnet import StochasticDepth

NUMBER_WORDS = 4097
EMBEDDING_DIM = 768
CNN_KERNEL_SIZE = 40
POOL_KERNEL_SIZE = 20
OUT_CHANNELS = 64
# LEARNING_RATE = 1e-3

# Load embedding matrix
embedding_matrix_full = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
embedding_matrix = embedding_matrix_full[:NUMBER_WORDS, :]


# 位置编码
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
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
        
        # Stochastic Depth 配置
        self.num_transformer_layers = 2  # 每个transformer的层数
        self.stochastic_depth_rate = 0.2  # 最大drop概率

        # Embedding layers
        self.embedding_en = nn.Embedding(4097, 768)
        self.embedding_pr = nn.Embedding(4097, 768)
        self.embedding_en.weight = nn.Parameter(embedding_matrix)
        self.embedding_pr.weight = nn.Parameter(embedding_matrix)
        self.embedding_en.requires_grad = True
        self.embedding_pr.requires_grad = True
        
        # CNN特征提取
        self.enhancer_sequential = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2)
        )
        self.promoter_sequential = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2)
        )
        
        # 固定的位置编码
        self.pos_encoder = FixedPositionalEncoding(d_model=64, max_len=1000)
        
        # # Transformer编码器
        # encoder_layer_1 = nn.TransformerEncoderLayer(
        #     d_model=64, nhead=8, dim_feedforward=256, 
        #     dropout=0.1, batch_first=False
        # )
        # self.enhancer_transformer = nn.TransformerEncoder(encoder_layer_1, num_layers=2)
        
        # encoder_layer_2 = nn.TransformerEncoderLayer(
        #     d_model=64, nhead=8, dim_feedforward=256,
        #     dropout=0.1, batch_first=False
        # )
        # self.promoter_transformer = nn.TransformerEncoder(encoder_layer_2, num_layers=2)
        # Transformer编码器 - 使用独立层以支持Stochastic Depth
        # 计算线性递增的drop概率
        total_layers = self.num_transformer_layers * 2  # enhancer + promoter
        drop_probs = [i / (total_layers - 1) * self.stochastic_depth_rate 
                    for i in range(total_layers)]

        # Enhancer Transformer layers
        self.enhancer_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64, nhead=8, dim_feedforward=256, 
                dropout=0.1, batch_first=False
            ) for _ in range(self.num_transformer_layers)
        ])
        self.enhancer_stochastic_depth = nn.ModuleList([
            StochasticDepth(drop_probs[i]) 
            for i in range(self.num_transformer_layers)
        ])

        # Promoter Transformer layers
        self.promoter_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64, nhead=8, dim_feedforward=256,
                dropout=0.1, batch_first=False
            ) for _ in range(self.num_transformer_layers)
        ])
        self.promoter_stochastic_depth = nn.ModuleList([
            StochasticDepth(drop_probs[self.num_transformer_layers + i]) 
            for i in range(self.num_transformer_layers)
        ])
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=False)
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=False)
        
        # Pooling层 - 将序列降维为固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        
        # 固定大小的分类头
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),  # 增强子+启动子各64维
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.33),
            nn.Linear(128, 1)
        )
        
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=LEARNING_RATE, weight_decay=0.001
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