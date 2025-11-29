"""
PRISM Model - 简化版BERT预训练模型
架构: CNN -> Self-Attention -> Cross-Attention -> MLM Head
"""

from torch import nn
import torch
import torch.nn.functional as F
from config import (
    LEARNING_RATE, EMBEDDING_DIM, CNN_KERNEL_SIZE, POOL_KERNEL_SIZE, OUT_CHANNELS,
    TRANSFORMER_HEADS, TRANSFORMER_FF_DIM, CNN_DROPOUT,
    WEIGHT_DECAY, DNA_EMBEDDING_VOCAB_SIZE, DNA_EMBEDDING_DIM,
    DNA_EMBEDDING_PADDING_IDX, DNA_EMBEDDING_INIT_STD,
    BERT_LEARNING_RATE, BERT_WARMUP_STEPS, BERT_MAX_GRAD_NORM,
    PRISM_USE_CROSS_ATTENTION, PRISM_POOLING_TYPE
)
from models.pleat.embedding import create_dna_embedding_layer
from models.pleat.RoPE import RoPEConfig
from models.layers.attn import RoPEAttention


class PRISMModel(nn.Module):
    """
    PRISM预训练模型
    
    架构流程:
    1. DNA Embedding (6-mer tokenization)
    2. CNN特征提取
    3. Self-Attention (RoPE)
    4. Cross-Attention (Enhancer query Promoter)
    5. MLM Head (预测masked tokens)
    """
    
    def __init__(self):
        super(PRISMModel, self).__init__()
        
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
        
        # CNN特征提取
        self.enhancer_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, 
                     kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        self.promoter_cnn = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, 
                     kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )
        
        # Self-Attention (使用RoPE)
        TRANSFORMER_DROPOUT = RoPEConfig.ROPE_DROPOUT
        self.enhancer_self_attn = RoPEAttention(
            d_model=OUT_CHANNELS, 
            num_heads=TRANSFORMER_HEADS, 
            dropout=TRANSFORMER_DROPOUT
        )
        self.promoter_self_attn = RoPEAttention(
            d_model=OUT_CHANNELS, 
            num_heads=TRANSFORMER_HEADS, 
            dropout=TRANSFORMER_DROPOUT
        )
        
        # Cross-Attention (Enhancer query Promoter)
        if PRISM_USE_CROSS_ATTENTION:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=OUT_CHANNELS, 
                num_heads=TRANSFORMER_HEADS, 
                batch_first=False
            )
        
        # MLM Head - 预测masked tokens
        self.mlm_head = nn.Sequential(
            nn.Linear(OUT_CHANNELS, TRANSFORMER_FF_DIM),
            nn.GELU(),
            nn.LayerNorm(TRANSFORMER_FF_DIM),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(TRANSFORMER_FF_DIM, DNA_EMBEDDING_VOCAB_SIZE)
        )
        
        # Layer Norms
        self.norm_en = nn.LayerNorm(OUT_CHANNELS)
        self.norm_pr = nn.LayerNorm(OUT_CHANNELS)
        
        # 损失函数 - 使用交叉熵
        self.criterion = nn.CrossEntropyLoss(ignore_index=DNA_EMBEDDING_PADDING_IDX)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=BERT_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    
    def forward(self, enhancer_ids, promoter_ids, enhancer_mask_positions=None):
        """
        前向传播
        
        Args:
            enhancer_ids: [B, L_en] Enhancer token IDs
            promoter_ids: [B, L_pr] Promoter token IDs
            enhancer_mask_positions: [B, L_en] bool tensor, True表示该位置被mask
            
        Returns:
            mlm_logits: [B, L_en', vocab_size] MLM预测logits (注意长度变化)
            enhancer_repr: [B, L_en', D] Enhancer表示
            promoter_repr: [B, L_pr', D] Promoter表示
            seq_length_mapping: 原始序列长度到CNN后序列长度的映射信息
        """
        # 保存原始序列长度
        original_en_length = enhancer_ids.size(1)
        
        # 填充短序列
        min_required_length = 59
        if enhancer_ids.size(1) < min_required_length:
            padding_size = min_required_length - enhancer_ids.size(1)
            enhancer_ids = F.pad(enhancer_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)
        if promoter_ids.size(1) < min_required_length:
            padding_size = min_required_length - promoter_ids.size(1)
            promoter_ids = F.pad(promoter_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)
        
        # DNA嵌入
        enhancer_emb = self.embedding_en(enhancer_ids)  # [B, L_en, D_emb]
        promoter_emb = self.embedding_pr(promoter_ids)  # [B, L_pr, D_emb]
        
        # CNN特征提取
        # [B, L, D] -> [B, D, L] -> CNN -> [B, C, L']
        enhancer_cnn_out = self.enhancer_cnn(enhancer_emb.transpose(1, 2))  # [B, C, L_en']
        promoter_cnn_out = self.promoter_cnn(promoter_emb.transpose(1, 2))  # [B, C, L_pr']
        
        # 记录CNN后的序列长度
        cnn_seq_length = enhancer_cnn_out.size(2)
        
        # [B, C, L'] -> [B, L', C]
        enhancer_cnn_out = enhancer_cnn_out.transpose(1, 2)
        promoter_cnn_out = promoter_cnn_out.transpose(1, 2)
        
        # Self-Attention (RoPE)
        enhancer_self_out = self.enhancer_self_attn(enhancer_cnn_out)  # [B, L_en', C]
        promoter_self_out = self.promoter_self_attn(promoter_cnn_out)  # [B, L_pr', C]
        
        # Layer Norm
        enhancer_self_out = self.norm_en(enhancer_self_out)
        promoter_self_out = self.norm_pr(promoter_self_out)
        
        # Cross-Attention (Enhancer query Promoter)
        if PRISM_USE_CROSS_ATTENTION:
            # MultiheadAttention需要 [L, B, C] 格式
            enhancer_cross_in = enhancer_self_out.transpose(0, 1)  # [L_en', B, C]
            promoter_cross_in = promoter_self_out.transpose(0, 1)  # [L_pr', B, C]
            
            enhancer_cross_out, _ = self.cross_attention(
                enhancer_cross_in,  # query
                promoter_cross_in,  # key
                promoter_cross_in   # value
            )  # [L_en', B, C]
            
            # 残差连接
            enhancer_cross_out = enhancer_cross_out.transpose(0, 1)  # [B, L_en', C]
            enhancer_final = enhancer_self_out + enhancer_cross_out
        else:
            enhancer_final = enhancer_self_out
        
        # MLM Head - 预测masked tokens
        mlm_logits = self.mlm_head(enhancer_final)  # [B, L_en', vocab_size]
        
        # 返回序列长度映射信息
        seq_length_mapping = {
            'original_length': original_en_length,
            'cnn_length': cnn_seq_length
        }
        
        return mlm_logits, enhancer_final, promoter_self_out, seq_length_mapping
    
    def compute_mlm_loss(self, mlm_logits, original_ids, mask_positions, seq_length_mapping=None):
        """
        计算MLM损失
        
        Args:
            mlm_logits: [B, L_cnn, vocab_size] 模型预测 (CNN后的序列长度)
            original_ids: [B, L_orig] 原始token IDs
            mask_positions: [B, L_orig] bool tensor, True表示该位置被mask
            seq_length_mapping: 序列长度映射信息
            
        Returns:
            loss: MLM损失
            accuracy: 预测准确率
        """
        B, L_cnn, V = mlm_logits.shape
        L_orig = original_ids.size(1)
        
        # 如果序列长度不匹配，需要对mask_positions进行下采样
        if L_cnn != L_orig:
            # 使用最近邻下采样将mask_positions从L_orig映射到L_cnn
            # 计算下采样比例
            downsample_ratio = L_orig / L_cnn
            
            # 创建新的mask_positions [B, L_cnn]
            mask_positions_downsampled = torch.zeros(B, L_cnn, dtype=torch.bool, device=mask_positions.device)
            original_ids_downsampled = torch.zeros(B, L_cnn, dtype=torch.long, device=original_ids.device)
            
            for i in range(L_cnn):
                # 计算对应的原始序列位置
                orig_idx = int(i * downsample_ratio)
                if orig_idx < L_orig:
                    mask_positions_downsampled[:, i] = mask_positions[:, orig_idx]
                    original_ids_downsampled[:, i] = original_ids[:, orig_idx]
            
            mask_positions = mask_positions_downsampled
            original_ids = original_ids_downsampled
        
        # 展平
        mlm_logits_flat = mlm_logits.view(-1, V)  # [B*L_cnn, V]
        original_ids_flat = original_ids.view(-1)  # [B*L_cnn]
        mask_positions_flat = mask_positions.view(-1)  # [B*L_cnn]
        
        # 选择masked位置
        masked_logits = mlm_logits_flat[mask_positions_flat]  # [N_masked, V]
        masked_labels = original_ids_flat[mask_positions_flat]  # [N_masked]
        
        # 计算损失
        if masked_logits.size(0) == 0:
            # 没有masked token
            return torch.tensor(0.0, device=mlm_logits.device, requires_grad=True), 0.0
        
        loss = F.cross_entropy(masked_logits, masked_labels)
        
        # 计算准确率
        with torch.no_grad():
            predictions = masked_logits.argmax(dim=-1)
            accuracy = (predictions == masked_labels).float().mean().item()
        
        return loss, accuracy


def create_mlm_mask(token_ids, mask_prob=0.15, mask_token_id=4096, 
                    vocab_size=4097, pad_token_id=0):
    """
    创建MLM mask
    
    Args:
        token_ids: [B, L] 原始token IDs
        mask_prob: mask概率
        mask_token_id: [MASK] token ID
        vocab_size: 词汇表大小 (实际有效索引是0到vocab_size-1)
        pad_token_id: padding token ID
        
    Returns:
        masked_ids: [B, L] masked后的token IDs
        mask_positions: [B, L] bool tensor, True表示该位置被mask
        original_ids: [B, L] 原始token IDs (用于计算损失)
    """
    B, L = token_ids.shape
    device = token_ids.device
    
    # 复制原始IDs
    masked_ids = token_ids.clone()
    original_ids = token_ids.clone()
    
    # 创建mask positions (不mask padding和超出vocab范围的token)
    is_valid = (token_ids != pad_token_id) & (token_ids < vocab_size) & (token_ids >= 0)
    
    # 随机选择mask_prob比例的token
    rand = torch.rand(B, L, device=device)
    mask_positions = (rand < mask_prob) & is_valid
    
    # 对于被选中的token:
    # 80%替换为[MASK], 10%替换为随机token, 10%保持不变
    mask_rand = torch.rand(B, L, device=device)
    
    # 80%: 替换为[MASK]
    replace_with_mask = mask_positions & (mask_rand < 0.8)
    masked_ids[replace_with_mask] = mask_token_id
    
    # 10%: 替换为随机token (确保不超出vocab范围)
    replace_with_random = mask_positions & (mask_rand >= 0.8) & (mask_rand < 0.9)
    # 生成1到vocab_size-1之间的随机token (避免0和mask_token_id)
    random_tokens = torch.randint(1, vocab_size - 1, (B, L), device=device)
    masked_ids[replace_with_random] = random_tokens[replace_with_random]
    
    # 10%: 保持不变 (不需要操作)
    
    return masked_ids, mask_positions, original_ids
