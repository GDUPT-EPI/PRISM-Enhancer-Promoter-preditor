"""
PRISM Model - 简化版BERT预训练模型
架构: CNN -> Self-Attention -> Cross-Attention -> MLM Head
"""

from torch import nn
import torch
import torch.nn.functional as F
from config import *
from models.pleat.embedding import create_dna_embedding_layer
from models.pleat.RoPE import RoPEConfig
from models.layers.footprint import FootprintExpert, FootprintConfig
from models.layers.attn import *
from models.layers.FourierKAN import FourierKAN
from models.EPIModel import CBATTransformerEncoderLayer
from config import *


class PRISMModel(nn.Module):
    """
    PRISM预训练模型 - BERT风格的DNA序列预训练
    
    架构流程 (与EPIModel对齐):
    1. DNA Embedding (6-mer tokenization) - 6-mer重叠DNA序列标记化
    2. CNN特征提取 - 提取局部序列模式和motif
    3. Pre-CBAT Self-Attention (RoPE) - 位置编码增强的自注意力
    4. CBAT Transformer Encoder Layers - 基于卷积的注意力Transformer层
    5. Cross-Attention (Enhancer query Promoter) - 增强子查询启动子
    6. MLM Head (预测masked tokens) - 掩码语言模型预测头
    """
    
    def __init__(self, num_cell_lines=0):
        super(PRISMModel, self).__init__()
        
        self.num_transformer_layers = TRANSFORMER_LAYERS
        assert OUT_CHANNELS % TRANSFORMER_HEADS == 0
        
        # 细胞系分类头（如果提供了num_cell_lines）
        self.num_cell_lines = num_cell_lines
        if self.num_cell_lines > 0:
            self.cell_classifier = nn.Sequential(
                nn.Linear(OUT_CHANNELS, OUT_CHANNELS // 2),
                nn.ReLU(),
                nn.Dropout(CNN_DROPOUT),
                nn.Linear(OUT_CHANNELS // 2, num_cell_lines)
            )


        # DNA嵌入层 - 使用6-mer overlapping tokenization
        self.embedding_en = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )  # 增强子DNA序列嵌入矩阵W_en ∈ R^(vocab×d_emb)
        self.embedding_pr = create_dna_embedding_layer(
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            embed_dim=DNA_EMBEDDING_DIM,
            padding_idx=DNA_EMBEDDING_PADDING_IDX,
            init_std=DNA_EMBEDDING_INIT_STD
        )  # 启动子DNA序列嵌入矩阵W_pr ∈ R^(vocab×d_emb)
        
        # CNN特征提取
        self.enhancer_sequential = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )  # 增强子CNN特征提取器 - 局部motif检测
        self.promoter_sequential = nn.Sequential(
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.BatchNorm1d(OUT_CHANNELS),
            nn.Dropout(p=CNN_DROPOUT)
        )  # 启动子CNN特征提取器 - 局部motif检测
        
        # Transformer参数
        TRANSFORMER_DROPOUT = RoPEConfig.ROPE_DROPOUT
        img_size = 16  # 与EPIModel保持一致
        
        # Enhancer CBAT Transformer layers
        self.enhancer_transformer_layers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS, 
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,
                img_size=img_size
            ) for _ in range(self.num_transformer_layers)
        ])  # 增强子CBAT Transformer层堆叠 - 基于卷积的注意力机制
        # Promoter CBAT Transformer layers
        self.promoter_transformer_layers = nn.ModuleList([
            CBATTransformerEncoderLayer(
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,
                img_size=img_size
            ) for _ in range(self.num_transformer_layers)
        ])  # 启动子CBAT Transformer层堆叠 - 基于卷积的注意力机制
        
        # 早期自注意力机制 (Pre-CBAT) - 使用RoPE
        self.pre_enhancer_self_attn = RoPEAttention(
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )  # 增强子RoPE位置编码自注意力 - 旋转位置嵌入
        self.pre_promoter_self_attn = RoPEAttention(
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )  # 启动子RoPE位置编码自注意力 - 旋转位置嵌入
        
        # Footprint模块 (LCWnet)
        # 用于在Self-Attention后提取时频特征
        # 使用 FootprintExpert 替代原始 LCWnetFootprint，包含投影头
        self.enhancer_footprint = FootprintExpert(
            d_model=OUT_CHANNELS,
        )  # 增强子专家模块
        self.promoter_footprint = FootprintExpert(
            d_model=OUT_CHANNELS,
        )  # 启动子专家模块
        
        # Footprint门控融合 (Self-Attn Footprint + Cross-Attn Footprint)
        # 输入维度是 2 * d_model (两个footprint向量拼接)
        # 输出维度是 d_model (融合后的向量)
        self.footprint_fusion_gate = nn.Sequential(
            nn.Linear(OUT_CHANNELS * 2, OUT_CHANNELS),
            nn.Sigmoid()
        )  # Footprint门控网络 - 自注意力与交叉注意力特征的加权融合
        self.footprint_inject_proj = nn.Sequential(
            nn.LayerNorm(OUT_CHANNELS),
            nn.Linear(OUT_CHANNELS, OUT_CHANNELS),
            nn.GELU()
        )  # Footprint注入投影器 - 融合特征注入Transformer的预处理
        self.footprint_alpha = nn.Parameter(torch.tensor(0.5))  # Footprint注入权重α - 控制融合特征的影响强度
        
        # 融合后的Footprint投影到序列长度 (用于注入Transformer)
        # 将 [B, D] 投影并重复到 [B, L, D]
        # 注意：这里我们实际上不需要投影到序列长度，而是将其作为全局上下文添加到每个token
        # 或者将其加到Transformer的输入中
        
        # Cross-Attention (Enhancer query Promoter)
        # EPIModel中使用的是 cross_attention_1
        if PRISM_USE_CROSS_ATTENTION:
            self.cross_attention_1 = nn.MultiheadAttention(
                embed_dim=OUT_CHANNELS, 
                num_heads=TRANSFORMER_HEADS, 
                batch_first=False
            )  # 交叉注意力模块 - 增强子查询启动子进行特征交互
        
        # MLM Head - 预测masked tokens
        # 注意：EPIModel在cross_attention_1之后还有残差连接和后续层，PRISM截断到这里
        # 输入是 enhancer_final (cross attention output + residual)
        self.mlm_head = nn.Sequential(
            nn.Linear(OUT_CHANNELS, TRANSFORMER_FF_DIM),
            nn.GELU(),
            nn.LayerNorm(TRANSFORMER_FF_DIM),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(TRANSFORMER_FF_DIM, DNA_EMBEDDING_VOCAB_SIZE)
        )  # MLM预测头 - 掩码语言模型预测网络
        
        # Layer Norms (保留原有的，虽然EPIModel主要在TransformerLayer内部做norm)
        # 为了对齐MLM head前的状态，我们可能需要一个LayerNorm
        self.norm_final = nn.LayerNorm(OUT_CHANNELS)  # 最终层归一化 - 稳定训练过程
        
        # 损失函数 - 使用交叉熵
        self.criterion = nn.CrossEntropyLoss(ignore_index=DNA_EMBEDDING_PADDING_IDX)  # 交叉熵损失 - MLM任务目标函数
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=BERT_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )  # AdamW优化器 - 自适应学习率权重衰减优化器
    
    def forward(
        self,
        enhancer_ids: torch.Tensor,
        promoter_ids: torch.Tensor,
        enhancer_mask_positions: torch.Tensor,
    ):
        # 保存原始序列长度
        original_en_length = enhancer_ids.size(1)  # 原始增强子序列长度L_orig
        
        min_required_length = 59  # 最小序列长度要求 - 满足CNN和Transformer计算需求
        
        # 填充短序列
        if enhancer_ids.size(1) < min_required_length:
            padding_size = min_required_length - enhancer_ids.size(1)
            enhancer_ids = F.pad(enhancer_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)  # 增强子序列填充至最小长度
        if promoter_ids.size(1) < min_required_length:
            padding_size = min_required_length - promoter_ids.size(1)
            promoter_ids = F.pad(promoter_ids, (0, padding_size), value=DNA_EMBEDDING_PADDING_IDX)  # 启动子序列填充至最小长度
        
        # DNA嵌入
        enhancer_embedding = self.embedding_en(enhancer_ids)  # E_en = Embed(en_ids) - 增强子序列嵌入表示
        promoter_embedding = self.embedding_pr(promoter_ids)  # E_pr = Embed(pr_ids) - 启动子序列嵌入表示
        
        # CNN特征提取
        enhancers_output = self.enhancer_sequential(enhancer_embedding.permute(0, 2, 1))  # H_en = CNN(E_en) - 增强子CNN特征提取
        promoters_output = self.promoter_sequential(promoter_embedding.permute(0, 2, 1))  # H_pr = CNN(E_pr) - 启动子CNN特征提取
        
        # 记录CNN后的序列长度
        cnn_seq_length = enhancers_output.size(2)  # CNN处理后的增强子序列长度L_cnn
        cnn_seq_length_pr = promoters_output.size(2)  # CNN处理后的启动子序列长度L_cnn_pr
        
        # 转换为Transformer格式 (seq_len, batch, features)
        enhancers_output = enhancers_output.permute(2, 0, 1)  # [L_cnn, B, D] - Transformer输入格式
        promoters_output = promoters_output.permute(2, 0, 1)  # [L_cnn_pr, B, D] - Transformer输入格式
        
        # 早期自注意力 (Pre-CBAT) - 使用RoPEAttention
        # RoPEAttention需要 (batch, seq_len, d_model) 格式
        enhancers_output = enhancers_output.permute(1, 0, 2)  # [B, L_cnn, D] - RoPE注意力输入格式
        promoters_output = promoters_output.permute(1, 0, 2)  # [B, L_cnn_pr, D] - RoPE注意力输入格式
        
        enhancers_output = self.pre_enhancer_self_attn(enhancers_output)  # A_en = RoPEAttn(H_en) - 增强子RoPE自注意力
        promoters_output = self.pre_promoter_self_attn(promoters_output)  # A_pr = RoPEAttn(H_pr) - 启动子RoPE自注意力
        
        # 转回 (seq_len, batch, features) 供CBAT模块使用
        enhancers_output = enhancers_output.permute(1, 0, 2)  # [L_cnn, B, D] - CBAT输入格式
        promoters_output = promoters_output.permute(1, 0, 2)  # [L_cnn_pr, B, D] - CBAT输入格式
        
        # -------------------------------------------------------------------
        # Footprint Integration Point 1: CNN -> Self-Attn 后
        # -------------------------------------------------------------------
        # 输入: [B, L, D] (batch_first=True for LCWnetFootprint)
        # enhancers_output目前是 [L, B, D]，需要转置
        
        enhancers_for_footprint = enhancers_output.permute(1, 0, 2) # [B, L, D] - 增强子特征重排为Footprint输入格式
        promoters_for_footprint = promoters_output.permute(1, 0, 2) # [B, L, D] - 启动子特征重排为Footprint输入格式
        
        # 提取Enhancer Footprint
        # expert返回: seq_out, sample_vec, z_com, z_spec
        _, enhancer_footprint_vec_1, z_com_en, z_spec_en = self.enhancer_footprint(enhancers_for_footprint) 
        
        # 提取Promoter Footprint (虽然目前只用enhancer做对比学习，但为了完整性)
        _, promoter_footprint_vec_1, _, _ = self.promoter_footprint(promoters_for_footprint) 
        
        fused_footprint = enhancer_footprint_vec_1  # 初始融合footprint
        
        # 返回投影特征供Loss计算
        self.current_z_com = z_com_en
        self.current_z_spec = z_spec_en
        
        # Transformer编码 - 使用CBAT模块
        for layer in self.enhancer_transformer_layers:
            enhancers_output, _ = layer(enhancers_output)  # CBAT增强子Transformer层堆叠处理

        for layer in self.promoter_transformer_layers:
            promoters_output, _ = layer(promoters_output)  # CBAT启动子Transformer层堆叠处理
            
        # 保存promoter output用于返回
        promoter_final = promoters_output.permute(1, 0, 2) # [B, L, D] - 最终启动子特征输出
        
        # Cross-Attention 1: 增强子查询启动子
        if PRISM_USE_CROSS_ATTENTION:
            # MultiheadAttention 默认 batch_first=False (seq_len, batch, embed_dim)
            promoter_pad_orig = (promoter_ids == DNA_EMBEDDING_PADDING_IDX)  # 启动子原始填充位置掩码
            K = CNN_KERNEL_SIZE  # CNN卷积核大小
            P = POOL_KERNEL_SIZE  # 池化核大小
            promoter_key_padding_mask = torch.zeros(promoter_ids.size(0), cnn_seq_length_pr, dtype=torch.bool, device=promoter_ids.device)  # 交叉注意力键填充掩码
            for j in range(cnn_seq_length_pr):
                start = j * P
                end = min(start + P + K - 2, promoter_ids.size(1) - 1)
                region_all_pad = promoter_pad_orig[:, start:end+1].all(dim=-1)
                promoter_key_padding_mask[:, j] = region_all_pad  # CNN池化区域全填充的掩码
            enhancers_attended_1, _ = self.cross_attention_1(
                enhancers_output, promoters_output, promoters_output,
                key_padding_mask=promoter_key_padding_mask
            )  # A_en^CA = CrossAttn(A_en, A_pr, A_pr) - 增强子查询启动子交叉注意力
            
            # 残差连接 1
            enhancers_output = enhancers_output + enhancers_attended_1  # A_en^res = A_en + A_en^CA - 交叉注意力残差连接
            
            # -------------------------------------------------------------------
            # Footprint Integration Point 2: Cross-Attn 1 后
            # -------------------------------------------------------------------
            # 再次提取Footprint，这次基于Cross-Attn后的特征
            # 输入: [B, L, D]
            enhancers_after_cross = enhancers_output.permute(1, 0, 2) # [B, L, D] - 交叉注意力后增强子特征重排
            
            # 使用同一个Enhancer Footprint模块或者共享权重的模块
            # 这里我们复用 self.enhancer_footprint 提取特征
            _, enhancer_footprint_vec_2, _, _ = self.enhancer_footprint(enhancers_after_cross) 
            
            # -------------------------------------------------------------------
            # Footprint Fusion: Gate Control
            # -------------------------------------------------------------------
            # 两个footprint使用门控拼接
            # f_fused = Gate([f1, f2]) * f1 + (1 - Gate([f1, f2])) * f2  <-- 或者是简单的拼接后投影
            # 根据代码实现：gate_input = cat([f1, f2]), gate = sigmoid(linear(gate_input))
            # output = gate * f1 + (1-gate) * f2 (假设维度一致)
            # 或者更通用的：output = Linear(cat(f1, f2)) * gate
            
            # 这里我们采用简单的加权融合
            footprint_concat = torch.cat([enhancer_footprint_vec_1, enhancer_footprint_vec_2], dim=-1) # [B, 2D] - 拼接两个阶段footprint特征
            gate_weight = self.footprint_fusion_gate(footprint_concat) # [B, D] (Sigmoid output) - 门控权重g = σ(MLP([f1;f2]))
            
            # 融合后的Footprint向量
            fused_footprint = gate_weight * enhancer_footprint_vec_1 + (1 - gate_weight) * enhancer_footprint_vec_2 # [B, D] - f_fused = g*f1 + (1-g)*f2
            
            # -------------------------------------------------------------------
            # Footprint Injection 1: Feed to first CBAT after cross attn (Simulated)
            # -------------------------------------------------------------------
            # 这里的 "first CBAT after cross attn" 在PRISMModel中实际上是 MLM Head 之前的处理
            # 因为PRISMModel只包含到CrossAttn的部分。
            # 如果按照EPIModel的完整结构，后面还有CBAT层。
            # 但在PRISM中，我们直接将融合后的footprint注入到当前特征中
            
            # 将 [B, D] 扩展为 [L, B, D] 并加到 enhancer_output
            fused_footprint_proj = self.footprint_inject_proj(fused_footprint)  # f_proj = MLP(f_fused) - footprint特征投影
            fused_footprint_expanded = fused_footprint_proj.unsqueeze(0).expand(enhancers_output.size(0), -1, -1)  # [L, B, D] - 复制到序列长度
            enhancers_output = enhancers_output + self.footprint_alpha * fused_footprint_expanded  # A_en^inj = A_en^res + α*f_proj - footprint注入
            
            # -------------------------------------------------------------------
            # Footprint Injection 2: Residual connect to single CBAT before F-KAN
            # -------------------------------------------------------------------
            # 在PRISMModel中没有显式的 "single CBAT before F-KAN"，
            # F-KAN通常在EPIModel的末端。
            # 在这里，我们将 fused_footprint 返回，或者将其融入到 enhancer_final 中
            # 使得后续如果有 F-KAN 模块（在下游任务微调时），可以利用这个特征。
            # 也可以理解为：在进入MLM Head之前，不仅注入到序列特征，
            # 还可能作为全局特征影响最后的分类。
            
            # 对于PRISM预训练任务，我们已经将其加到了 enhancers_output 中，
            # 这会通过 LayerNorm 和 MLM Head 传播。
            
        # 转换为 (batch, seq_len, features) 用于MLM Head
        enhancer_final = enhancers_output.permute(1, 0, 2)  # [B, L_cnn, D] - MLM Head输入格式
        
        # Layer Norm
        enhancer_final = self.norm_final(enhancer_final)  # LayerNorm(enhancer_final) - 最终层归一化
        
        # MLM Head
        mlm_logits = self.mlm_head(enhancer_final)  # logits = MLMHead(Final) - 掩码语言模型预测
        
        # 返回序列长度映射信息
        seq_length_mapping = {
            'original_length': original_en_length,  # 原始序列长度
            'cnn_length': cnn_seq_length  # CNN后序列长度
        }
        
        # 细胞系分类 Logits
        cell_logits = None
        if self.num_cell_lines > 0:
            # 使用特异性特征进行分类预测
            cell_logits = self.cell_classifier(self.current_z_spec)
        
        return mlm_logits, enhancer_final, promoter_final, seq_length_mapping, fused_footprint, cell_logits  # 返回MLM预测、最终特征和融合footprint
    
    def compute_mlm_loss(self, mlm_logits, original_ids, mask_positions, seq_length_mapping=None):
        """
        计算MLM损失 - Masked Language Model损失计算
        
        Args:
            mlm_logits: [B, L_cnn, vocab_size] 模型预测 (CNN后的序列长度)
            original_ids: [B, L_orig] 原始token IDs
            mask_positions: [B, L_orig] bool tensor, True表示该位置被mask
            seq_length_mapping: 序列长度映射信息
            
        Returns:
            loss: MLM损失
            accuracy: 预测准确率
        """
        B, L_cnn, V = mlm_logits.shape  # B=batch, L_cnn=CNN后序列长度, V=词汇表大小
        L_orig = original_ids.size(1)  # L_orig=原始序列长度
        
        if L_cnn != L_orig:
            K = CNN_KERNEL_SIZE  # CNN卷积核大小
            P = POOL_KERNEL_SIZE  # 池化核大小
            mask_positions_cnn = torch.zeros(B, L_cnn, dtype=torch.bool, device=mask_positions.device)  # CNN长度下的mask位置
            original_ids_cnn = torch.zeros(B, L_cnn, dtype=torch.long, device=original_ids.device)  # CNN长度下的原始ID
            for j in range(L_cnn):
                start = j * P  # 池化起始位置
                end = min(start + P + K - 2, L_orig - 1)  # 池化结束位置
                region_mask = mask_positions[:, start:end+1].any(dim=-1)  # 区域内是否有mask
                center = start + (end - start) // 2  # 区域中心位置
                original_ids_cnn[:, j] = original_ids[:, center]  # 使用中心位置的原始ID
                mask_positions_cnn[:, j] = region_mask  # 区域内有mask则标记
            mask_positions = mask_positions_cnn
            original_ids = original_ids_cnn
        
        # 展平
        mlm_logits_flat = mlm_logits.view(-1, V)  # [B*L_cnn, V] - 展平预测logits
        original_ids_flat = original_ids.view(-1)  # [B*L_cnn] - 展平真实标签
        mask_positions_flat = mask_positions.view(-1)  # [B*L_cnn] - 展平mask位置
        
        # 选择masked位置
        masked_logits = mlm_logits_flat[mask_positions_flat]  # [N_masked, V] - 只取被mask位置的预测
        masked_labels = original_ids_flat[mask_positions_flat]  # [N_masked] - 只取被mask位置的真实标签
        
        # 计算损失
        if masked_logits.size(0) == 0:
            # 没有masked token
            return torch.tensor(0.0, device=mlm_logits.device, requires_grad=True), 0.0  # 返回零损失和零准确率
        
        loss = F.cross_entropy(masked_logits, masked_labels)  # CrossEntropyLoss - 交叉熵损失计算
        
        # 计算准确率
        with torch.no_grad():
            predictions = masked_logits.argmax(dim=-1)  # 获取预测类别
            accuracy = (predictions == masked_labels).float().mean().item()  # 准确率计算
        
        return loss, accuracy  # 返回损失值和准确率


def create_mlm_mask(token_ids, mask_prob=0.15, mask_token_id=4096, 
                    vocab_size=4097, pad_token_id=0, block_mask: bool = False, block_size: int = 3):
    """
    创建MLM mask - 掩码语言模型掩码生成
    
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
    B, L = token_ids.shape  # B=batch_size, L=序列长度
    device = token_ids.device
    
    # 复制原始IDs
    masked_ids = token_ids.clone()  # 创建masked token IDs副本
    original_ids = token_ids.clone()  # 保留原始token IDs用于损失计算
    
    # 创建mask positions (不mask padding和超出vocab范围的token)
    is_valid = (token_ids != pad_token_id) & (token_ids < vocab_size) & (token_ids >= 0)  # 有效token标记
    
    if block_mask:
        mask_positions = torch.zeros(B, L, dtype=torch.bool, device=device)  # 初始化掩码位置
        rand = torch.rand(B, L, device=device)  # 生成随机数
        start_positions = (rand < mask_prob) & is_valid  # 找到掩码起始位置
        for b in range(B):
            starts = torch.nonzero(start_positions[b], as_tuple=False).flatten()  # 获取该batch的起始位置
            for s in starts:
                e = min(s + block_size, L)  # 计算块结束位置
                mask_positions[b, s:e] = True  # 标记该块为mask
        mask_positions = mask_positions & is_valid  # 确保只mask有效token
    else:
        rand = torch.rand(B, L, device=device)  # 生成随机数
        mask_positions = (rand < mask_prob) & is_valid  # 随机选择mask位置
    
    # 对于被选中的token:
    # 80%替换为[MASK], 10%替换为随机token, 10%保持不变
    mask_rand = torch.rand(B, L, device=device)  # 生成掩码策略随机数
    
    # 80%: 替换为[MASK]
    replace_with_mask = mask_positions & (mask_rand < 0.8)  # 80%的mask策略
    masked_ids[replace_with_mask] = mask_token_id  # 替换为[MASK] token
    
    # 10%: 替换为随机token (确保不超出vocab范围)
    replace_with_random = mask_positions & (mask_rand >= 0.8) & (mask_rand < 0.9)  # 10%的随机策略
    # 生成1到vocab_size-1之间的随机token (避免0和mask_token_id)
    random_tokens = torch.randint(1, vocab_size - 1, (B, L), device=device)  # 生成随机token
    masked_ids[replace_with_random] = random_tokens[replace_with_random]  # 替换为随机token
    
    # 10%: 保持不变 (不需要操作)
    
    return masked_ids, mask_positions, original_ids  # 返回masked IDs、掩码位置和原始IDs
