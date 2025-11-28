from models.pleat.RoPE import *


@rope_attention_config_decorator(RoPEConfig)
class RoPEAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        dropout: float = None,
        rope_base: float = None,
        rope_scaling_type: str = None,
        rope_training_length: int = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 关键：使用新版数值稳定 RoPE
        self.rope = RoPE(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            rope_scaling_type=rope_scaling_type,
            rope_training_length=rope_training_length,
        )
        
        self.dropout = nn.Dropout(dropout or RoPEConfig.ROPE_DROPOUT)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(q, seq_len=L, position_ids=position_ids)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out

@rope_adaptive_attention_config_decorator(RoPEConfig)
class RoPE_AdaptAttention(nn.Module):
    """
    Adaptive RoPE Attention with token selection for EP prediction
    
    Features:
    - Meta-parameterized scoring using dictionary learning
    - Unified information bottleneck loss
    - Sequence-specific modulation (positional + structural)
    - Fully compatible with RoPEAttention interface
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        dropout: float = None,
        rope_base: float = None,
        rope_scaling_type: str = None,
        rope_training_length: int = None,
        # Adaptive特有参数
        compress_dim: int = 64,          # 压缩维度 d_z
        base_vectors: int = 16,         # 字典基向量数 K (从num_bases重命名)
        sparsity_target: float = 0.4,    # token保留率 ρ (从keep_ratio重命名)
        loss_weight: float = 0.01,       # 自适应损失权重 (从adaptive_loss_weight重命名)
        compress_layers: int = 2,       # 压缩MLP层数
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.d_model = d_model
        self.sparsity_target = sparsity_target
        self.loss_weight = loss_weight
        
        # 标准组件（与RoPEAttention相同）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RoPE(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            rope_scaling_type=rope_scaling_type,
            rope_training_length=rope_training_length,
        )
        
        self.dropout = nn.Dropout(dropout or RoPEConfig.ROPE_DROPOUT)
        
        # Adaptive组件
        
        # 1. 压缩MLP (d_model -> d_z)
        self.compress = nn.Sequential(
            nn.Linear(d_model, compress_dim * 2),
            nn.LayerNorm(compress_dim * 2),
            nn.GELU(),
            nn.Linear(compress_dim * 2, compress_dim),
        )
        
        # 2. 可学习基向量字典 {v_k}
        self.bases = nn.Parameter(torch.randn(base_vectors, compress_dim) / np.sqrt(compress_dim))
        
        # 3. 元参数MLP（生成组合权重α）
        self.meta_net = nn.Sequential(
            nn.Linear(d_model, base_vectors * 2),
            nn.GELU(),
            nn.Linear(base_vectors * 2, base_vectors),
        )
        
        # 4. 位置先验（可学习傅里叶谱）
        self.pos_amplitude = nn.Parameter(torch.randn(6) * 0.1)  # 固定为6个频率
        self.pos_phase = nn.Parameter(torch.rand(6) * 2 * np.pi)
        
        # 5. 结构感知权重MLP
        self.struct_gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # [λ_local, λ_global]
            nn.Softmax(dim=-1),
        )
        
        # 6. CLS查询向量（用于全局重要性）
        self.global_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        
        # 7. 梯度贡献度缓存（EMA）
        self.register_buffer('contrib_ema', None)
        self.ema_decay = 0.9
        
    def _position_modulation(self, L: int, device: torch.device) -> torch.Tensor:
        """
        位置调制 m_pos(i) = 1 + 0.5·tanh(Σ a_f·cos(2πf·i/L + φ_f))
        
        Args:
            L: 序列长度
            device: 设备
            
        Returns:
            位置调制向量 [L]
        """
        pos = torch.arange(L, device=device, dtype=torch.float32)
        modulation = 0.0
        
        for f in range(len(self.pos_amplitude)):
            freq = (f + 1) * 2.0 * np.pi / L
            modulation = modulation + self.pos_amplitude[f] * torch.cos(
                freq * pos + self.pos_phase[f]
            )
        
        return 1.0 + 0.5 * torch.tanh(modulation)  # 范围[0.5, 1.5]
    
    def _structure_modulation(self, x: torch.Tensor) -> torch.Tensor:
        """
        结构调制 m_struct = λ_local·r_local + λ_global·r_global
        
        Args:
            x: 输入张量 [B, L, D]
            
        Returns:
            结构调制向量 [B, L]
        """
        B, L, D = x.shape
        
        # 局部重要性：与邻居的余弦相似度
        x_norm = F.normalize(x, dim=-1, p=2)
        x_shift_prev = torch.cat([x_norm[:, :1], x_norm[:, :-1]], dim=1)
        x_shift_next = torch.cat([x_norm[:, 1:], x_norm[:, -1:]], dim=1)
        
        local_sim = (
            (x_norm * x_shift_prev).sum(dim=-1) +
            (x_norm * x_shift_next).sum(dim=-1)
        ) / 2.0  # [B, L]
        
        # 全局重要性：与全局query的注意力
        query = self.global_query.expand(B, -1)  # [B, D]
        global_attn = torch.matmul(query.unsqueeze(1), x.transpose(1, 2)).squeeze(1)  # [B, L]
        global_sim = torch.softmax(global_attn * self.scale, dim=-1)
        
        # 自适应权重
        global_state = x.mean(dim=1)  # [B, D]
        lambdas = self.struct_gate(global_state)  # [B, 2]
        
        # 组合
        modulation = lambdas[:, 0:1] * local_sim + lambdas[:, 1:2] * global_sim
        
        return modulation
    
    def _compute_scores(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算token重要性评分
        
        Args:
            x: 输入张量 [B, L, D]
            
        Returns:
            scores: 重要性评分 [B, L]
            debug_info: 调试信息字典
        """
        B, L, D = x.shape
        
        # Step 1: 压缩
        z = self.compress(x)  # [B, L, d_z]
        
        # Step 2: 元参数化组合
        global_repr = x.mean(dim=1)  # [B, D]
        alpha = torch.softmax(self.meta_net(global_repr), dim=-1)  # [B, K]
        
        # w = Σ α_k · v_k，然后归一化
        w = torch.matmul(alpha, self.bases)  # [B, d_z]
        w = F.normalize(w, dim=-1, p=2)
        
        # 原始评分 s' = σ(w^T z)
        raw_score = torch.sigmoid(torch.einsum('bd,bld->bl', w, z))  # [B, L]
        
        # Step 3: 位置调制
        m_pos = self._position_modulation(L, x.device)  # [L]
        
        # Step 4: 结构调制
        m_struct = self._structure_modulation(x)  # [B, L]
        
        # Step 5: 最终评分
        final_score = raw_score * m_pos.unsqueeze(0) * m_struct  # [B, L]
        
        # 归一化：保持期望保留率
        final_score = final_score / (final_score.mean(dim=-1, keepdim=True) + 1e-6)
        
        debug_info = {
            'raw_score': raw_score.detach(),
            'm_pos': m_pos.detach(),
            'm_struct': m_struct.detach(),
            'alpha': alpha.detach(),
        }
        
        return final_score, debug_info
    
    def _adaptive_loss(self, scores: torch.Tensor, contrib: torch.Tensor) -> torch.Tensor:
        """
        统一自适应损失：L = E[s·(1-c)] + γ·KL(P||U)
        
        Args:
            scores: token重要性评分 [B, L]
            contrib: 梯度贡献度（EMA）[B, L]
            
        Returns:
            自适应损失值
        """
        # 效率项：惩罚低贡献token的高评分
        efficiency_term = (scores * (1.0 - contrib)).mean()
        
        # 负载均衡项：防止评分坍缩
        mean_scores = scores.mean(dim=0)  # [L]
        prob_scores = torch.softmax(mean_scores, dim=-1)
        uniform_target = torch.ones_like(prob_scores) / prob_scores.shape[0]
        
        balance_term = F.kl_div(
            torch.log(prob_scores + 1e-8),
            uniform_target,
            reduction='batchmean'
        )
        
        return efficiency_term + 0.1 * balance_term
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播（接口与RoPEAttention一致）
        
        Args:
            x: 输入张量 [B, L, D]
            attention_mask: 注意力掩码 [B, 1, 1, L] 或 None
            position_ids: 位置ID [B, L] 或 None
            return_loss: 是否返回adaptive_loss
        
        Returns:
            如果return_loss=True: (输出 [B,L,D], 自适应损失 [标量])
            否则: 输出 [B,L,D]
        """
        B, L, D = x.shape
        
        # Step 1: 计算token重要性评分
        scores, debug = self._compute_scores(x)  # [B, L]
        
        # Step 2: 标准QKV投影
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状: [B, H, L, d_head]
        
        # Step 3: 应用RoPE
        cos, sin = self.rope(q, seq_len=L, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Step 4: Token选择（Top-k稀疏化）
        k_select = max(1, int(L * self.sparsity_target))
        
        # 选择top-k indices
        topk_scores, topk_idx = torch.topk(scores, k_select, dim=-1, sorted=False)  # [B, k]
        
        # 高效gather操作
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)  # [B,1,1,1]
        head_idx = torch.arange(self.num_heads, device=x.device).view(1, self.num_heads, 1, 1)  # [1,H,1,1]
        token_idx = topk_idx.view(B, 1, k_select, 1)  # [B,1,k,1]
        
        # 扩展索引
        token_idx_exp = token_idx.expand(B, self.num_heads, k_select, self.head_dim)
        
        q_selected = torch.gather(q, 2, token_idx_exp)  # [B, H, k, d]
        k_selected = torch.gather(k, 2, token_idx_exp)
        v_selected = torch.gather(v, 2, token_idx_exp)
        
        # Step 5: 稀疏注意力计算
        attn = torch.matmul(q_selected, k_selected.transpose(-2, -1)) * self.scale  # [B,H,k,k]
        
        # 处理attention_mask（如果需要）
        if attention_mask is not None and attention_mask.numel() > 1:
            # 简化：对选中的token，mask通常不需要调整（EP任务中序列无padding）
            pass
        
        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out_selected = torch.matmul(attn_weights, v_selected)  # [B, H, k, d]
        
        # Step 6: 恢复完整序列
        # 创建全零输出（未选中token输出0，依赖residual connection）
        out_full = torch.zeros(B, self.num_heads, L, self.head_dim, 
                              device=x.device, dtype=x.dtype)
        
        # Scatter回去
        out_full.scatter_(2, token_idx_exp, out_selected)
        
        # Reshape & 输出投影
        out = out_full.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        # Step 7: 计算自适应损失
        if return_loss and self.training:
            # 更新贡献度EMA（使用注意力权重的统计量）
            with torch.no_grad():
                # 使用选中token的平均注意力权重作为贡献度代理
                contrib_selected = attn_weights.mean(dim=(1, 2))  # [B, k]
                
                # 扩展到完整序列
                contrib_full = torch.zeros(B, L, device=x.device)
                contrib_full.scatter_(1, topk_idx, contrib_selected)
                
                # EMA更新
                if self.contrib_ema is None or self.contrib_ema.shape != contrib_full.shape:
                    self.contrib_ema = contrib_full
                else:
                    self.contrib_ema = (
                        self.ema_decay * self.contrib_ema + 
                        (1 - self.ema_decay) * contrib_full
                    )
            
            adaptive_loss = self._adaptive_loss(scores, self.contrib_ema)
            adaptive_loss = adaptive_loss * self.loss_weight
            
            return out, adaptive_loss
        
        else:
            if return_loss:
                return out, torch.tensor(0.0, device=x.device, requires_grad=True)
            return out

@rope_attention_config_decorator(RoPEConfig)
class RoPE_CausalBlockAttention(nn.Module):
    """
    Causal Block Attention with RoPE
    
    Features:
    - Block-level causality: block_j <= block_i
    - Intra-block full attention (recommended for sequence tasks)
    - Fully compatible with RoPEAttention interface
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 128,              # Block大小
        max_seq_len: int = None,
        dropout: float = None,
        rope_base: float = None,
        rope_scaling_type: str = None,
        rope_training_length: int = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} 必须能被 num_heads={num_heads} 整除"
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        
        # 标准投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE位置编码
        self.rope = RoPE(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            rope_scaling_type=rope_scaling_type,
            rope_training_length=rope_training_length,
        )
        
        self.dropout = nn.Dropout(dropout or RoPEConfig.ROPE_DROPOUT)
        
        # Causal Block Mask缓存
        self.register_buffer("causal_block_mask_cached", torch.empty(0), persistent=False)
        self.cached_mask_seq_len = 0
    
    def _build_causal_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        构建Causal Block Attention Mask
        
        规则：
        - Block内全连接（同一block的token可以互相attend）
        - Block间因果（只能attend到当前及之前的block）
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            mask: [seq_len, seq_len] bool tensor, True表示可以attend
        """
        # 计算每个token属于哪个block
        # 例如: block_size=4, seq_len=12 -> [0,0,0,0, 1,1,1,1, 2,2,2,2]
        block_ids = torch.arange(seq_len, device=device) // self.block_size
        
        # Block级因果: token_i只能attend到block_id <= block_i的所有token
        # [seq_len, 1] >= [1, seq_len] -> [seq_len, seq_len]
        mask = block_ids.unsqueeze(1) >= block_ids.unsqueeze(0)
        
        return mask
    
    def _get_causal_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取causal block mask（带缓存优化）
        
        Returns:
            attention_mask: [1, 1, seq_len, seq_len]
                           可attend位置为0，不可attend位置为-inf
        """
        # 检查是否需要重建缓存
        if (seq_len != self.cached_mask_seq_len or 
            self.causal_block_mask_cached.numel() == 0 or
            self.causal_block_mask_cached.device != device):
            
            # 构建bool mask
            mask = self._build_causal_block_mask(seq_len, device)  # [L, L]
            
            # 转换为attention mask格式
            # True (可attend) -> 0, False (不可attend) -> -inf
            attn_mask = torch.zeros(1, 1, seq_len, seq_len, 
                                   device=device, 
                                   dtype=torch.get_default_dtype())
            attn_mask.masked_fill_(~mask, float('-inf'))
            
            # 更新缓存
            self.causal_block_mask_cached = attn_mask
            self.cached_mask_seq_len = seq_len
        
        return self.causal_block_mask_cached
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播（接口与RoPEAttention完全一致）
        
        Args:
            x: 输入张量 [B, L, D]
            attention_mask: 额外的注意力掩码 [B, 1, 1, L] 或 None
                          (例如padding mask)
            position_ids: 位置ID [B, L] 或 None
            
        Returns:
            输出张量 [B, L, D]
        """
        B, L, D = x.shape
        
        # QKV投影: [B, L, D] -> [B, L, H, d_head] -> [B, H, L, d_head]
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用RoPE位置编码
        cos, sin = self.rope(q, seq_len=L, position_ids=position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 计算注意力分数: [B, H, L, L]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用Causal Block Mask: [B, H, L, L] + [1, 1, L, L]
        causal_block_mask = self._get_causal_block_mask(L, x.device)
        attn = attn + causal_block_mask
        
        # 应用额外的attention_mask（如padding mask）
        if attention_mask is not None:
            attn = attn + attention_mask
        
        # Softmax归一化
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和: [B, H, L, L] @ [B, H, L, d_head] -> [B, H, L, d_head]
        out = torch.matmul(attn, v)
        
        # 恢复形状: [B, H, L, d_head] -> [B, L, H, d_head] -> [B, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out

class CBAT(nn.Module):
    """
    Causal Block Adaptive Transformer Module
    
    结构:
    - Module 1: 7x7conv -> sep_3x3 -> CausalBlockAttn -> 1x1conv -> gate (residual from 7x7)
    - Module 2: AdaptAttn -> FFN -> gate (residual from AdaptAttn)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        img_size: int = 32,  # 假设输入可reshape为正方形，如32x32
        block_size: int = 128,
        max_seq_len: int = None,
        dropout: float = None,
        rope_base: float = None,
        rope_scaling_type: str = None,
        rope_training_length: int = None,
        # AdaptAttention参数
        compress_dim: int = 64,
        base_vectors: int = 16,
        sparsity_target: float = 0.4,
        loss_weight: float = 0.01,
        compress_layers: int = 2,
        # FFN参数
        ffn_hidden_dim: int = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.img_size = img_size
        self.seq_len = img_size * img_size
        
        if ffn_hidden_dim is None:
            ffn_hidden_dim = d_model * 4
        
        # ============ Module 1 组件 ============
        
        # 7x7 Conv (提取特征 + 残差路径)
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        
        # 可分离3x3卷积 (Depthwise + Pointwise)
        self.separable_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, 
                     groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            # Pointwise
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
        )
        
        # RoPE Causal Block Attention
        self.causal_block_attn = RoPE_CausalBlockAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_base=rope_base,
            rope_scaling_type=rope_scaling_type,
            rope_training_length=rope_training_length,
        )
        
        # 1x1 Conv
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
        )
        
        # Gate for Module 1 (门控融合)
        self.gate_1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # ============ Module 2 组件 ============
        
        # RoPE Adaptive Attention
        self.adaptive_attn = RoPE_AdaptAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_base=rope_base,
            rope_scaling_type=rope_scaling_type,
            rope_training_length=rope_training_length,
            compress_dim=compress_dim,
            base_vectors=base_vectors,
            sparsity_target=sparsity_target,
            loss_weight=loss_weight,
            compress_layers=compress_layers,
        )
        
        # FFN (前馈神经网络)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout or RoPEConfig.ROPE_DROPOUT),
            nn.Linear(ffn_hidden_dim, d_model),
            nn.Dropout(dropout or RoPEConfig.ROPE_DROPOUT),
        )
        
        # Gate for Module 2
        self.gate_2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # Layer Norms
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] -> [B, D, H, W]"""
        B, L, D = x.shape
        H = W = self.img_size
        
        # 如果序列长度不等于H*W，进行填充或截断
        if L != H * W:
            if L < H * W:
                # 填充到H*W长度
                padding_size = H * W - L
                x = torch.nn.functional.pad(x, (0, 0, 0, padding_size))
            else:
                # 截断到H*W长度
                x = x[:, :H * W, :]
        
        # [B, L, D] -> [B, H, W, D] -> [B, D, H, W]
        return x.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
    
    def _to_1d(self, x: torch.Tensor) -> torch.Tensor:
        """[B, D, H, W] -> [B, L, D]"""
        B, D, H, W = x.shape
        # [B, D, H, W] -> [B, D, L] -> [B, L, D]
        return x.flatten(2).transpose(1, 2).contiguous()
    
    def _to_1d_with_length(self, x: torch.Tensor, original_length: int) -> torch.Tensor:
        """[B, D, H, W] -> [B, L, D] 并恢复到原始长度"""
        B, D, H, W = x.shape
        # [B, D, H, W] -> [B, D, L]
        x_flat = x.flatten(2)
        
        # 如果原始长度小于H*W，截断到原始长度
        if original_length < H * W:
            x_flat = x_flat[:, :, :original_length]
        
        # [B, D, L] -> [B, L, D]
        return x_flat.transpose(1, 2).contiguous()
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, L, D] 输入序列
            attention_mask: 注意力掩码
            position_ids: 位置ID
            return_loss: 是否返回adaptive loss
            
        Returns:
            output [B, L, D] 或 (output, adaptive_loss)
        """
        B, L, D = x.shape
        original_length = L  # 保存原始长度
        identity = x  # 全局残差
        
        # ============ Module 1 ============
        
        # Pre-norm
        x_norm = self.norm_1(x)  # [B, L, D]
        
        # 转2D: [B, L, D] -> [B, D, H, W]
        x_2d = self._to_2d(x_norm)
        
        # 7x7 Conv (同时作为残差路径)
        residual_7x7 = self.conv_7x7(x_2d)  # [B, D, H, W]
        
        # 可分离3x3
        x_2d = self.separable_conv(residual_7x7)  # [B, D, H, W]
        
        # 转1D: [B, D, H, W] -> [B, L, D]
        x_1d = self._to_1d_with_length(x_2d, original_length)
        
        # Causal Block Attention
        x_attn = self.causal_block_attn(
            x_1d,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )  # [B, L, D]
        
        # 残差连接1: 7*7和CausalBlock之间
        residual_7x7_1d = self._to_1d_with_length(residual_7x7, original_length)
        x_attn = x_attn + residual_7x7_1d  # [B, L, D]
        
        # 转2D
        x_2d = self._to_2d(x_attn)  # [B, D, H, W]
        
        # 1x1 Conv
        x_2d = self.conv_1x1(x_2d)  # [B, D, H, W]
        
        # 转1D准备gate
        x_main = self._to_1d_with_length(x_2d, original_length)
        residual_1 = self._to_1d_with_length(residual_7x7, original_length)
        
        # Gate融合 (主路径 vs 7x7残差)
        gate_input = torch.cat([x_main, residual_1], dim=-1)  # [B, L, 2D]
        gate_weight = self.gate_1(gate_input)  # [B, L, D]
        x_out_1 = gate_weight * x_main + (1 - gate_weight) * residual_1  # [B, L, D]
        
        # 残差连接2: CausalBlock和模块一门控之间
        x_out_1 = x_out_1 + x_attn  # [B, L, D]
        
        # 全局残差连接 - 确保维度匹配
        if x_out_1.shape[1] != identity.shape[1]:
            # 如果维度不匹配，截断或填充identity
            if x_out_1.shape[1] > identity.shape[1]:
                x_out_1 = x_out_1[:, :identity.shape[1], :]
            else:
                padding_size = identity.shape[1] - x_out_1.shape[1]
                x_out_1 = torch.nn.functional.pad(x_out_1, (0, 0, 0, padding_size))
        
        x_out_1 = x_out_1 + identity  # [B, L, D]
        
        # ============ Module 2 ============
        
        # Pre-norm
        x_norm = self.norm_2(x_out_1)  # [B, L, D]
        
        # Adaptive Attention (返回loss)
        if return_loss and self.training:
            x_adapt, adaptive_loss = self.adaptive_attn(
                x_norm,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_loss=True,
            )  # [B, L, D], scalar
        else:
            if return_loss:
                x_adapt, adaptive_loss = self.adaptive_attn(
                    x_norm,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_loss=True,
                )
            else:
                x_adapt = self.adaptive_attn(
                    x_norm,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_loss=False,
                )
                adaptive_loss = torch.tensor(0.0, device=x.device)
        
        # AdaptAttn输出作为残差路径
        residual_2 = x_adapt  # [B, L, D]
        
        # FFN
        x_ffn = self.norm_3(x_adapt)  # [B, L, D]
        x_ffn = self.ffn(x_ffn)  # [B, L, D]
        
        # Gate融合 (FFN输出 vs AdaptAttn残差)
        gate_input = torch.cat([x_ffn, residual_2], dim=-1)  # [B, L, 2D]
        gate_weight = self.gate_2(gate_input)  # [B, L, D]
        x_out_2 = gate_weight * x_ffn + (1 - gate_weight) * residual_2  # [B, L, D]
        
        # 残差连接3: 模块一门控和模块二门控之间
        x_out_2 = x_out_2 + x_out_1  # [B, L, D]
        
        # 全局残差连接 - 确保维度匹配
        if x_out_2.shape[1] != x_out_1.shape[1]:
            # 如果维度不匹配，截断或填充x_out_2
            if x_out_2.shape[1] > x_out_1.shape[1]:
                x_out_2 = x_out_2[:, :x_out_1.shape[1], :]
            else:
                padding_size = x_out_1.shape[1] - x_out_2.shape[1]
                x_out_2 = torch.nn.functional.pad(x_out_2, (0, 0, 0, padding_size))
        
        output = x_out_2 + x_out_1  # [B, L, D]
        
        if return_loss:
            return output, adaptive_loss
        else:
            return output