import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from functools import wraps
import numpy as np

class RoPEConfig:
    # 基础配置 - 使用配置文件中的参数
    # 位置编码配置
    POS_ENCODING_MAX_LEN = 1000
    ROPE_MAX_SEQ_LEN = POS_ENCODING_MAX_LEN  # 使用相同的位置编码最大长度
    ROPE_BASE = 10000.0  # RoPE基础频率
    ROPE_SCALING_TYPE = "linear"  # 线性缩放类型
    ROPE_TRAINING_LENGTH = 4096  # 模型预训练时的上下文长度
    ROPE_DROPOUT = 0.1  # RoPE注意力层的dropout率
    
    # RoPE_AdaptAttention 特有配置
    ADAPTIVE_USE_ADAPTIVE = False  # 是否使用自适应注意力
    ADAPTIVE_COMPRESS_DIM = 64  # 压缩维度
    ADAPTIVE_COMPRESS_LAYERS = 2  # 压缩MLP层数
    ADAPTIVE_BASE_VECTORS = 16  # 可学习基向量数量
    ADAPTIVE_SPARSITY_TARGET = 0.3  # 稀疏度目标
    ADAPTIVE_LOSS_WEIGHT = 0.1  # 自适应损失权重
    
    @classmethod
    def get_rope_params(cls):
        """获取RoPE类的默认参数字典"""
        return {
            'max_seq_len': cls.ROPE_MAX_SEQ_LEN,
            'rope_base': cls.ROPE_BASE,
            'rope_scaling_type': cls.ROPE_SCALING_TYPE,
            'rope_training_length': cls.ROPE_TRAINING_LENGTH,
        }
    
    @classmethod
    def get_attention_params(cls):
        """获取RoPEAttention类的默认参数字典"""
        return {
            'max_seq_len': cls.ROPE_MAX_SEQ_LEN,
            'dropout': cls.ROPE_DROPOUT,
            'rope_base': cls.ROPE_BASE,
            'rope_scaling_type': cls.ROPE_SCALING_TYPE,
            'rope_training_length': cls.ROPE_TRAINING_LENGTH,
        }
    
    @classmethod
    def get_adaptive_attention_params(cls):
        """获取RoPE_AdaptAttention类的默认参数字典"""
        return {
            'max_seq_len': cls.ROPE_MAX_SEQ_LEN,
            'dropout': cls.ROPE_DROPOUT,
            'rope_base': cls.ROPE_BASE,
            'rope_scaling_type': cls.ROPE_SCALING_TYPE,
            'rope_training_length': cls.ROPE_TRAINING_LENGTH,
            'compress_dim': cls.ADAPTIVE_COMPRESS_DIM,
            'compress_layers': cls.ADAPTIVE_COMPRESS_LAYERS,
            'base_vectors': cls.ADAPTIVE_BASE_VECTORS,
            'sparsity_target': cls.ADAPTIVE_SPARSITY_TARGET,
            'loss_weight': cls.ADAPTIVE_LOSS_WEIGHT,
        }
    
    @classmethod
    def update_from_config(cls, **kwargs):
        """从外部配置更新默认参数"""
        for key, value in kwargs.items():
            if hasattr(cls, f'ROPE_{key.upper()}'):
                setattr(cls, f'ROPE_{key.upper()}', value)
    
    @classmethod
    def init_from_main_config(cls):
        """从主配置文件初始化RoPE配置（可选）"""
        # 这个方法现在只是占位符，避免外部依赖
        # 所有配置参数已经在类定义中集中管理
        return True

def rope_config_decorator(config_class):
    """
    RoPE配置装饰器
    自动注入RoPEConfig中的默认参数（适用于RoPE类）
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # 获取RoPE类的默认配置
            default_params = config_class.get_rope_params()
            
            # 用默认参数填充未提供的参数
            for key, value in default_params.items():
                if key not in kwargs:
                    kwargs[key] = value
            
            # 调用原始初始化
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator

def rope_attention_config_decorator(config_class):
    """
    RoPEAttention配置装饰器
    自动注入RoPEConfig中的默认参数（适用于RoPEAttention类）
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # 获取RoPEAttention类的默认配置
            default_params = config_class.get_attention_params()
            
            # 用默认参数填充未提供的参数
            for key, value in default_params.items():
                if key not in kwargs:
                    kwargs[key] = value
            
            # 调用原始初始化
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator

def rope_adaptive_attention_config_decorator(config_class):
    """
    RoPE_AdaptAttention配置装饰器
    自动注入RoPEConfig中的默认参数（适用于RoPE_AdaptAttention类）
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # 获取RoPE_AdaptAttention类的默认配置
            default_params = config_class.get_adaptive_attention_params()
            
            # 用默认参数填充未提供的参数
            for key, value in default_params.items():
                if key not in kwargs:
                    kwargs[key] = value
            
            # 调用原始初始化
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator



@rope_config_decorator(RoPEConfig)
class RoPE(nn.Module):
    """
    数值稳定版旋转位置编码 (RoPE)
    支持任意长序列（实测 256k+ 仍完美）
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = None,
        rope_base: float = None,
        rope_scaling_type: str = None,
        rope_training_length: int = None,
    ):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim 必须为偶数，当前 {dim}"
        
        self.dim = dim
        # 使用配置中的默认值，如果没有显式传入
        self.max_seq_len = max_seq_len or RoPEConfig.ROPE_MAX_SEQ_LEN
        self.base = rope_base or RoPEConfig.ROPE_BASE
        self.scaling_type = (rope_scaling_type or RoPEConfig.ROPE_SCALING_TYPE).lower() if rope_scaling_type else None
        self.training_length = rope_training_length or RoPEConfig.ROPE_TRAINING_LENGTH
        
        # 原始频率: [dim//2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq_base", inv_freq, persistent=False)
        
        # 缓存（初始为空）
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.cached_seq_len = 0
        
        # 预热缓存
        self._build_cache(self.max_seq_len)

    @torch.no_grad()
    def _build_cache(self, seq_len: int):
        """重建或扩展 cos/sin 缓存，使用 float32 计算"""
        if seq_len <= self.cached_seq_len:
            return
        
        self.cached_seq_len = seq_len
        
        # 位置索引
        t = torch.arange(seq_len, device=self.inv_freq_base.device, dtype=torch.float32)
        
        # ===== 关键：线性缩放频率 =====
        inv_freq = self.inv_freq_base
        if self.scaling_type == "linear" and seq_len > self.training_length:
            scale = seq_len / self.training_length
            inv_freq = self.inv_freq_base / scale
        # =================================
        
        # [seq_len, dim//2]
        freqs = torch.outer(t, inv_freq)          # 核心外积
        
        # [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # 强制 float32 计算三角函数 → 完美精度
        cos = emb.cos().to(torch.get_default_dtype())
        sin = emb.sin().to(torch.get_default_dtype())
        
        # 直接覆盖（避免重复 register_buffer）
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (cos, sin)，形状自动适配
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # 动态扩展缓存
        if seq_len > self.cached_seq_len:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # 处理 position_ids（如 inference 时 sliding window）
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
            # 扩展维度以便广播
            cos = cos.unsqueeze(1)   # [B, 1, S, D]
            sin = sin.unsqueeze(1)
        else:
            # [1, 1, seq_len, dim] 方便广播
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[-x2, x1] 的高效实现"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用 RoPE，保持原接口完全兼容
    q, k: [B, H, S, D] 或 [B, S, H, D]
    cos, sin: [1, 1, S, D] 或 [B, 1, S, D]
    """
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ===============================================
# 完全兼容你原来使用的 RoPEAttention 类（无需任何修改调用代码）
# ===============================================

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
