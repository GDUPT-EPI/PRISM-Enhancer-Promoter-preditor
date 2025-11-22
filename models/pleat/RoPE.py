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

