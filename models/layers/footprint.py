"""
LCWnet: Learnable Continuous Wavelet Transform Network
可学习连续小波变换网络 - 用于DNA序列的时频特征提取

核心思想：
1. 在傅里叶域直接构造可学习的母小波 F[ψ_θ](k)
2. 使用紧支撑 [0, k_max] 自动满足容许条件
3. 通过MLP参数化实部和虚部，端到端优化
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络功能模块
import numpy as np  # 数值计算库
from typing import Tuple, Optional  # 类型提示
from functools import wraps  # 函数装饰器工具


class FootprintConfig:
    K_MAX = 0.5  # 母小波频率支撑上限，定义紧支撑区间 [0, 0.5]
    HIDDEN_DIM = 64  # MLP隐藏层维度
    NUM_LAYERS = 3  # MLP层数
    FUSION_TYPE = 'attention'  # 多尺度特征融合方式
    LOGSPACE_START = 0.0  # 对数尺度起始值 (10^0 = 1)
    LOGSPACE_END = 2.0  # 对数尺度结束值 (10^2 = 100)
    NUM_SCALES = 32  # CWT尺度数量

    @classmethod
    def get_mother_params(cls):
        """获取可学习母小波参数配置"""
        return {
            'k_max': cls.K_MAX,  # 频率支撑上限
            'hidden_dim': cls.HIDDEN_DIM,  # MLP隐藏维度
            'num_layers': cls.NUM_LAYERS,  # MLP层数
        }

    @classmethod
    def get_cwt_params(cls):
        """获取连续小波变换参数配置"""
        scales = np.logspace(cls.LOGSPACE_START, cls.LOGSPACE_END, cls.NUM_SCALES).tolist()  # 生成对数尺度序列
        return {
            'scales': scales,  # CWT尺度参数列表
            'k_max': cls.K_MAX,  # 母小波频率上限
            'hidden_dim': cls.HIDDEN_DIM,  # 母小波MLP隐藏维度
            'num_layers': cls.NUM_LAYERS,  # 母小波MLP层数
        }

    @classmethod
    def get_module_params(cls):
        """获取Footprint模块参数配置"""
        return {
            'fusion_type': cls.FUSION_TYPE,  # 特征融合方式
        }


def footprint_mother_config_decorator(config_class):
    """可学习母小波参数配置装饰器"""
    def decorator(cls):
        original_init = cls.__init__  # 保存原始初始化函数

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            defaults = config_class.get_mother_params()  # 获取默认母小波参数
            for k, v in defaults.items():
                kwargs.setdefault(k, v)  # 设置默认参数
            original_init(self, *args, **kwargs)  # 调用原始初始化
        cls.__init__ = new_init  # 替换为新的初始化函数
        return cls
    return decorator


def footprint_cwt_config_decorator(config_class):
    """连续小波变换参数配置装饰器"""
    def decorator(cls):
        original_init = cls.__init__  # 保存原始初始化函数

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            defaults = config_class.get_cwt_params()  # 获取默认CWT参数
            for k, v in defaults.items():
                kwargs.setdefault(k, v)  # 设置默认参数
            original_init(self, *args, **kwargs)  # 调用原始初始化
        cls.__init__ = new_init  # 替换为新的初始化函数
        return cls
    return decorator


def footprint_module_config_decorator(config_class):
    """Footprint模块参数配置装饰器"""
    def decorator(cls):
        original_init = cls.__init__  # 保存原始初始化函数

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            defaults = config_class.get_module_params()  # 获取默认模块参数
            for k, v in defaults.items():
                setattr(self, k, kwargs.pop(k, v))  # 设置为实例属性
            original_init(self, *args, **kwargs)  # 调用原始初始化
        cls.__init__ = new_init  # 替换为新的初始化函数
        return cls
    return decorator


@footprint_mother_config_decorator(FootprintConfig)
class LearnableMotherWavelet(nn.Module):
    """
    可学习母小波模块
    
    在傅里叶域构造复值母小波: F[ψ_θ](k) = F[ψ^(r)](k) + i·F[ψ^(i)](k)
    
    Args:
        k_max: 频率支撑上限，定义紧支撑区间 [0, k_max]
        hidden_dim: MLP隐藏层维度
        num_layers: MLP层数
    """
    
    def __init__(
        self,
        k_max: float,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.k_max = k_max  # 频率支撑上限
        
        # 实部MLP: MLP_θr(k) - 学习母小波实部频域表示
        layers_real = []
        layers_real.append(nn.Linear(1, hidden_dim))  # 输入维度1 (频率k)
        layers_real.append(nn.ReLU())  # 激活函数
        
        for _ in range(num_layers - 2):
            layers_real.append(nn.Linear(hidden_dim, hidden_dim))  # 隐藏层
            layers_real.append(nn.ReLU())  # 激活函数
        
        layers_real.append(nn.Linear(hidden_dim, 1))  # 输出层线性激活 (维度1)
        self.mlp_real = nn.Sequential(*layers_real)  # 实部MLP网络
        
        # 虚部MLP: MLP_θi(k) - 学习母小波虚部频域表示
        layers_imag = []
        layers_imag.append(nn.Linear(1, hidden_dim))  # 输入维度1 (频率k)
        layers_imag.append(nn.ReLU())  # 激活函数
        
        for _ in range(num_layers - 2):
            layers_imag.append(nn.Linear(hidden_dim, hidden_dim))  # 隐藏层
            layers_imag.append(nn.ReLU())  # 激活函数
        
        layers_imag.append(nn.Linear(hidden_dim, 1))  # 输出层线性激活 (维度1)
        self.mlp_imag = nn.Sequential(*layers_imag)  # 虚部MLP网络
    
    def _compact_support_window(self, k: torch.Tensor) -> torch.Tensor:
        """
        紧支撑窗函数: ReLU(k) · ReLU(k_max - k)
        
        确保输出只在 [0, k_max] 区间非零
        
        Args:
            k: 频率张量 [..., N]
            
        Returns:
            窗函数值 [..., N]
        """
        return F.relu(k) * F.relu(self.k_max - k)  # 构建紧支撑窗函数
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        计算母小波在频域的值
        
        F[ψ^(r)](k) = MLP_θr(k) · ReLU(k) · ReLU(k_max - k)  (公式4)
        F[ψ^(i)](k) = MLP_θi(k) · ReLU(k) · ReLU(k_max - k)  (公式5)
        F[ψ_θ](k) = F[ψ^(r)](k) + i·F[ψ^(i)](k)            (公式6)
        
        Args:
            k: 频率张量 [..., N, 1]
            
        Returns:
            复值母小波 [..., N] (复数张量)
        """
        # 计算紧支撑窗函数
        window = self._compact_support_window(k.squeeze(-1))  # [..., N] - 紧支撑窗函数
        
        # 实部: MLP_θr(k) · window
        psi_real = self.mlp_real(k).squeeze(-1) * window  # [..., N] - 母小波实部
        
        # 虚部: MLP_θi(k) · window
        psi_imag = self.mlp_imag(k).squeeze(-1) * window  # [..., N] - 母小波虚部
        
        # 合成复值母小波
        psi_complex = torch.complex(psi_real, psi_imag)  # [..., N] - 复数母小波
        
        return psi_complex  # 返回复值母小波


@footprint_cwt_config_decorator(FootprintConfig)
class ContinuousWaveletTransform(nn.Module):
    """
    连续小波变换模块
    
    使用可学习母小波在傅里叶域高效计算CWT:
    T_wav^ψ f(a, b) = √a · F^(-1)[F[f](·) · F[ψ]*(a·)](b)  (公式3)
    
    Args:
        scales: 尺度参数列表 [a_0, ..., a_{S-1}]
        k_max: 母小波频率支撑上限
        hidden_dim: 母小波MLP隐藏维度
        num_layers: 母小波MLP层数
    """
    
    def __init__(
        self,
        scales: list,
        k_max: float,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        self.num_scales = len(scales)
        self.mother_wavelet = LearnableMotherWavelet(
            k_max=k_max,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算连续小波系数 (CWC)
        
        算法流程:
        1. 计算输入信号的傅里叶变换: F_f = DFT(f)
        2. 对每个尺度 a_s:
           a. 生成缩放母小波: Ψ_{a_s} = F[ψ_θ]*(a_s · q)
           b. 频域相乘: Y_{a_s} = F_f ⊙ Ψ_{a_s}
           c. 逆变换: T_s = IDFT(Y_{a_s})
        3. 堆叠所有尺度: CWC ∈ C^{S×L}
        
        Args:
            x: 输入序列 [B, L, D]
            
        Returns:
            cwc: 连续小波系数 [B, S, L, D] (复数张量)
        """
        B, L, D = x.shape
        device = x.device
        
        # Step 1: 计算输入信号的傅里叶变换
        # [B, L, D] -> [B, D, L] -> FFT -> [B, D, L]
        x_freq = torch.fft.rfft(x.transpose(1, 2), dim=-1)  # [B, D, L//2+1]
        freq_len = x_freq.shape[-1]
        
        # 构造归一化频率向量 q ∈ [0, 0.5]
        # q_l = l / L, l = 0, ..., L//2
        q = torch.linspace(0, 0.5, freq_len, device=device)  # [freq_len]
        
        # Step 2: 迭代计算每个尺度的CWC
        cwc_list = []
        
        for scale in self.scales:
            # Step 2a: 生成该尺度的母小波频域响应
            # k = a_s · q
            k_scaled = scale * q  # [freq_len]
            k_scaled = k_scaled.unsqueeze(-1)  # [freq_len, 1]
            
            # F[ψ_θ]*(a_s · q) - 计算并取复共轭
            psi_freq = self.mother_wavelet(k_scaled)  # [freq_len]
            psi_freq_conj = torch.conj(psi_freq)  # 复共轭
            
            # Step 2b: 频域相乘
            # [B, D, freq_len] * [freq_len] -> [B, D, freq_len]
            y_freq = x_freq * psi_freq_conj.unsqueeze(0).unsqueeze(0)
            
            # Step 2c: 逆傅里叶变换得到时域响应
            # [B, D, freq_len] -> IFFT -> [B, D, L]
            cwc_scale = torch.fft.irfft(y_freq, n=L, dim=-1)  # [B, D, L]
            
            # 乘以尺度因子 √a
            cwc_scale = cwc_scale * torch.sqrt(scale)
            
            cwc_list.append(cwc_scale)
        
        # Step 3: 堆叠所有尺度
        # List of [B, D, L] -> [B, S, D, L]
        cwc = torch.stack(cwc_list, dim=1)  # [B, S, D, L]
        
        # 转换为 [B, S, L, D] 以匹配序列格式
        cwc = cwc.permute(0, 1, 3, 2)  # [B, S, L, D]
        
        return cwc


@footprint_module_config_decorator(FootprintConfig)
class LCWnetFootprint(nn.Module):
    """
    LCWnet Footprint模块 - 用于DNA序列的时频特征提取
    
    架构:
    1. 可学习连续小波变换 (CWT)
    2. 尺度-时间特征融合
    3. 门控机制与原始特征融合
    
    Args:
        d_model: 输入特征维度
        scales: CWT尺度列表
        k_max: 母小波频率上限
        hidden_dim: 母小波MLP隐藏维度
        num_layers: 母小波MLP层数
        fusion_type: 特征融合方式 ('mean', 'max', 'attention')
    """
    
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion_type = getattr(self, 'fusion_type', FootprintConfig.FUSION_TYPE)
        self.cwt = ContinuousWaveletTransform()
        num_scales = self.cwt.num_scales
        
        # 尺度-时间特征融合
        if self.fusion_type == 'attention':
            # 使用注意力机制融合不同尺度
            self.scale_attention = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1),
            )
        elif self.fusion_type == 'mean':
            pass  # 简单平均
        elif self.fusion_type == 'max':
            pass  # 最大池化
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        
        # 特征投影层 (将CWT特征投影回d_model维度)
        self.cwt_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # 门控融合层 (原始特征 vs CWT特征)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer Norm
        self.norm = nn.LayerNorm(d_model)
        
        # 样本级footprint压缩网络 Φ_CWC (Conv + Pooling)
        # 输入: [B, S, L] (CWC幅度谱)
        # 输出: [B, D_v] (样本级footprint向量)
        # 这里 D_v 设为 d_model 以便与主干融合
        
        self.footprint_compressor = nn.Sequential(
            # 第一层Conv: 压缩尺度维 S -> S/2
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # [B, 8, S/2, L/2]
            
            # 第二层Conv: 进一步提取特征
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),  # 压缩到固定大小 [B, 16, 4, 8]
            
            nn.Flatten(),  # [B, 16*4*8 = 512]
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def _extract_sample_footprint(self, cwc: torch.Tensor) -> torch.Tensor:
        """
        从CWC提取样本级footprint向量 v_i = Φ_CWC(|CWC_i|)
        
        Args:
            cwc: [B, S, L, D] 连续小波系数
            
        Returns:
            v: [B, D_v] 样本级footprint向量
        """
        B, S, L, D = cwc.shape
        
        # 1. 计算幅度谱 M = |CWC|
        # [B, S, L, D] -> [B, S, L] (取D维度的平均或最大值作为代表)
        # 这里我们取D维度的L2范数，表示该位置的总能量
        magnitude = cwc.abs().norm(dim=-1)  # [B, S, L]
        
        # 2. 增加通道维度供Conv2d使用
        magnitude = magnitude.unsqueeze(1)  # [B, 1, S, L]
        
        # 3. 通过压缩网络提取特征
        v = self.footprint_compressor(magnitude)  # [B, D_v]
        
        return v

    def _fuse_scales(self, cwc: torch.Tensor) -> torch.Tensor:
        """
        融合多尺度CWT特征
        
        Args:
            cwc: [B, S, L, D] 连续小波系数
            
        Returns:
            fused: [B, L, D] 融合后的特征
        """
        B, S, L, D = cwc.shape
        
        if self.fusion_type == 'mean':
            # 简单平均
            fused = cwc.mean(dim=1)  # [B, L, D]
            
        elif self.fusion_type == 'max':
            # 最大池化
            fused = cwc.max(dim=1)[0]  # [B, L, D]
            
        elif self.fusion_type == 'attention':
            # 注意力加权融合
            # [B, S, L, D] -> [B*L, S, D]
            cwc_reshaped = cwc.permute(0, 2, 1, 3).reshape(B * L, S, D)
            
            # 计算注意力权重: [B*L, S, D] -> [B*L, S, 1]
            attn_scores = self.scale_attention(cwc_reshaped)  # [B*L, S, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [B*L, S, 1]
            
            # 加权求和: [B*L, S, D] * [B*L, S, 1] -> [B*L, D]
            fused = (cwc_reshaped * attn_weights).sum(dim=1)  # [B*L, D]
            
            # 恢复形状: [B*L, D] -> [B, L, D]
            fused = fused.view(B, L, D)
        
        return fused
    
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [B, L, D]
            residual: 残差连接输入 [B, L, D] (可选)
            
        Returns:
            output: 序列级输出 [B, L, D] (用于继续Transformer流)
            sample_footprint: 样本级footprint向量 [B, D] (用于注入到后续层)
        """
        B, L, D = x.shape
        
        # 保存原始输入用于门控
        identity = x if residual is None else residual
        
        # Step 1: 连续小波变换
        cwc = self.cwt(x)  # [B, S, L, D]
        
        # Step 2: 提取样本级footprint向量 (新增功能)
        sample_footprint = self._extract_sample_footprint(cwc)  # [B, D]
        
        # Step 3: 多尺度特征融合 (用于序列级输出)
        cwt_features = self._fuse_scales(cwc)  # [B, L, D]
        
        # Step 4: CWT特征投影
        cwt_features = self.cwt_proj(cwt_features)  # [B, L, D]
        
        # Step 5: 门控融合 (原始特征 vs CWT特征)
        gate_input = torch.cat([identity, cwt_features], dim=-1)  # [B, L, 2D]
        gate_weight = self.gate(gate_input)  # [B, L, D]
        
        # 门控加权: g·cwt + (1-g)·identity
        output = gate_weight * cwt_features + (1 - gate_weight) * identity
        
        # Step 6: 输出投影和归一化
        output = self.out_proj(output)
        output = self.norm(output)
        
        return output, sample_footprint

    def forward_vector(self, x: torch.Tensor) -> torch.Tensor:
        cwc = self.cwt(x)
        sample_footprint = self._extract_sample_footprint(cwc)
        return sample_footprint


# ============================================================================
# 损失函数部分 (暂未实现，仅注释说明)
# ============================================================================

"""
LCWnet损失函数设计 (待实现)

根据论文，LCWnet的训练采用端到端方式，损失函数包含两部分：

1. **下游任务损失** (Primary Loss):
   - 对于EP预测任务，使用AdaptiveIMMAXLoss (已在EPIModel中实现)
   - Loss_task = AdaptiveIMMAXLoss(predictions, labels)

2. **小波正则化损失** (Wavelet Regularization Loss, 可选):
   - 目的：鼓励学习到的母小波具有良好的时频局部化特性
   - 可能的正则化项：
     a. 频域平滑性: L_smooth = ||∇_k F[ψ_θ](k)||^2
     b. 时域紧凑性: L_compact = ∫ |x|^2 |ψ(x)|^2 dx
     c. 能量归一化: L_energy = (||ψ||_2 - 1)^2

总损失:
Loss_total = Loss_task + λ_reg · Loss_wavelet

其中 λ_reg 是正则化权重超参数 (建议范围: 0.001 - 0.01)

实现建议:
- 在训练循环中，将LCWnetFootprint的输出传递给下游模型
- 计算任务损失后，可选地添加小波正则化项
- 通过反向传播同时优化母小波参数和下游模型参数

示例代码框架:
```python
# 前向传播
cwt_features = lcwnet_footprint(x)
predictions = downstream_model(cwt_features)

# 计算损失
task_loss = criterion(predictions, labels)

# 可选：计算小波正则化损失
# wavelet_reg_loss = compute_wavelet_regularization(lcwnet_footprint.cwt.mother_wavelet)
# total_loss = task_loss + lambda_reg * wavelet_reg_loss

total_loss = task_loss  # 简化版本

# 反向传播
total_loss.backward()
optimizer.step()
```

注意事项:
- 母小波的学习是隐式的，通过任务损失的梯度反向传播实现
- 紧支撑设计保证了容许条件自动满足，无需额外约束
- 正则化项是可选的，对于大多数任务，仅使用任务损失即可
"""
