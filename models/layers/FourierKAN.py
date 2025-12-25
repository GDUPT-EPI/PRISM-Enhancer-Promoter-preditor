import torch
from torch import nn
from typing import Optional
import math


class FourierKAN(nn.Module):
    """
    FourierKAN: 基于傅里叶级数的 Kolmogorov-Arnold Network
    
    数学原理:
    =========
    对于每个输入特征 x_j，计算傅里叶展开：
        φ_i(x_j) = Σ_k [a_{ijk} · cos(k·x̃_j) + b_{ijk} · sin(k·x̃_j)]
    
    其中 x̃_j = tanh(x_j) · π 确保输入在 [-π, π] 范围内
    
    隐藏层：h_i = σ(Σ_j φ_i(x_j))
    输出层：y = W_out · h + bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        grid_size: int = 5,
        width: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        # 增大默认宽度以提升表达能力
        self.width = width or max(4 * in_features, 32)

        k = grid_size + 1  # 频率数量 [0, 1, ..., grid_size]
        
        # 傅里叶系数参数
        self.a = nn.Parameter(torch.empty(self.width, in_features, k))
        self.b = nn.Parameter(torch.empty(self.width, in_features, k))
        
        # 输出层参数
        self.W_out = nn.Parameter(torch.empty(out_features, self.width))
        self.bias_out = nn.Parameter(torch.zeros(out_features))

        # 正确的初始化
        self._init_parameters()
        
        self.act = nn.SiLU()

    def _init_parameters(self):
        """
        初始化策略：保证前向传播时输出方差稳定
        
        分析：
        - φ_i = Σ_{j,k} [a·cos + b·sin]，共 in_features × (grid_size+1) 项
        - cos²(θ) 和 sin²(θ) 期望值均为 0.5
        - 若 a,b ~ N(0, σ²)，则每项方差 ≈ σ² × 0.5
        - 总方差 = in_features × k × σ² × 0.5
        - 令总方差 ≈ 1，则 σ = sqrt(2 / (in_features × k))
        """
        k = self.grid_size + 1
        std = math.sqrt(2.0 / (self.in_features * k))
        
        nn.init.normal_(self.a, mean=0.0, std=std)
        nn.init.normal_(self.b, mean=0.0, std=std)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 严格维度控制版本
        
        维度流：
        x: (bsz, in_features)
        → x_norm: (bsz, in_features) 
        → ang: (bsz, in_features, k)
        → phi: (bsz, width, in_features)
        → s: (bsz, width)
        → out: (bsz, out_features)
        """
        bsz = x.shape[0]
        
        # ========== 【关键修复1】输入归一化 ==========
        # tanh 将任意范围映射到 (-1, 1)，乘以 π 得到 (-π, π)
        # 这保证 cos(kx), sin(kx) 最多振荡 grid_size 个周期
        x_norm = torch.tanh(x) * math.pi  # (bsz, in_features)
        
        # ========== 构造频率向量 ==========
        k = torch.arange(
            self.grid_size + 1, 
            device=x.device, 
            dtype=x.dtype
        )  # (k,)
        
        # ========== 【关键修复2】正确的维度扩展 ==========
        # 注意：不要多余的 unsqueeze(1)
        x_e = x_norm.unsqueeze(-1)  # (bsz, in_features, 1)
        k_e = k.view(1, 1, -1)      # (1, 1, k)
        
        # 计算角度 θ_{j,k} = k × x̃_j
        ang = x_e * k_e  # (bsz, in_features, k) - 正确的3维
        
        # 傅里叶基函数
        cos_kx = torch.cos(ang)  # (bsz, in_features, k)
        sin_kx = torch.sin(ang)  # (bsz, in_features, k)
        
        # ========== 傅里叶展开计算 ==========
        # 扩展维度以与参数广播
        cos_kx = cos_kx.unsqueeze(1)   # (bsz, 1, in_features, k)
        sin_kx = sin_kx.unsqueeze(1)   # (bsz, 1, in_features, k)
        a = self.a.unsqueeze(0)        # (1, width, in_features, k)
        b = self.b.unsqueeze(0)        # (1, width, in_features, k)
        
        # 广播验证：
        # (bsz, 1, in_features, k) × (1, width, in_features, k) 
        # → (bsz, width, in_features, k) ✓
        
        # φ_{i,j}(x_j) = Σ_k [a_{ijk}·cos(kx_j) + b_{ijk}·sin(kx_j)]
        phi = (a * cos_kx + b * sin_kx).sum(dim=-1)  # (bsz, width, in_features)
        
        # ========== 隐藏层聚合 ==========
        # s_i = Σ_j φ_{i,j}(x_j)
        s = phi.sum(dim=-1)  # (bsz, width)
        
        # 非线性激活
        y = self.act(s)  # (bsz, width)
        
        # ========== 输出层 ==========
        out = torch.matmul(y, self.W_out.t()) + self.bias_out  # (bsz, out_features)
        
        return out

class FourierEnergyKAN(nn.Module):
    """
    Fourier Energy Operator: 专门用于计算物理势能和环境阻抗
    
    Energy(z) = FourierKAN(z)
    
    特点:
    - 强制输出非负能量 (Softplus)
    - 可用于 U_I (内势) 和 R_E (阻抗) 的计算
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        grid_size: int = 5,
        width: Optional[int] = None,
        non_negative: bool = False,  # 是否强制非负 (用于阻抗 R_E)
    ):
        super().__init__()
        self.kan = FourierKAN(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            width=width
        )
        self.non_negative = non_negative
        if non_negative:
            self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = self.kan(x)
        if self.non_negative:
            energy = self.softplus(energy)
        return energy


# # ============ 完整训练示例 ============
# if __name__ == "__main__":
#     import torch.optim as optim
#     from torch.utils.data import DataLoader, TensorDataset
    
#     # 生成测试数据
#     torch.manual_seed(42)
#     n_samples = 1000
#     in_features = 10
    
#     X = torch.randn(n_samples, in_features)
#     # 非线性可分的二分类问题
#     y = ((X[:, 0] * X[:, 1] + X[:, 2]**2 - X[:, 3]) > 0).float().unsqueeze(-1)
    
#     # 划分训练/测试集
#     train_X, test_X = X[:800], X[800:]
#     train_y, test_y = y[:800], y[800:]
    
#     train_loader = DataLoader(
#         TensorDataset(train_X, train_y), 
#         batch_size=32, 
#         shuffle=True
#     )
    
#     # 创建模型
#     model = FourierKAN(
#         in_features=in_features,
#         out_features=1,
#         grid_size=5,
#         width=64
#     )
    
#     # 优化器与损失
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     criterion = nn.BCEWithLogitsLoss()
    
#     # 训练循环
#     for epoch in range(100):
#         model.train()
#         total_loss = 0
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             output = model(batch_X)
#             loss = criterion(output, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         # 评估
#         if (epoch + 1) % 10 == 0:
#             model.eval()
#             with torch.no_grad():
#                 train_pred = (torch.sigmoid(model(train_X)) > 0.5).float()
#                 test_pred = (torch.sigmoid(model(test_X)) > 0.5).float()
#                 train_acc = (train_pred == train_y).float().mean()
#                 test_acc = (test_pred == test_y).float().mean()
            
#             print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | "
#                   f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")