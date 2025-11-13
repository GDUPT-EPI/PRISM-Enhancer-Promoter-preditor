import torch
from torch import nn

# Stochastic Depth
class StochasticDepth(nn.Module):
    """线性递增的随机深度"""
    def __init__(self, drop_prob, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 保持batch维度
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        
        if self.scale_by_keep and keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        
        return x * random_tensor
