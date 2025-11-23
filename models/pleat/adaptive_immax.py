import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveIMMAXLoss(nn.Module):
    """
    自适应IMMAX损失函数，用于处理类别不平衡问题
    
    基于对偶优化理论，通过样本困难度动态调整正负样本的权重α
    使得模型能够在0.5阈值下正常判断正负样本
    """
    def __init__(self, alpha_init=0.5, alpha_momentum=0.9, eps=1e-8, margin_clip=10.0):
        """
        Args:
            alpha_init: α的初始值，默认0.5（平衡状态）
            alpha_momentum: α的指数移动平均系数，用于平滑更新，避免剧烈波动
            eps: 数值稳定性常数，防止除零
            margin_clip: 边际值裁剪阈值，防止数值溢出
        """
        super(AdaptiveIMMAXLoss, self).__init__()
        
        # 使用register_buffer使得α可以被保存但不被优化器更新
        self.register_buffer('alpha', torch.tensor(alpha_init))
        self.alpha_momentum = alpha_momentum
        self.eps = eps
        self.margin_clip = margin_clip
        
    def forward(self, pred, target, return_alpha=False):
        """
        前向传播计算损失
        
        Args:
            pred: 模型预测logits，形状 (batch, 1) 或 (batch,)，已经过sigmoid
            target: 真实标签，形状 (batch,)，取值0或1
            return_alpha: 是否返回当前α值
            
        Returns:
            loss: 标量损失值
            如果return_alpha=True，则返回 (loss, alpha)
        """
        # 确保维度一致
        if pred.dim() == 2:
            pred = pred.squeeze(1)  # (batch, 1) -> (batch,)
        
        # 确保target也是一维的
        if target.dim() == 2:
            target = target.squeeze(1)  # (batch, 1) -> (batch,)
        
        target = target.float()
        batch_size = pred.size(0)
        
        # 将sigmoid输出转换为logits（为了计算margin）
        # pred = sigmoid(h(x))，则 h(x) = logit(pred) = log(pred / (1 - pred))
        pred_clipped = torch.clamp(pred, self.eps, 1 - self.eps)
        logits = torch.log(pred_clipped / (1 - pred_clipped))
        
        # 计算margin: z_i = y_i * h(x_i)
        # 对于二分类：y_i ∈ {-1, +1}，所以需要将{0,1}转换为{-1,+1}
        y_signed = 2 * target - 1  # {0,1} -> {-1,+1}
        margin = y_signed * logits  # (batch,)
        margin = torch.clamp(margin, -self.margin_clip, self.margin_clip)
        
        # 分离正负样本索引
        pos_mask = target == 1
        neg_mask = target == 0
        
        # 处理边界情况：batch中全是一类样本
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        
        if n_pos == 0 or n_neg == 0:
            # 退化为标准BCE
            bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')
            if return_alpha:
                return bce_loss, self.alpha.item()
            return bce_loss
        
        # 使用当前α计算scaled margin
        alpha_current = self.alpha.item()
        
        # 计算Ψ函数及其导数
        # Ψ(u) = log(1 + exp(-u))，这是log-loss的形式
        # Ψ'(u) = -1 / (1 + exp(u))
        
        # 正样本: margin_scaled = margin / α
        margin_pos = margin[pos_mask] / (alpha_current + self.eps)
        psi_pos = torch.log1p(torch.exp(-margin_pos))  # 数值稳定版本
        psi_prime_pos = -1.0 / (1.0 + torch.exp(margin_pos))
        
        # 负样本: margin_scaled = margin / (1 - α)
        margin_neg = margin[neg_mask] / (1 - alpha_current + self.eps)
        psi_neg = torch.log1p(torch.exp(-margin_neg))
        psi_prime_neg = -1.0 / (1.0 + torch.exp(margin_neg))
        
        # 计算困难度指标 S+ 和 S-
        # S = Σ |z_i * Ψ'(z_i / α)|
        # 注意：margin已经是z_i，psi_prime是Ψ'
        grad_contrib_pos = torch.abs(margin[pos_mask] * psi_prime_pos)  # (n_pos,)
        grad_contrib_neg = torch.abs(margin[neg_mask] * psi_prime_neg)  # (n_neg,)
        
        S_pos = grad_contrib_pos.sum() + self.eps  # 加eps防止除零
        S_neg = grad_contrib_neg.sum() + self.eps
        
        # 计算新的α: α* = sqrt(S+) / (sqrt(S+) + sqrt(S-))
        sqrt_S_pos = torch.sqrt(S_pos)
        sqrt_S_neg = torch.sqrt(S_neg)
        alpha_new = sqrt_S_pos / (sqrt_S_pos + sqrt_S_neg)
        
        # 使用指数移动平均更新α，避免剧烈波动
        # 只在训练时更新
        if self.training:
            self.alpha.mul_(self.alpha_momentum).add_(alpha_new * (1 - self.alpha_momentum))
            # 限制α的范围在[0.01, 0.99]之间，防止极端值
            self.alpha.clamp_(0.01, 0.99)
        
        # 使用更新后的α重新计算损失（用于反向传播）
        # 注意：这里需要用detach后的alpha以避免二阶导数
        alpha_for_loss = self.alpha.detach()
        
        margin_pos_final = margin[pos_mask] / (alpha_for_loss + self.eps)
        margin_neg_final = margin[neg_mask] / (1 - alpha_for_loss + self.eps)
        
        loss_pos = torch.log1p(torch.exp(-margin_pos_final)).mean()
        loss_neg = torch.log1p(torch.exp(-margin_neg_final)).mean()
        
        # 总损失是正负样本损失的平均
        # 注意：这里不需要按样本数量加权，因为α已经包含了这个信息
        loss = (loss_pos + loss_neg) / 2.0
        
        if return_alpha:
            return loss, self.alpha.item()
        
        return loss
    
    def get_alpha(self):
        """获取当前α值"""
        return self.alpha.item()
    
    def reset_alpha(self, value=0.5):
        """重置α值"""
        self.alpha.fill_(value)