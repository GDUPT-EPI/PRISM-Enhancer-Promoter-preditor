import torch
import torch.nn as nn

class SpeculationPenaltyLoss(nn.Module):
    """
    显式投机惩罚损失，用于抑制模型在类别不平衡时的“全预测为正类”的投机行为。

    组成部分：
    - 负类高置信度惩罚：当 `y=0` 且 `p` 高于阈值时进行平滑惩罚
    - 先验匹配惩罚：约束批次预测均值接近标签均值，避免整体偏置

    Args:
        fp_weight (float): 负类投机惩罚权重
        threshold (float): 惩罚触发阈值（通常为0.5）
        slope (float): 平滑惩罚的斜率（越大越接近硬阈值）
        prior_weight (float): 先验匹配项权重

    Returns:
        torch.Tensor: 标量惩罚损失
    """
    def __init__(
        self,
        fp_weight: float = 0.85,
        threshold: float = 0.5,
        slope: float = 6.0,
        prior_weight: float = 0.06,
    ):
        super().__init__()
        self.fp_weight = fp_weight
        self.threshold = threshold
        self.slope = slope
        self.prior_weight = prior_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 概率预测 `[B]` 或 `[B,1]`（已sigmoid）
            target: 二分类标签 `[B]`，取值 {0,1}

        Returns:
            标量惩罚损失
        """
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        target = target.float()

        neg_mask = (target == 0)
        if neg_mask.any():
            p_neg = pred[neg_mask]
            margin = p_neg - self.threshold
            fp_term = torch.nn.functional.relu(self.slope * margin).pow(2).mean()
        else:
            fp_term = pred.new_tensor(0.0)

        mean_pred = pred.mean()
        mean_target = target.mean()
        prior_term = (mean_pred - mean_target).pow(2)

        return self.fp_weight * fp_term + self.prior_weight * prior_term
