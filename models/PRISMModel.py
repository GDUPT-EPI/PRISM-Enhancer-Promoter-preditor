"""
PRISM组件

主干网络(`PRISMBackbone`)用于EP互作概率预测
"""

from torch import nn
import torch
import torch.nn.functional as F
from config import *
from models.pleat.embedding import create_dna_embedding_layer
from models.pleat.RoPE import RoPEConfig
from models.layers.attn import *
from models.layers.FourierKAN import FourierKAN
from models.layers.ISAB import ISAB
from models.pleat.adaptive_immax import AdaptiveIMMAXLoss
from models.pleat.SpeculationPenalty import SpeculationPenaltyLoss
from typing import Optional, Tuple


class CBATTransformerEncoderLayer(nn.Module):  # 定义CBAT Transformer编码器层类
    """
    使用CBAT模块的Transformer编码器层
    替换标准的多头自注意力为CBAT注意力机制
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, img_size=32):  # 初始化函数
        super(CBATTransformerEncoderLayer, self).__init__()  # 调用父类初始化
        self.self_attn = CBAT(  # 创建CBAT注意力模块
            d_model=d_model,  # 模型维度
            num_heads=nhead,  # 注意力头数
            img_size=img_size,  # CBAT图像大小
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,  # RoPE最大序列长度
            dropout=dropout  # Dropout比例
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 前馈层1
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 前馈层2
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化1
        self.norm2 = nn.LayerNorm(d_model)  # 层归一化2
        self.dropout1 = nn.Dropout(dropout)  # 残差Dropout1
        self.dropout2 = nn.Dropout(dropout)  # 残差Dropout2
        self.activation = nn.ReLU()  # 激活函数
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):  # 编码器层前向
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (seq_len, batch, d_model)
            src_mask: 注意力掩码
            src_key_padding_mask: key填充掩码
            
        Returns:
            返回 (输出张量, adaptive_loss)
            输出张量形状 (seq_len, batch, d_model)
        """
        # 保存原始输入用于残差连接
        residual = x  # 残差1
        
        # 自注意力计算 (使用CBAT)
        # 转换维度以适配CBAT: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        x_transposed = x.permute(1, 0, 2)  # 转换到(batch, seq, d)
        
        # CBAT返回(output, adaptive_loss)
        attn_output, adaptive_loss = self.self_attn(x_transposed, attention_mask=src_mask, return_loss=True)  # CBAT输出
            
        # 转回原始维度: (batch, seq_len, d_model) -> (seq_len, batch, d_model)
        attn_output = attn_output.permute(1, 0, 2)  # 转回(seq, batch, d)
        
        # 残差连接和层归一化
        x = residual + self.dropout1(attn_output)  # 残差加权
        x = self.norm1(x)  # 归一化1
        
        # 保存用于第二个残差连接
        residual = x  # 残差2
        
        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))  # 前馈输出
        
        # 残差连接和层归一化
        x = residual + self.dropout2(ff_output)  # 残差加权
        x = self.norm2(x)  # 归一化2
        
        return x, adaptive_loss

class AttnPool1d(nn.Module):  # 定义1D注意力池化类
    def __init__(self, d: int):  # 1D注意力池化初始化
        super().__init__()  # 调用父类初始化
        self.proj = nn.Linear(d, d)  # 投影层
        self.v = nn.Parameter(torch.zeros(d))  # 注意力向量
        nn.init.normal_(self.v, mean=0.0, std=0.02)  # 初始化
        self.drop = nn.Dropout(CNN_DROPOUT)  # Dropout
    def forward(self, x: torch.Tensor, mask: torch.Tensor):  # 前向传播
        h = torch.tanh(self.proj(self.drop(x)))  # 非线性投影
        s = (h * self.v).sum(-1)  # 注意力分数
        s = s.masked_fill(mask, -1e9)  # 掩码填充
        w = torch.softmax(s, dim=-1)  # 权重
        w = w.masked_fill(mask, 0.0)  # 掩码权重置零
        norm = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # 归一化因子
        w = w / norm  # 归一化
        return (x * w.unsqueeze(-1)).sum(dim=1)  # 加权求和

class AttnPool1dWindow(nn.Module):  # 定义滑窗注意力池化类
    def __init__(self, d: int, kernel_size: int, stride: int):  # 滑窗注意力池化初始化
        super().__init__()  # 调用父类初始化
        self.kernel_size = kernel_size  # 窗长
        self.stride = stride  # 步长
        self.attn = AttnPool1d(d)  # 注意力池化
    def forward(self, x: torch.Tensor, mask: torch.Tensor):  # 前向传播
        B, C, L = x.shape  # 维度
        step = self.stride  # 步长
        win = self.kernel_size  # 窗长
        if L < win:
            return x.new_zeros(B, C, 0)  # 长度不足返回空
        L_pool = 1 + (L - win) // step  # 片段数
        outs = []  # 输出列表
        for j in range(L_pool):  # 遍历窗口
            start = j * step  # 起点
            end = start + win  # 终点
            xw = x[:, :, start:end].permute(0, 2, 1)  # 片段
            mw = mask[:, start:end]  # 掩码片段
            ow = self.attn(xw, mw)  # 注意力池化
            outs.append(ow.unsqueeze(-1))  # 追加
        if len(outs) == 0:
            return x.new_zeros(B, C, 0)  # 无输出返回空
        return torch.cat(outs, dim=-1)  # 拼接

# 统一由PRISMBackbone组成

class PRISMBackbone(nn.Module):  # 定义PRISM主干网络类
    """PRISM主干网络
    
    编码E/P序列、建模跨序列注意力，输出互作概率。
    
    Args:
        num_classes: 预留参数，当前未使用。
    """
    def __init__(self, num_classes: int = None):  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.num_transformer_layers = TRANSFORMER_LAYERS  # Transformer层数
        TRANSFORMER_DROPOUT = RoPEConfig.ROPE_DROPOUT  # Transformer Dropout
        img_size = PRISM_IMG_SIZE  # PRISM图像大小

        self.enh_embedding = create_dna_embedding_layer(  # 创建增强子嵌入层
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,  # 词汇表大小
            embed_dim=DNA_EMBEDDING_DIM,  # 嵌入维度
            padding_idx=DNA_EMBEDDING_PADDING_IDX,  # 填充索引
            init_std=DNA_EMBEDDING_INIT_STD  # 初始化标准差
        )
        self.pr_embedding = create_dna_embedding_layer(  # 创建启动子嵌入层
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,  # 词汇表大小
            embed_dim=DNA_EMBEDDING_DIM,  # 嵌入维度
            padding_idx=DNA_EMBEDDING_PADDING_IDX,  # 填充索引
            init_std=DNA_EMBEDDING_INIT_STD  # 初始化标准差
        )
        self.enh_cnn = nn.Sequential(  # 增强子CNN序列
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),  # 最大池化
            nn.BatchNorm1d(OUT_CHANNELS),  # 批归一化
            nn.Dropout(p=CNN_DROPOUT)  # Dropout
        )
        self.pr_cnn = nn.Sequential(  # 启动子CNN序列
            nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),  # 最大池化
            nn.BatchNorm1d(OUT_CHANNELS),  # 批归一化
            nn.Dropout(p=CNN_DROPOUT)  # Dropout
        )
        self.pre_enh_attn = RoPEAttention(  # 增强子预注意力
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )  # 增强子预注意力
        self.pre_pr_attn = RoPEAttention(  # 启动子预注意力
            d_model=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, dropout=TRANSFORMER_DROPOUT
        )  # 启动子预注意力
        self.cross_attn_1 = nn.MultiheadAttention(  # 第一次跨序列注意力
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )  # 第一次跨序列注意力
        self.cbat_layers = nn.ModuleList([  # 增强子CBAT层列表
            CBATTransformerEncoderLayer(  # CBAT编码器层
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,  # 模型维度和头数
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,  # 前馈维度和dropout
                img_size=img_size  # 图像大小
            ) for _ in range(self.num_transformer_layers)  # 创建多层
        ])  # 增强子CBAT层
        self.pr_cbat_layers = nn.ModuleList([  # 启动子CBAT层列表
            CBATTransformerEncoderLayer(  # CBAT编码器层
                d_model=OUT_CHANNELS, nhead=TRANSFORMER_HEADS,  # 模型维度和头数
                dim_feedforward=TRANSFORMER_FF_DIM, dropout=TRANSFORMER_DROPOUT,  # 前馈维度和dropout
                img_size=img_size  # 图像大小
            ) for _ in range(self.num_transformer_layers)  # 创建多层
        ])  # 启动子CBAT层
        self.cross_attn_2 = nn.MultiheadAttention(  # 第二次跨序列注意力
            embed_dim=OUT_CHANNELS, num_heads=TRANSFORMER_HEADS, batch_first=False
        )  # 第二次跨序列注意力
        self.post_cbat = CBAT(  # 末端CBAT
            d_model=OUT_CHANNELS,  # 模型维度
            num_heads=TRANSFORMER_HEADS,  # 注意力头数
            img_size=img_size,  # 图像大小
            max_seq_len=RoPEConfig.ROPE_MAX_SEQ_LEN,  # 最大序列长度
            dropout=TRANSFORMER_DROPOUT,  # Dropout
        )  # 末端CBAT
        self.isab = ISAB(  # ISAB上下文层（批级集合）
            d_model=OUT_CHANNELS,
            num_heads=TRANSFORMER_HEADS,
            num_inducing=ISAB_NUM_INDUCING,
            dropout=ISAB_DROPOUT,
        )
        self.attn_pool_en = AttnPool1d(OUT_CHANNELS)  # 注意力池化
        self.classifier = FourierKAN(  # KAN分类头
            in_features=OUT_CHANNELS,  # 输入特征数
            out_features=1,  # 输出特征数
            grid_size=5,  # 网格大小
            width=2 * OUT_CHANNELS + 1,  # 宽度
        )  # KAN分类头
        self.criterion = AdaptiveIMMAXLoss()  # 基础损失
        self.spec_penalty = SpeculationPenaltyLoss()  # 投机惩罚
        self.optimizer = torch.optim.AdamW(  # 优化器
            self.parameters(),  # 参数
            lr=LEARNING_RATE,  # 学习率
            weight_decay=WEIGHT_DECAY,  # 权重衰减
        )


    def forward(  # 前向传播
        self,
        enhancer_ids: torch.Tensor,  # 增强子ID
        promoter_ids: torch.Tensor,  # 启动子ID
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回类型
        """前向传播

        编码E/P序列、进行跨序列交互与CBAT增强，输出互作概率。

        Args:
            enhancer_ids: 增强子序列ID，形状 `[B, L_en]`
            promoter_ids: 启动子序列ID，形状 `[B, L_pr]`

        Returns:
            (prob, adaptive_loss): 概率与自适应注意力损失
        """
        K = CNN_KERNEL_SIZE  # 卷积核大小
        P = POOL_KERNEL_SIZE  # 池化核大小
        min_required_length = K + P - 1  # 最小所需长度
        if enhancer_ids.size(1) < min_required_length:  # 检查增强子长度
            enhancer_ids = F.pad(enhancer_ids, (0, min_required_length - enhancer_ids.size(1)), value=DNA_EMBEDDING_PADDING_IDX)  # 填充
        if promoter_ids.size(1) < min_required_length:  # 检查启动子长度
            promoter_ids = F.pad(promoter_ids, (0, min_required_length - promoter_ids.size(1)), value=DNA_EMBEDDING_PADDING_IDX)  # 填充

        embed_en = self.enh_embedding(enhancer_ids)  # 增强子嵌入
        embed_pr = self.pr_embedding(promoter_ids)  # 启动子嵌入
        enh = self.enh_cnn(embed_en.permute(0, 2, 1))  # 增强子CNN
        pr = self.pr_cnn(embed_pr.permute(0, 2, 1))  # 启动子CNN
        enh = enh.permute(2, 0, 1)  # 转置维度
        pr = pr.permute(2, 0, 1)  # 转置维度

        B_en = enhancer_ids.size(0)  # 增强子批次大小
        L_en_orig = enhancer_ids.size(1)  # 增强子原始长度
        L_en = enh.size(0)  # 增强子处理后长度
        pad_en = (enhancer_ids == DNA_EMBEDDING_PADDING_IDX)  # 增强子填充掩码
        enh_pad_mask = torch.zeros(B_en, L_en, dtype=torch.bool, device=enhancer_ids.device)  # 增强子填充掩码张量
        for j in range(L_en):  # 遍历长度
            s = j * POOL_KERNEL_SIZE  # 起点
            e = min(s + POOL_KERNEL_SIZE + CNN_KERNEL_SIZE - 2, L_en_orig - 1)  # 终点
            enh_pad_mask[:, j] = pad_en[:, s:e+1].all(dim=-1)  # 设置掩码
        enh_attn_mask = torch.zeros(B_en, 1, L_en, L_en, device=enhancer_ids.device, dtype=torch.float32)  # 增强子注意力掩码
        if enh_pad_mask.any():  # 如果有填充
            mask_cols = enh_pad_mask  # 掩码列
            for b in range(B_en):  # 遍历批次
                cols = mask_cols[b]  # 获取列
                if cols.any():  # 如果有掩码
                    enh_attn_mask[b, 0, :, cols] = float('-inf')  # 设置负无穷
        enh_pre = self.pre_enh_attn(enh.permute(1, 0, 2), attention_mask=enh_attn_mask)  # 增强子预注意力
        enh = enh_pre.permute(1, 0, 2)  # 转置维度
        B_pr = promoter_ids.size(0)  # 启动子批次大小
        L_pr_orig = promoter_ids.size(1)  # 启动子原始长度
        L_pr = pr.size(0)  # 启动子处理后长度
        pad_pr = (promoter_ids == DNA_EMBEDDING_PADDING_IDX)  # 启动子填充掩码
        pr_pad_mask = torch.zeros(B_pr, L_pr, dtype=torch.bool, device=promoter_ids.device)  # 启动子填充掩码张量
        for j in range(L_pr):  # 遍历长度
            s = j * POOL_KERNEL_SIZE  # 起点
            e = min(s + POOL_KERNEL_SIZE + CNN_KERNEL_SIZE - 2, L_pr_orig - 1)  # 终点
            pr_pad_mask[:, j] = pad_pr[:, s:e+1].all(dim=-1)  # 设置掩码
        pr_attn_mask = torch.zeros(B_pr, 1, L_pr, L_pr, device=promoter_ids.device, dtype=torch.float32)  # 启动子注意力掩码
        if pr_pad_mask.any():  # 如果有填充
            for b in range(B_pr):  # 遍历批次
                cols = pr_pad_mask[b]  # 获取列
                if cols.any():  # 如果有掩码
                    pr_attn_mask[b, 0, :, cols] = float('-inf')  # 设置负无穷
        pr_pre = self.pre_pr_attn(pr.permute(1, 0, 2), attention_mask=pr_attn_mask)  # 启动子预注意力
        pr = pr_pre.permute(1, 0, 2)  # 转置维度

        att1, _ = self.cross_attn_1(enh, pr, pr, key_padding_mask=pr_pad_mask)  # 第一次跨注意力
        enh = enh + att1  # 残差连接


        total_adaptive_loss = 0.0  # 总自适应损失
        for layer in self.cbat_layers:  # 遍历CBAT层
            enh, layer_loss = layer(enh, src_mask=enh_attn_mask)  # 前向传播
            total_adaptive_loss += layer_loss  # 累加损失
        for layer in self.pr_cbat_layers:  # 遍历启动子CBAT层
            pr, layer_loss_pr = layer(pr, src_mask=pr_attn_mask)  # 前向传播
            total_adaptive_loss += layer_loss_pr  # 累加损失

        att2, _ = self.cross_attn_2(enh, pr, pr, key_padding_mask=pr_pad_mask)  # 第二次跨注意力
        enh = enh + att2  # 残差连接

        post_out, post_loss = self.post_cbat(enh.permute(1, 0, 2), return_loss=True)
        total_adaptive_loss += post_loss
        enh = post_out.permute(1, 0, 2)

        x_seq = enh.permute(1, 0, 2)
        pooled = self.attn_pool_en(x_seq, enh_pad_mask)
        pooled = F.layer_norm(pooled, pooled.shape[-1:])

        x_set = pooled.unsqueeze(0)
        y_set = self.isab(x_set)
        y = y_set.squeeze(0)
        result = self.classifier(y)
        return torch.sigmoid(result), total_adaptive_loss  # 返回sigmoid结果和损失

    def compute_loss(  # 计算损失
        self,
        outputs: torch.Tensor,  # 输出
        labels: torch.Tensor,  # 标签
        adaptive_loss: torch.Tensor | float = 0.0,  # 自适应损失
        return_details: bool = False,  # 是否返回细节
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:  # 返回类型
        """计算总损失

        组成：`AdaptiveIMMAX`基础损失 + 自适应注意力损失 + 投机惩罚损失。

        Args:
            outputs: 预测概率 `[B]` 或 `[B,1]`
            labels: 二分类标签 `[B]`
            adaptive_loss: 自适应注意力损失
            return_details: 返回损失细节

        Returns:
            总损失或(损失, 细节)
        """
        base_loss = self.criterion(outputs, labels)  # 基础损失
        penalty_loss = self.spec_penalty(outputs, labels)  # 惩罚损失
        total = base_loss + adaptive_loss + 0.1 * penalty_loss  # 总损失
        if return_details:  # 如果需要返回细节
            return total, {  # 返回总损失和细节
                'base': float(base_loss.detach().item()),  # 基础损失
                'adaptive': float((adaptive_loss.detach().item() if isinstance(adaptive_loss, torch.Tensor) else adaptive_loss)),  # 自适应损失
                'penalty': float(penalty_loss.detach().item()),  # 惩罚损失
            }
        return total
