#!/usr/bin/env python3
"""
PRISM预训练脚本 - BERT风格的Masked Language Modeling
"""

from config import *
from config import PRISM_SAVE_MODEL_DIR, PRISM_BATCH_SIZE
from data_loader import load_prism_data, PRISMDataset, PRISMContrastiveSampler

import logging
from datetime import datetime
from torch.utils.data import DataLoader
from models.pleat.optimized_pre import create_optimized_dataset
from models.PRISMModel import PRISMModel, create_mlm_mask
from models.layers.footprint import LCWnetFootprint, FootprintConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


# 预训练投影配置类 - 定义高维投影任务的超参数
class PretrainProjConfig:
    MIN_SPEC_DIM = 8  # 特异子空间最小维度 d_spec
    SPEC_RATIO = 8    # 特异子空间与公共子空间的维度比例 d_com/d_spec
    ALPHA_INIT = 0.1  # 门控参数α的初始值，控制共同vs特异特征的占比
    LAMBDA_ALPHA = 1e-3  # α的正则化权重 λ_α，惩罚过多使用特异特征
    LAMBDA_INVAR = 1e-3  # 跨细胞公共一致性损失权重 λ_invar
    LAMBDA_VAR = 1e-3    # 公共空间方差约束损失权重 λ_var
    LAMBDA_CELL = 1e-2   # 特异空间细胞分类损失权重 λ_cell
    LAMBDA_SPEC = 1e-3   # 特异空间稀疏正则化权重 λ_spec
    LAMBDA_ORTHO = 1e-3  # 公共/特异子空间正交约束权重 λ_ortho
    EMA_BETA = 0.1       # 公共特征全局EMA均值更新系数β
    GAMMA = 1.0          # 期望的最小标准差阈值γ



device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


# 配置日志系统
def setup_logging():
    """配置日志系统"""
    log_filename = os.path.join(LOG_DIR, f"prism_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()
logger.info("PRISM预训练日志系统已初始化")
logger.info(f"日志文件: {LOG_DIR}")
logger.info(f"预处理线程数: {PREPROCESS_NUM_THREADS}")


def prism_collate_fn(batch):
    """
    PRISM特供collate函数
    batch中每个item: (enhancer_seq_str, promoter_seq_str, cell_line_str, label_int)
    """
    from models.pleat.embedding import KMerTokenizer
    
    enhancer_seqs = [item[0] for item in batch]
    promoter_seqs = [item[1] for item in batch]
    cell_lines = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
    # 创建tokenizer (只创建一次)
    if not hasattr(prism_collate_fn, 'tokenizer'):
        prism_collate_fn.tokenizer = KMerTokenizer()
    
    tokenizer = prism_collate_fn.tokenizer
    
    # 将DNA序列转换为token IDs
    enhancer_ids_list = [tokenizer.encode(seq) for seq in enhancer_seqs]
    promoter_ids_list = [tokenizer.encode(seq) for seq in promoter_seqs]
    
    # 填充序列
    padded_enhancer_ids = pad_sequence(enhancer_ids_list, batch_first=True, padding_value=0)
    padded_promoter_ids = pad_sequence(promoter_ids_list, batch_first=True, padding_value=0)
    
    # 确保最小长度
    if padded_enhancer_ids.size(1) < MAX_ENHANCER_LENGTH:
        padding_size = MAX_ENHANCER_LENGTH - padded_enhancer_ids.size(1)
        padded_enhancer_ids = torch.nn.functional.pad(
            padded_enhancer_ids, (0, padding_size), mode='constant', value=0
        )
    
    if padded_promoter_ids.size(1) < MAX_PROMOTER_LENGTH:
        padding_size = MAX_PROMOTER_LENGTH - padded_promoter_ids.size(1)
        padded_promoter_ids = torch.nn.functional.pad(
            padded_promoter_ids, (0, padding_size), mode='constant', value=0
        )
    
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    
    return padded_enhancer_ids, padded_promoter_ids, cell_lines, labels_tensor


def train_epoch(model, dataloader, optimizer, scheduler, epoch_idx, cell_label_map, ema_mu_com, proj_params, kb, loss_weights):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    train_pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{EPOCH} [Training]", 
                      leave=True, dynamic_ncols=True)
    
    for data in train_pbar:
        enhancer_ids, promoter_ids, cell_lines, labels = data
        enhancer_ids = enhancer_ids.to(device, non_blocking=True)
        promoter_ids = promoter_ids.to(device, non_blocking=True)
        
        # 创建MLM mask - BERT风格的掩码预测
        masked_enhancer_ids, mask_positions, original_enhancer_ids = create_mlm_mask(
            enhancer_ids,
            mask_prob=BERT_MASK_PROB,  # 掩码概率，平衡预测难度与信息保留
            mask_token_id=BERT_MASK_TOKEN_ID,
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            pad_token_id=BERT_PAD_TOKEN_ID,
            block_mask=True,  # 块级掩码，提高预测难度
            block_size=KMER_SIZE
        )
        
        # 前向传播
        mlm_logits, enhancer_final, _, seq_length_mapping, fused_footprint = model(masked_enhancer_ids, promoter_ids, mask_positions)

        v = fused_footprint
        z_com = model.current_z_com
        z_spec = model.current_z_spec

        B = enhancer_ids.size(0)  # batch大小
        V = mlm_logits.size(-1)   # 词汇表大小
        D_com = z_com.size(-1)    # 公共子空间维度 d_com

        unique_cells = []
        for c in cell_lines:
            if c not in unique_cells:
                unique_cells.append(c)

        g_com_list = []  # 收集细胞级公共特征 g^com_c
        g_spec_list = []  # 收集细胞级特异特征 g^spec_c
        cell_label_list = []  # 收集细胞标签用于分类损失
        for c in unique_cells:
            idx = [i for i in range(B) if cell_lines[i] == c]  # 找到属于细胞c的所有样本索引
            if len(idx) == 0:
                continue
            idx_t = torch.tensor(idx, device=device)
            g_c_com = z_com.index_select(0, idx_t).mean(dim=0)  # g^com_c = (1/K)∑z^com_{c,i} - 细胞级公共特征均值
            g_c_spec = z_spec.index_select(0, idx_t).mean(dim=0)  # g^spec_c = (1/K)∑z^spec_{c,i} - 细胞级特异特征均值
            g_com_list.append(g_c_com)
            g_spec_list.append(g_c_spec)
            cell_label_list.append(cell_label_map.get(c, 0))  # 映射细胞名称到标签索引

        kb_beta = proj_params['ema_beta']  # EMA更新系数β，用于知识库中心点平滑更新
        if isinstance(kb, dict):
            kb.setdefault('spec_centers', {})  # 初始化特异特征中心点缓存
            kb.setdefault('com_centers', {})   # 初始化公共特征中心点缓存
            
            # 确保所有需要的键都在
            if 'model_state' not in kb:
                kb['model_state'] = None
                
            for i, c in enumerate(unique_cells):
                g_c_spec = g_spec_list[i]
                g_c_com = g_com_list[i]
                if c in kb['spec_centers']:
                    kb['spec_centers'][c] = kb['spec_centers'][c] * (1 - kb_beta) + g_c_spec.detach().cpu() * kb_beta  # 指数移动平均
                else:
                    kb['spec_centers'][c] = g_c_spec.detach().cpu()
                if c in kb['com_centers']:
                    kb['com_centers'][c] = kb['com_centers'][c] * (1 - kb_beta) + g_c_com.detach().cpu() * kb_beta  # 指数移动平均
                else:
                    kb['com_centers'][c] = g_c_com.detach().cpu()
        
        
        
        bert_loss, accuracy = model.compute_mlm_loss(mlm_logits, original_enhancer_ids, mask_positions, seq_length_mapping)  # L_BERT - BERT掩码预测损失

        g_com_stack = torch.stack(g_com_list) if len(g_com_list) > 0 else torch.zeros(1, D_com, device=device, dtype=v.dtype)  # [C, d_com] - 细胞级公共特征矩阵
        g_spec_stack = torch.stack(g_spec_list) if len(g_spec_list) > 0 else torch.zeros(1, z_spec.size(-1), device=device, dtype=v.dtype)  # [C, d_spec] - 细胞级特异特征矩阵

        loss_invar = proj_params['lambda_invar'] * ((g_com_stack - ema_mu_com).pow(2).mean())  # L_invar = λ_invar * ||g^com_c - μ_com||^2 - 跨细胞公共一致性损失
        ema_mu_com.mul_(1 - proj_params['ema_beta']).add_(g_com_stack.detach().mean(dim=0) * proj_params['ema_beta'])  # μ_com ← (1-β)μ_com + β·mean(g^com_c) - EMA更新全局均值

        zc_center = z_com - z_com.mean(dim=0)  # 中心化处理，计算协方差矩阵
        var = zc_center.pow(2).mean(dim=0)     # 各维度方差 (d_com维度的方差向量)
        loss_var = proj_params['lambda_var'] * ((proj_params['gamma'] - torch.sqrt(var + 1e-8)).clamp(min=0).pow(2).sum())  # L_var = λ_var * Σ max(0, γ-√Var)^2 - 公共空间维度方差约束，确保每个维度都有信息
        normed = zc_center / (torch.sqrt(var + 1e-8))  # 标准化处理，用于计算协方差
        cov = (normed.T @ normed) / max(B, 1)          # 协方差矩阵 (d_com × d_com)
        off = cov - torch.diag(torch.diag(cov))        # 去掉对角线元素，得到非对角协方差
        loss_cov = proj_params['lambda_var'] * (off.pow(2).sum())  # L_cov = λ_var * ||off-diagonal(C_com)||_F^2 - 公共空间去相关约束

        if g_spec_stack.size(0) > 0:
            centers = g_spec_stack  # 特异特征中心点（细胞级均值）
            centers_norm = F.normalize(centers, dim=-1)  # 归一化中心点，用于对比学习
            z_spec_norm = F.normalize(z_spec, dim=-1)    # 归一化样本级特异特征
            logits = z_spec_norm @ centers_norm.T        # 计算相似度分数矩阵 [B, C]
            logits = logits / max(loss_weights.get('tau', 0.07), 1e-6)  # 温度参数τ，控制分布锐度
            local_labels = torch.tensor([unique_cells.index(cell_lines[i]) for i in range(B)], device=device)  # 样本对应的细胞索引标签
            loss_cell = F.cross_entropy(logits, local_labels)  # L_cell - 特异空间细胞分类/对比损失
        else:
            loss_cell = torch.tensor(0.0, device=device)

        loss_spec_reg = proj_params['lambda_spec'] * (g_spec_stack.abs().mean() + g_spec_stack.pow(2).mean())  # L_spec_reg = λ_spec * (||g^spec_c||_1 + ||g^spec_c||_2^2) - 特异空间稀疏正则化，鼓励使用少量维度

        g_spec_map = {}  # 构建细胞到特异特征中心的映射
        for i, c in enumerate(unique_cells):
            g_spec_map[c] = g_spec_list[i]
        g_per_sample = torch.stack([g_spec_map.get(cell_lines[i], torch.zeros_like(z_spec[0])) for i in range(B)]).to(device)  # [B, d_spec] - 每个样本对应的细胞中心
        loss_within = proj_params['lambda_spec'] * ((z_spec - g_per_sample).pow(2).mean())  # L_within = λ_spec * ||z^spec_{c,i} - g^spec_c||^2 - 细胞内一致性约束，防止噪声主导

        w_ortho = model.enhancer_footprint.w_com.weight @ model.enhancer_footprint.w_spec.weight.t()
        loss_ortho = proj_params['lambda_ortho'] * (w_ortho.pow(2).sum())  # L_ortho = λ_ortho * ||W_com · W_spec^T||_F^2 - 子空间正交约束，避免特征污染

        total_loss_batch = loss_weights.get('w_cell', 1.0) * loss_cell + loss_weights.get('w_mlm', 0.1) * bert_loss + loss_invar + loss_var + loss_cov + loss_spec_reg + loss_within + loss_ortho  # 总损失: L = w_cell*L_cell + w_mlm*L_BERT + L_invar + L_var + L_cov + L_spec_reg + L_within + L_ortho
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), BERT_MAX_GRAD_NORM)
        
        optimizer.step()
        
        # 统计
        total_loss += total_loss_batch.item()
        total_accuracy += accuracy
        num_batches += 1
        
        # 更新进度条
        train_pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate(model, dataloader, cell_name="", cell_label_map=None, loss_weights=None):
    """验证函数"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    val_pbar = tqdm(dataloader, desc=f"Validation [{cell_name}]", 
                   leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for data in val_pbar:
            enhancer_ids, promoter_ids, cell_lines, labels = data
            enhancer_ids = enhancer_ids.to(device, non_blocking=True)
            promoter_ids = promoter_ids.to(device, non_blocking=True)
            
            # 创建MLM mask
            masked_enhancer_ids, mask_positions, original_enhancer_ids = create_mlm_mask(
                enhancer_ids,
                mask_prob=BERT_MASK_PROB,
                mask_token_id=BERT_MASK_TOKEN_ID,
                vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
                pad_token_id=BERT_PAD_TOKEN_ID,
                block_mask=True,
                block_size=KMER_SIZE
            )
            
            # 前向传播
            mlm_logits, enhancer_final, _, seq_length_mapping, fused_footprint = model(masked_enhancer_ids, promoter_ids, mask_positions)
            
            # 计算损失
            loss, accuracy = model.compute_mlm_loss(mlm_logits, original_enhancer_ids, mask_positions, seq_length_mapping)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate_separability(model, dataloader):
    model.eval()
    with torch.no_grad():
        spec_vectors = {}
        com_vectors = {}
        counts = {}
        for data in dataloader:
            enhancer_ids, promoter_ids, cell_lines, labels = data
            enhancer_ids = enhancer_ids.to(device, non_blocking=True)
            promoter_ids = promoter_ids.to(device, non_blocking=True)
            masked_enhancer_ids, mask_positions, _ = create_mlm_mask(
                enhancer_ids,
                mask_prob=BERT_MASK_PROB,
                mask_token_id=BERT_MASK_TOKEN_ID,
                vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
                pad_token_id=BERT_PAD_TOKEN_ID,
                block_mask=True,
                block_size=KMER_SIZE,
            )
            _logits, _en_final, _pr_final, _map, _fused = model(masked_enhancer_ids, promoter_ids, mask_positions)
            z_com = model.current_z_com
            z_spec = model.current_z_spec
            for i, c in enumerate(cell_lines):
                if c not in spec_vectors:
                    spec_vectors[c] = []
                    com_vectors[c] = []
                    counts[c] = 0
                spec_vectors[c].append(z_spec[i].detach())
                com_vectors[c].append(z_com[i].detach())
                counts[c] += 1
        spec_centers = {}
        com_centers = {}
        for c in spec_vectors:
            spec_centers[c] = torch.stack(spec_vectors[c], dim=0).mean(dim=0)
            com_centers[c] = torch.stack(com_vectors[c], dim=0).mean(dim=0)
        intra_spec = []
        for c in spec_vectors:
            center = spec_centers[c]
            vecs = torch.stack(spec_vectors[c], dim=0)
            intra_spec.append(((vecs - center).pow(2).sum(dim=-1)).mean())
        intra_spec_mean = torch.stack(intra_spec).mean() if intra_spec else torch.tensor(0.0)
        centers_list = list(spec_centers.values())
        inter = []
        for i in range(len(centers_list)):
            for j in range(i + 1, len(centers_list)):
                d = (centers_list[i] - centers_list[j]).pow(2).sum().sqrt()
                inter.append(d)
        inter_mean = torch.stack(inter).mean() if inter else torch.tensor(0.0)
        fisher_ratio = (inter_mean / (intra_spec_mean + 1e-8)).item()
        com_center_vals = list(com_centers.values())
        if com_center_vals:
            mu_com = torch.stack(com_center_vals, dim=0).mean(dim=0)
            com_disp = torch.stack([(c - mu_com).pow(2).sum().sqrt() for c in com_center_vals]).mean().item()
        else:
            com_disp = 0.0
    return {
        'spec_fisher_ratio': fisher_ratio,
        'com_center_dispersion': com_disp,
        'num_cells': len(spec_centers),
    }


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("PRISM预训练开始 (Domain-KL数据)")
    logger.info("=" * 80)
    
    # 加载PRISM特供数据
    logger.info("加载训练数据 (domain-kl)...")
    train_pairs_df, train_e_seqs, train_p_seqs = load_prism_data("train")
    logger.info(f"训练样本数: {len(train_pairs_df)}")
    logger.info(f"训练细胞系: {', '.join(sorted(train_pairs_df['cell_line'].unique()))}")
    
    logger.info("加载验证数据 (domain-kl)...")
    val_pairs_df, val_e_seqs, val_p_seqs = load_prism_data("val")
    logger.info(f"验证样本数: {len(val_pairs_df)}")
    logger.info(f"验证细胞系: {', '.join(sorted(val_pairs_df['cell_line'].unique()))}")
    
    # 创建数据集
    train_dataset = PRISMDataset(train_pairs_df, train_e_seqs, train_p_seqs)
    val_dataset = PRISMDataset(val_pairs_df, val_e_seqs, val_p_seqs)
    
    # 创建对比采样器
    train_sampler = PRISMContrastiveSampler(
        train_dataset, 
        batch_size=PRISM_BATCH_SIZE, 
        shuffle=True
    )
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=prism_collate_fn,
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=PRISM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=prism_collate_fn,
    )
    
    # 创建模型
    logger.info("创建PRISM模型...")
    model = PRISMModel()
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"GPU可用: {torch.cuda.is_available()}")
    logger.info(f"模型在GPU上: {next(model.parameters()).is_cuda}")
    
    # 创建优化器和调度器
    unique_cells_train = sorted(train_pairs_df['cell_line'].unique())
    cell_label_map = {c: i for i, c in enumerate(unique_cells_train)}

    ema_mu_com = torch.zeros(OUT_CHANNELS, device=device)
    proj_params = {
        'lambda_alpha': PretrainProjConfig.LAMBDA_ALPHA,
        'lambda_invar': PretrainProjConfig.LAMBDA_INVAR,
        'lambda_var': PretrainProjConfig.LAMBDA_VAR,
        'lambda_cell': PretrainProjConfig.LAMBDA_CELL,
        'lambda_spec': PretrainProjConfig.LAMBDA_SPEC,
        'lambda_ortho': PretrainProjConfig.LAMBDA_ORTHO,
        'ema_beta': PretrainProjConfig.EMA_BETA,
        'gamma': PretrainProjConfig.GAMMA,
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BERT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCH
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        threshold=1e-3,
        threshold_mode='rel',
        min_lr=1e-6
    )
    
    logger.info(f"批量大小: {PRISM_BATCH_SIZE} (对比采样: {PRISM_BATCH_SIZE//2}同细胞系 + {PRISM_BATCH_SIZE//2}不同细胞系)")
    logger.info(f"训练轮数: {EPOCH}")
    logger.info(f"学习率: {BERT_LEARNING_RATE}")
    logger.info(f"总训练步数: {total_steps}")
    
    # 训练循环
    logger.info("=" * 80)
    logger.info("开始训练")
    logger.info("=" * 80)
    
    kb = {}
    for epoch_idx in range(EPOCH):
        # 训练
        loss_weights = {'w_cell': 1.0, 'w_mlm': 0.1, 'tau': 0.07}
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch_idx,
            W_com,
            W_spec,
            cell_label_map,
            ema_mu_com,
            proj_params,
            kb,
            loss_weights,
        )
        
        logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_epoch_{epoch_idx+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")

        # 保存知识库 (包含中心点和模型状态)
        kb_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"footprint_kb_epoch_{epoch_idx+1}.pt")
        
        # 更新KB中的模型状态，方便直接加载专家
        kb['model_state'] = model.state_dict()
        
        try:
            torch.save(kb, kb_path)
            logger.info(f"保存知识库: {kb_path}")
        except Exception as e:
            logger.error(f"保存知识库失败: {e}")
        
        # 验证
        if epoch_idx % VALIDATION_INTERVAL == 0 or epoch_idx == EPOCH - 1:
            val_loss, val_acc = validate(
                model,
                val_loader,
                "ALL",
                cell_label_map,
                loss_weights,
            )
            sep_metrics = validate_separability(model, val_loader)
            logger.info(f"Separability: fisher={sep_metrics['spec_fisher_ratio']:.4f}, com_disp={sep_metrics['com_center_dispersion']:.4f}, cells={sep_metrics['num_cells']}")
            logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            scheduler.step(val_loss)
            logger.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # # 保存最佳模型
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     save_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_best.pth")
            #     torch.save(model.state_dict(), save_path)
            #     logger.info(f"保存最佳模型: {save_path}")
    
    logger.info("=" * 80)
    logger.info("PRISM预训练完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

