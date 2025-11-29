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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


# 配置类
class PretrainProjConfig:
    MIN_SPEC_DIM = 8
    SPEC_RATIO = 8
    ALPHA_INIT = 0.1
    LAMBDA_ALPHA = 1e-3
    LAMBDA_INVAR = 1e-3
    LAMBDA_VAR = 1e-3
    LAMBDA_CELL = 1e-2
    LAMBDA_SPEC = 1e-3
    LAMBDA_ORTHO = 1e-3
    EMA_BETA = 0.1
    GAMMA = 1.0



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


def train_epoch(model, dataloader, optimizer, scheduler, epoch_idx, footprinter, W_com, W_spec, P_spec2com, alpha, cell_classifier, context_proj, cell_label_map, ema_mu_com, proj_params):
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
        mlm_logits, enhancer_final, _, seq_length_mapping = model(masked_enhancer_ids, promoter_ids, mask_positions)

        v = footprinter.forward_vector(enhancer_final.detach())
        z_com = W_com(v)
        z_spec = W_spec(v)

        B = enhancer_ids.size(0)
        V = mlm_logits.size(-1)
        D_com = z_com.size(-1)

        unique_cells = []
        for c in cell_lines:
            if c not in unique_cells:
                unique_cells.append(c)

        u_batch = torch.zeros(B, D_com, device=device, dtype=mlm_logits.dtype)
        g_com_list = []
        g_spec_list = []
        cell_label_list = []
        for c in unique_cells:
            idx = [i for i in range(B) if cell_lines[i] == c]
            if len(idx) == 0:
                continue
            idx_t = torch.tensor(idx, device=device)
            g_c_com = z_com.index_select(0, idx_t).mean(dim=0)
            g_c_spec = z_spec.index_select(0, idx_t).mean(dim=0)
            tilde_g_spec = P_spec2com(g_c_spec)
            u_c = g_c_com + alpha * tilde_g_spec
            u_batch.index_copy_(0, idx_t, u_c.unsqueeze(0).expand(len(idx), -1))
            g_com_list.append(g_c_com)
            g_spec_list.append(g_c_spec)
            cell_label_list.append(cell_label_map.get(c, 0))

        context_bias = context_proj(u_batch)
        context_bias = context_bias.unsqueeze(1).expand(B, mlm_logits.size(1), V)
        mlm_logits_biased = mlm_logits + context_bias
        
        # 计算损失
        bert_loss, accuracy = model.compute_mlm_loss(mlm_logits_biased, original_enhancer_ids, mask_positions, seq_length_mapping)

        g_com_stack = torch.stack(g_com_list) if len(g_com_list) > 0 else torch.zeros(1, D_com, device=device, dtype=v.dtype)
        g_spec_stack = torch.stack(g_spec_list) if len(g_spec_list) > 0 else torch.zeros(1, z_spec.size(-1), device=device, dtype=v.dtype)

        loss_alpha = proj_params['lambda_alpha'] * (alpha.pow(2).sum())
        loss_invar = proj_params['lambda_invar'] * ((g_com_stack - ema_mu_com).pow(2).mean())
        ema_mu_com.mul_(1 - proj_params['ema_beta']).add_(g_com_stack.detach().mean(dim=0) * proj_params['ema_beta'])

        zc_center = z_com - z_com.mean(dim=0)
        var = zc_center.pow(2).mean(dim=0)
        loss_var = proj_params['lambda_var'] * ((proj_params['gamma'] - torch.sqrt(var + 1e-8)).clamp(min=0).pow(2).sum())
        normed = zc_center / (torch.sqrt(var + 1e-8))
        cov = (normed.T @ normed) / max(B, 1)
        off = cov - torch.diag(torch.diag(cov))
        loss_cov = proj_params['lambda_var'] * (off.pow(2).sum())

        if g_spec_stack.size(0) > 0:
            logits_cell = cell_classifier(g_spec_stack.detach())
            labels_tensor = torch.tensor(cell_label_list, device=device)
            loss_cell = proj_params['lambda_cell'] * F.cross_entropy(logits_cell, labels_tensor)
        else:
            loss_cell = torch.tensor(0.0, device=device)

        loss_spec_reg = proj_params['lambda_spec'] * (g_spec_stack.abs().mean() + g_spec_stack.pow(2).mean())

        g_spec_map = {}
        for i, c in enumerate(unique_cells):
            g_spec_map[c] = g_spec_list[i]
        g_per_sample = torch.stack([g_spec_map.get(cell_lines[i], torch.zeros_like(z_spec[0])) for i in range(B)]).to(device).detach()
        loss_within = proj_params['lambda_spec'] * ((z_spec - g_per_sample).pow(2).mean())

        w_ortho = W_com.weight @ W_spec.weight.t()
        loss_ortho = proj_params['lambda_ortho'] * (w_ortho.pow(2).sum())

        total_loss_batch = bert_loss + loss_alpha + loss_invar + loss_var + loss_cov + loss_cell + loss_spec_reg + loss_within + loss_ortho
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), BERT_MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += total_loss_batch.item()
        total_accuracy += accuracy
        num_batches += 1
        
        # 更新进度条
        train_pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate(model, dataloader, cell_name="", footprinter=None, W_com=None, W_spec=None, P_spec2com=None, alpha=None, context_proj=None, cell_label_map=None):
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
            mlm_logits, enhancer_final, _, seq_length_mapping = model(masked_enhancer_ids, promoter_ids, mask_positions)

            if footprinter is not None:
                v = footprinter.forward_vector(enhancer_final)
                z_com = W_com(v)
                z_spec = W_spec(v)
                B = enhancer_ids.size(0)
                D_com = z_com.size(-1)
                unique_cells = []
                for c in cell_lines:
                    if c not in unique_cells:
                        unique_cells.append(c)
                u_batch = torch.zeros(B, D_com, device=device, dtype=mlm_logits.dtype)
                for c in unique_cells:
                    idx = [i for i in range(B) if cell_lines[i] == c]
                    if len(idx) == 0:
                        continue
                    idx_t = torch.tensor(idx, device=device)
                    g_c_com = z_com.index_select(0, idx_t).mean(dim=0)
                    g_c_spec = z_spec.index_select(0, idx_t).mean(dim=0)
                    tilde_g_spec = P_spec2com(g_c_spec)
                    u_c = g_c_com + alpha * tilde_g_spec
                    u_batch.index_copy_(0, idx_t, u_c.unsqueeze(0).expand(len(idx), -1))
                context_bias = context_proj(u_batch)
                context_bias = context_bias.unsqueeze(1).expand(B, mlm_logits.size(1), mlm_logits.size(-1))
                mlm_logits = mlm_logits + context_bias
            
            # 计算损失
            loss, accuracy = model.compute_mlm_loss(mlm_logits, original_enhancer_ids, mask_positions, seq_length_mapping)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


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

    footprinter = LCWnetFootprint(d_model=OUT_CHANNELS).to(device)
    D_com = OUT_CHANNELS
    D_spec = max(PretrainProjConfig.MIN_SPEC_DIM, OUT_CHANNELS // PretrainProjConfig.SPEC_RATIO)
    W_com = torch.nn.Linear(D_com, D_com, bias=False).to(device)
    W_spec = torch.nn.Linear(D_com, D_spec, bias=False).to(device)
    P_spec2com = torch.nn.Linear(D_spec, D_com, bias=False).to(device)
    alpha = torch.nn.Parameter(torch.tensor(PretrainProjConfig.ALPHA_INIT, device=device))
    cell_classifier = torch.nn.Linear(D_spec, len(unique_cells_train)).to(device)
    context_proj = torch.nn.Linear(D_com, DNA_EMBEDDING_VOCAB_SIZE).to(device)

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
        list(model.parameters())
        + list(footprinter.parameters())
        + list(W_com.parameters())
        + list(W_spec.parameters())
        + list(P_spec2com.parameters())
        + [alpha]
        + list(cell_classifier.parameters())
        + list(context_proj.parameters()),
        lr=BERT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCH
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    logger.info(f"批量大小: {PRISM_BATCH_SIZE} (对比采样: {PRISM_BATCH_SIZE//2}同细胞系 + {PRISM_BATCH_SIZE//2}不同细胞系)")
    logger.info(f"训练轮数: {EPOCH}")
    logger.info(f"学习率: {BERT_LEARNING_RATE}")
    logger.info(f"总训练步数: {total_steps}")
    
    # 训练循环
    logger.info("=" * 80)
    logger.info("开始训练")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch_idx in range(EPOCH):
        # 训练
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch_idx,
            footprinter,
            W_com,
            W_spec,
            P_spec2com,
            alpha,
            cell_classifier,
            context_proj,
            cell_label_map,
            ema_mu_com,
            proj_params,
        )
        
        logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 验证
        if epoch_idx % VALIDATION_INTERVAL == 0 or epoch_idx == EPOCH - 1:
            val_loss, val_acc = validate(
                model,
                val_loader,
                "ALL",
                footprinter,
                W_com,
                W_spec,
                P_spec2com,
                alpha,
                context_proj,
                cell_label_map,
            )
            logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # # 保存最佳模型
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     save_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_best.pth")
            #     torch.save(model.state_dict(), save_path)
            #     logger.info(f"保存最佳模型: {save_path}")
            
            # 定期保存检查点
            checkpoint_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_epoch_{epoch_idx+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path}")
    
    logger.info("=" * 80)
    logger.info("PRISM预训练完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
