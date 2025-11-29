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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


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


def train_epoch(model, dataloader, optimizer, scheduler, epoch_idx):
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
        mlm_logits, _, _, seq_length_mapping = model(masked_enhancer_ids, promoter_ids, mask_positions)
        
        # 计算损失
        loss, accuracy = model.compute_mlm_loss(mlm_logits, original_enhancer_ids, mask_positions, seq_length_mapping)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), BERT_MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
        
        # 更新进度条
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate(model, dataloader, cell_name=""):
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
            mlm_logits, _, _, seq_length_mapping = model(masked_enhancer_ids, promoter_ids, mask_positions)
            
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
    optimizer = model.optimizer
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
            epoch_idx
        )
        
        logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 验证
        if epoch_idx % VALIDATION_INTERVAL == 0 or epoch_idx == EPOCH - 1:
            val_loss, val_acc = validate(model, val_loader, "ALL")
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
