#!/usr/bin/env python3
"""
PRISM预训练脚本 - BERT风格的Masked Language Modeling
"""

from config import *
from config import PRISM_SAVE_MODEL_DIR
from data_loader import load_all_train_data, load_all_val_data, MyDataset

import logging
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset, Dataset
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


def simple_collate_fn(batch):
    """简化的collate函数"""
    enhancer_sequences = [item[0] for item in batch]
    promoter_sequences = [item[1] for item in batch]
    enhancer_features = [item[2] for item in batch]
    promoter_features = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    
    # 填充序列
    padded_enhancer_sequences = pad_sequence(enhancer_sequences, batch_first=True, padding_value=0)
    padded_promoter_sequences = pad_sequence(promoter_sequences, batch_first=True, padding_value=0)
    
    # 确保最小长度
    if padded_enhancer_sequences.size(1) < MAX_ENHANCER_LENGTH:
        padding_size = MAX_ENHANCER_LENGTH - padded_enhancer_sequences.size(1)
        padded_enhancer_sequences = torch.nn.functional.pad(
            padded_enhancer_sequences, (0, padding_size), mode='constant', value=0
        )
    
    if padded_promoter_sequences.size(1) < MAX_PROMOTER_LENGTH:
        padding_size = MAX_PROMOTER_LENGTH - padded_promoter_sequences.size(1)
        padded_promoter_sequences = torch.nn.functional.pad(
            padded_promoter_sequences, (0, padding_size), mode='constant', value=0
        )
    
    padded_enhancer_features = torch.stack(enhancer_features)
    padded_promoter_features = torch.stack(promoter_features)
    labels = torch.tensor(labels, dtype=torch.float)
    
    return padded_enhancer_sequences, padded_promoter_sequences, padded_enhancer_features, padded_promoter_features, labels


class OptimizedCombinedDataset(Dataset):
    """优化的组合数据集"""
    def __init__(self, enhancers, promoters, labels, cache_dir=None, use_cache=True):
        self.dataset = create_optimized_dataset(
            enhancers=enhancers,
            promoters=promoters,
            labels=labels,
            cache_dir=cache_dir,
            use_cache=use_cache
        )
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


def train_epoch(model, dataloader, optimizer, scheduler, epoch_idx):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    train_pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{EPOCH} [Training]", 
                      leave=True, dynamic_ncols=True)
    
    for data in train_pbar:
        enhancer_ids, promoter_ids, _, _, _ = data
        enhancer_ids = enhancer_ids.to(device, non_blocking=True)
        promoter_ids = promoter_ids.to(device, non_blocking=True)
        
        # 创建MLM mask
        masked_enhancer_ids, mask_positions, original_enhancer_ids = create_mlm_mask(
            enhancer_ids,
            mask_prob=BERT_MASK_PROB,
            mask_token_id=BERT_MASK_TOKEN_ID,
            vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
            pad_token_id=BERT_PAD_TOKEN_ID
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
            enhancer_ids, promoter_ids, _, _, _ = data
            enhancer_ids = enhancer_ids.to(device, non_blocking=True)
            promoter_ids = promoter_ids.to(device, non_blocking=True)
            
            # 创建MLM mask
            masked_enhancer_ids, mask_positions, original_enhancer_ids = create_mlm_mask(
                enhancer_ids,
                mask_prob=BERT_MASK_PROB,
                mask_token_id=BERT_MASK_TOKEN_ID,
                vocab_size=DNA_EMBEDDING_VOCAB_SIZE,
                pad_token_id=BERT_PAD_TOKEN_ID
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
    logger.info("PRISM预训练开始")
    logger.info("=" * 80)
    
    # 加载数据
    logger.info("加载训练数据...")
    train_data = load_all_train_data()
    val_data = load_all_val_data()
    
    logger.info(f"训练细胞系: {', '.join(sorted(train_data.keys()))}")
    logger.info(f"验证细胞系: {', '.join(sorted(val_data.keys()))}")
    
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
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loaders = {}
    train_folds = []
    
    for cell, (enhancers_train, promoters_train, labels_train) in train_data.items():
        train_dataset = MyDataset(enhancers_train, promoters_train, labels_train)
        enh_train_raw = [train_dataset[i][0] for i in range(len(train_dataset))]
        prom_train_raw = [train_dataset[i][1] for i in range(len(train_dataset))]
        labels_train_raw = [train_dataset[i][2] for i in range(len(train_dataset))]
        
        train_fold = OptimizedCombinedDataset(
            enhancers=enh_train_raw,
            promoters=prom_train_raw,
            labels=labels_train_raw,
            cache_dir=os.path.join(CACHE_DIR, f"prism_train_{cell}"),
            use_cache=True,
        )
        train_folds.append(train_fold)
        train_loaders[cell] = DataLoader(
            dataset=train_fold,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=simple_collate_fn,
        )
    
    # 合并所有训练数据
    if len(train_folds) > 0:
        all_train_dataset = ConcatDataset([tf.dataset for tf in train_folds])
        train_loaders["ALL"] = DataLoader(
            dataset=all_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=simple_collate_fn,
        )
    
    # 创建验证数据加载器
    val_loaders = {}
    for cell, (enhancers_val, promoters_val, labels_val) in val_data.items():
        val_dataset = MyDataset(enhancers_val, promoters_val, labels_val)
        enh_raw = [val_dataset[i][0] for i in range(len(val_dataset))]
        prom_raw = [val_dataset[i][1] for i in range(len(val_dataset))]
        labels_raw = [val_dataset[i][2] for i in range(len(val_dataset))]
        
        val_fold = OptimizedCombinedDataset(
            enhancers=enh_raw,
            promoters=prom_raw,
            labels=labels_raw,
            cache_dir=os.path.join(CACHE_DIR, f"prism_val_{cell}"),
            use_cache=True
        )
        val_loaders[cell] = DataLoader(
            dataset=val_fold,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=simple_collate_fn,
        )
    
    # 创建优化器和调度器
    optimizer = model.optimizer
    total_steps = len(train_loaders[TRAIN_CELL_LINE]) * EPOCH
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    logger.info(f"批量大小: {BATCH_SIZE}")
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
            train_loaders[TRAIN_CELL_LINE], 
            optimizer, 
            scheduler, 
            epoch_idx
        )
        
        logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 验证
        if epoch_idx % VALIDATION_INTERVAL == 0 or epoch_idx == EPOCH - 1:
            for cell, loader in val_loaders.items():
                val_loss, val_acc = validate(model, loader, cell)
                logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Val [{cell}] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                # # 保存最佳模型
                # if cell == TRAIN_CELL_LINE and val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     save_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_best_{cell}.pth")
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
