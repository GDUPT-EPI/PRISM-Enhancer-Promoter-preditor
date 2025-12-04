#!/usr/bin/env python3
"""
PRISM训练脚本

用途:
- 使用`PRISMBackbone`进行EP互作概率预测训练

说明:
- 已彻底移除专家网络（CellClassificationExpert）相关逻辑
- 已移除过时的MLM相关逻辑，简化损失与训练流程
"""

from config import *
from config import PRISM_SAVE_MODEL_DIR, PRISM_BATCH_SIZE
from data_loader import load_prism_data, PRISMDataset, CellBatchSampler

import logging
from datetime import datetime
from torch.utils.data import DataLoader
from models.PRISMModel import PRISMBackbone
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import re
import xml.etree.ElementTree as ET

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


# 配置日志系统
def setup_logging():
    """配置日志系统"""
    log_filename = os.path.join(LOG_DIR, f"prism_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")  # 日志文件路径
    log_format = '%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format=log_format,  # 设置日志格式
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),  # 写入文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)  # 返回logger实例


logger = setup_logging()  # 初始化日志
logger.info("PRISM预训练日志系统已初始化")  # 输出初始化信息
logger.info(f"日志文件: {LOG_DIR}")  # 日志目录
logger.info(f"预处理线程数: {PREPROCESS_NUM_THREADS}")  # 预处理线程数


def prism_collate_fn(batch):
    """PRISM批处理拼接函数

    输入项: `(enhancer_seq, promoter_seq, cell_line, label)`

    Returns:
        `(enh_ids, pr_ids, cell_lines, labels)`
    """
    from models.pleat.embedding import KMerTokenizer
    
    enhancer_seqs = [item[0] for item in batch]  # 增强子序列
    promoter_seqs = [item[1] for item in batch]  # 启动子序列
    cell_lines = [item[2] for item in batch]  # 细胞系名称
    labels = [item[3] for item in batch]  # 标签
    
    # 创建tokenizer (只创建一次)
    if not hasattr(prism_collate_fn, 'tokenizer'):
        prism_collate_fn.tokenizer = KMerTokenizer()
    
    tokenizer = prism_collate_fn.tokenizer
    
    # 将DNA序列转换为token IDs
    enhancer_ids_list = [tokenizer.encode(seq) for seq in enhancer_seqs]  # 编码增强子
    promoter_ids_list = [tokenizer.encode(seq) for seq in promoter_seqs]  # 编码启动子

    K = CNN_KERNEL_SIZE
    P = POOL_KERNEL_SIZE
    pad_id = DNA_EMBEDDING_PADDING_IDX  # PAD索引
    min_req = K + P - 1

    max_len_en = max(int(x.size(0)) for x in enhancer_ids_list) if enhancer_ids_list else min_req
    max_len_pr = max(int(x.size(0)) for x in promoter_ids_list) if promoter_ids_list else min_req

    adj_base_en = max(1, max_len_en - (K - 1))
    adj_base_pr = max(1, max_len_pr - (K - 1))
    target_len_en = (K - 1) + ((adj_base_en + P - 1) // P) * P
    target_len_pr = (K - 1) + ((adj_base_pr + P - 1) // P) * P
    target_len_en = max(min_req, min(target_len_en, MAX_ENHANCER_LENGTH))
    target_len_pr = max(min_req, min(target_len_pr, MAX_PROMOTER_LENGTH))

    processed_en = []  # 处理后的增强子序列
    for ids in enhancer_ids_list:
        L = int(ids.size(0))
        if L > target_len_en:
            s = (L - target_len_en) // 2
            ids = ids[s:s + target_len_en]
        processed_en.append(ids)
    processed_pr = []  # 处理后的启动子序列
    for ids in promoter_ids_list:
        L = int(ids.size(0))
        if L > target_len_pr:
            s = (L - target_len_pr) // 2
            ids = ids[s:s + target_len_pr]
        processed_pr.append(ids)

    padded_enhancer_ids = pad_sequence(processed_en, batch_first=True, padding_value=pad_id)
    padded_promoter_ids = pad_sequence(processed_pr, batch_first=True, padding_value=pad_id)

    if padded_enhancer_ids.size(1) < target_len_en:
        padded_enhancer_ids = torch.nn.functional.pad(padded_enhancer_ids, (0, target_len_en - padded_enhancer_ids.size(1)), value=pad_id)
    if padded_promoter_ids.size(1) < target_len_pr:
        padded_promoter_ids = torch.nn.functional.pad(padded_promoter_ids, (0, target_len_pr - padded_promoter_ids.size(1)), value=pad_id)
    
    labels_tensor = torch.tensor(labels, dtype=torch.float)  # 标签张量
    
    return padded_enhancer_ids, padded_promoter_ids, cell_lines, labels_tensor


# 过时随机掩码逻辑已移除


# 过时的MLM训练流程已移除

def _find_latest_epoch(save_dir: str) -> int:
    if not os.path.exists(save_dir):
        return 0
    epochs = []
    for name in os.listdir(save_dir):
        m1 = re.match(r"prism_epoch_(\d+)\.pth$", name)
        m2 = re.match(r"prism_full_epoch_(\d+)\.pt$", name)
        if m1:
            epochs.append(int(m1.group(1)))
        elif m2:
            epochs.append(int(m2.group(1)))
    return max(epochs) if epochs else 0

def _load_resume_state(save_dir: str, device: torch.device, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: ReduceLROnPlateau):
    latest_epoch = _find_latest_epoch(save_dir)
    if latest_epoch <= 0:
        return 0, {}, torch.zeros(OUT_CHANNELS, device=device)
    full_path = os.path.join(save_dir, f"prism_full_epoch_{latest_epoch}.pt")
    kb_path = os.path.join(save_dir, f"footprint_kb_epoch_{latest_epoch}.pt")
    model_path = os.path.join(save_dir, f"prism_epoch_{latest_epoch}.pth")
    ema_mu = torch.zeros(OUT_CHANNELS, device=device)
    kb = {}
    if os.path.exists(full_path):
        state = torch.load(full_path, map_location=device)
        if isinstance(state, dict):
            state_epoch = int(state.get('epoch', latest_epoch) or latest_epoch)
            latest_epoch = max(latest_epoch, state_epoch)
            logger.info(f"加载完整状态: {full_path} (epoch={state_epoch})")
            if 'model_state' in state:
                model.load_state_dict(state['model_state'], strict=False)
            if 'optimizer_state' in state:
                try:
                    saved_opt = state['optimizer_state']
                    saved_groups = len(saved_opt.get('param_groups', []))
                    curr_groups = len(optimizer.state_dict().get('param_groups', []))
                    if saved_groups == curr_groups:
                        optimizer.load_state_dict(saved_opt)
                        logger.info("优化器状态已恢复")
                    else:
                        logger.info("优化器状态未加载：参数组或训练模式不匹配")
                except Exception:
                    pass
            if 'scheduler_state' in state:
                try:
                    if scheduler is not None:
                        scheduler.load_state_dict(state['scheduler_state'])
                        logger.info("调度器状态已恢复")
                except Exception:
                    pass
            if 'ema_mu_com' in state:
                ema_vals = state['ema_mu_com']
                ema_mu = ema_vals.to(device) if isinstance(ema_vals, torch.Tensor) else torch.tensor(ema_vals, device=device)
            kb = state.get('kb', {}) or {}
    else:
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=device)
            if isinstance(sd, dict):
                if 'backbone' in sd:
                    model.load_state_dict(sd['backbone'], strict=False)
                else:
                    model.load_state_dict(sd, strict=False)
            logger.info(f"加载模型检查点: {model_path} (epoch={latest_epoch})")
        if os.path.exists(kb_path):
            kb = torch.load(kb_path, map_location='cpu')
    return latest_epoch, kb, ema_mu


# 过时的MLM验证流程已移除


# 过时的可分性验证流程已移除


def main():
    """入口函数"""
    logger.info("=" * 80)
    logger.info("PRISM预训练开始 (Domain-KL数据)")
    logger.info("=" * 80)
    
    # 加载PRISM特供数据
    logger.info("加载训练数据 (domain-kl)...")
    train_pairs_df, train_e_seqs, train_p_seqs = load_prism_data("train")
    logger.info(f"训练样本数: {len(train_pairs_df)}")
    logger.info(f"训练细胞系: {', '.join(sorted(train_pairs_df['cell_line'].unique()))}")
    
    unique_cells_train = sorted(train_pairs_df['cell_line'].unique())
    
    # 创建数据集
    train_dataset = PRISMDataset(train_pairs_df, train_e_seqs, train_p_seqs)
    
    # 创建对比采样器
    train_sampler = CellBatchSampler(train_dataset, batch_size=PRISM_BATCH_SIZE, shuffle=True)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=prism_collate_fn,
    )
    
    val_loader = None
    
    # 创建模型
    logger.info("创建PRISM模型...")
    xml_path = os.path.join(PROJECT_ROOT, "vocab", "cell_type.xml")  # 细胞系列表XML路径
    def load_cell_types(path: str):
        if os.path.exists(path):
            try:
                root = ET.parse(path).getroot()  # 解析XML
                names = []
                for node in root.findall(".//type"):
                    name = node.get("name")  # 读取name属性
                    if name:
                        names.append(name.strip())  # 去除空白并收集
                names = [n for n in names if n]
                if names:
                    return names
            except Exception:
                pass
        return []  # 文件不存在或解析失败时返回空列表
    fixed_cells = load_cell_types(xml_path)
    if not fixed_cells:
        fixed_cells = unique_cells_train
    label_map = {c: i for i, c in enumerate(fixed_cells)}
    other_id = label_map.get("OTHER", None)
    num_cells = len(fixed_cells)
    model = PRISMBackbone(num_classes=num_cells).to(device)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"GPU可用: {torch.cuda.is_available()}")
    logger.info(f"模型在GPU上: {next(model.parameters()).is_cuda}")
    
    # 创建优化器和调度器
    cell_label_map = label_map

    start_epoch = 0

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCH
    scheduler = None
    
    logger.info(f"批量大小: {PRISM_BATCH_SIZE} (纯细胞系批次)")
    logger.info(f"训练轮数: {EPOCH}")
    logger.info(f"学习率: {LEARNING_RATE}")
    logger.info(f"总训练步数: {total_steps}")
    
    # 训练循环
    logger.info("=" * 80)
    logger.info("开始训练")
    logger.info("=" * 80)
    
    os.makedirs(PRISM_SAVE_MODEL_DIR, exist_ok=True)
    start_epoch, _kb, _ema_mu = _load_resume_state(PRISM_SAVE_MODEL_DIR, device, model, optimizer, scheduler)
    if start_epoch > 0:
        logger.info(f"从最近权重恢复: epoch {start_epoch}")
    else:
        logger.info("未发现可恢复检查点，将从头开始训练")
    if start_epoch >= EPOCH:
        logger.info("已达到或超过目标训练轮数，无需继续。若需追加训练，请增大EPOCH或删除旧检查点。")
    for epoch_idx in range(start_epoch, EPOCH):
        # 训练
        logger.info("Loss Weights: ep=1.0")
        model.train()
        total_loss = 0.0; total_ep_acc = 0.0; n_batches = 0
        total_tp = 0; total_fp = 0; total_fn = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{EPOCH} [Training]", leave=True, dynamic_ncols=True)
        for batch in pbar:
            enh_ids, pr_ids, cell_lines, labels = batch
            enh_ids = enh_ids.to(device); pr_ids = pr_ids.to(device)
            # 暂停随机PAD掩码，避免训练初期数值不稳
            # enh_ids = apply_random_mask(enh_ids)
            # pr_ids = apply_random_mask(pr_ids)
            labels = labels.to(device)
            precision = 0.0; recall = 0.0; f1 = 0.0
            ep_outputs, adaptive_loss = model(enh_ids, pr_ids)
            ep_outputs = ep_outputs.squeeze(-1)
            ep_loss, loss_details = model.compute_loss(ep_outputs, labels.float(), adaptive_loss, return_details=True)
            with torch.no_grad():
                ep_preds = (ep_outputs >= 0.5).long()
                ep_acc = (ep_preds == labels.long()).float().mean().item()
                tp = int(((ep_preds == 1) & (labels.long() == 1)).sum().item())
                fp = int(((ep_preds == 1) & (labels.long() == 0)).sum().item())
                fn = int(((ep_preds == 0) & (labels.long() == 1)).sum().item())
                total_tp += tp; total_fp += fp; total_fn += fn
                precision = (tp / max(tp + fp, 1)) if (tp + fp) > 0 else 0.0
                recall = (tp / max(tp + fn, 1)) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / max(precision + recall, 1e-6)) if (precision + recall) > 0 else 0.0
            loss = ep_loss
            optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=GRAD_CLIP_MAX_NORM);
            optimizer.step()

            total_loss += loss.item(); total_ep_acc += ep_acc; n_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'base': f"{loss_details['base']:.4f}",
                'adaptive': f"{loss_details['adaptive']:.4f}",
                'penalty': f"{loss_details['penalty']:.4f}",
                'ep_acc': f'{ep_acc:.4f}',
                'prec': f'{precision:.4f}',
                'rec': f'{recall:.4f}',
                'f1': f'{f1:.4f}',
            })

        avg_loss = total_loss / max(n_batches, 1)
        avg_ep_acc = total_ep_acc / max(n_batches, 1)
        epoch_precision = (total_tp / max(total_tp + total_fp, 1)) if (total_tp + total_fp) > 0 else 0.0
        epoch_recall = (total_tp / max(total_tp + total_fn, 1)) if (total_tp + total_fn) > 0 else 0.0
        epoch_f1 = (2 * epoch_precision * epoch_recall / max(epoch_precision + epoch_recall, 1e-6)) if (epoch_precision + epoch_recall) > 0 else 0.0
        logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Train Loss: {avg_loss:.4f}, EP Acc: {avg_ep_acc:.4f}, Prec: {epoch_precision:.4f}, Rec: {epoch_recall:.4f}, F1: {epoch_f1:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_epoch_{epoch_idx+1}.pth")
        torch.save({'backbone': model.state_dict()}, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")
        full_state_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_full_epoch_{epoch_idx+1}.pt")
        full_state = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch_idx + 1,
        }
        if scheduler is not None:
            full_state['scheduler_state'] = scheduler.state_dict()
        torch.save(full_state, full_state_path)
        logger.info(f"保存完整状态: {full_state_path}")

        # 保存知识库 (包含中心点和模型状态)
        # 移除验证流程
            
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

