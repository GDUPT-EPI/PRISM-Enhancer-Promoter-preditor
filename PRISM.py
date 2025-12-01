#!/usr/bin/env python3
"""
PRISM预训练脚本 - BERT风格的Masked Language Modeling
"""

from config import *
from config import PRISM_SAVE_MODEL_DIR, PRISM_BATCH_SIZE
from data_loader import load_prism_data, PRISMDataset, CellBatchSampler

import logging
from datetime import datetime
from torch.utils.data import DataLoader
from models.pleat.optimized_pre import create_optimized_dataset
from models.PRISMModel import PRISMBackbone, CellClassificationExpert
from models.layers.footprint import LCWnetFootprint, FootprintConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import re
import xml.etree.ElementTree as ET


# 预训练投影配置类 - 定义高维投影任务的超参数
class PretrainProjConfig:
    pass



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

    K = CNN_KERNEL_SIZE
    P = POOL_KERNEL_SIZE
    pad_id = DNA_EMBEDDING_PADDING_IDX
    min_req = K + P - 1

    max_len_en = max(int(x.size(0)) for x in enhancer_ids_list) if enhancer_ids_list else min_req
    max_len_pr = max(int(x.size(0)) for x in promoter_ids_list) if promoter_ids_list else min_req

    adj_base_en = max(1, max_len_en - (K - 1))
    adj_base_pr = max(1, max_len_pr - (K - 1))
    target_len_en = (K - 1) + ((adj_base_en + P - 1) // P) * P
    target_len_pr = (K - 1) + ((adj_base_pr + P - 1) // P) * P
    target_len_en = max(min_req, min(target_len_en, MAX_ENHANCER_LENGTH))
    target_len_pr = max(min_req, min(target_len_pr, MAX_PROMOTER_LENGTH))

    processed_en = []
    for ids in enhancer_ids_list:
        L = int(ids.size(0))
        if L > target_len_en:
            s = (L - target_len_en) // 2
            ids = ids[s:s + target_len_en]
        processed_en.append(ids)
    processed_pr = []
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
    
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    
    return padded_enhancer_ids, padded_promoter_ids, cell_lines, labels_tensor


def apply_random_mask(x_ids: torch.Tensor, mask_prob: float = PRISM_RANDOM_MASK_PROB, pad_id: int = PRISM_RANDOM_MASK_PAD_ID) -> torch.Tensor:
    B, L = x_ids.shape
    valid = (x_ids != pad_id)
    rand = torch.rand(B, L, device=x_ids.device)
    sel = (rand < mask_prob) & valid
    x_masked = x_ids.clone()
    x_masked[sel] = pad_id
    return x_masked


def train_epoch(model, dataloader, optimizer, scheduler, epoch_idx, cell_label_map, ema_mu_com, proj_params, kb, loss_weights):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_spec_acc = 0.0
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
            spec_acc_batch = (logits.argmax(dim=1) == local_labels).float().mean()
        else:
            loss_cell = torch.tensor(0.0, device=device)
            spec_acc_batch = torch.tensor(0.0, device=device)

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
        total_spec_acc += spec_acc_batch.item()
        num_batches += 1
        
        # 更新进度条
        train_pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'spec_acc': f'{spec_acc_batch.item():.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    avg_spec_acc = total_spec_acc / num_batches
    return avg_loss, avg_accuracy, avg_spec_acc

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

def _load_resume_state(save_dir: str, device: torch.device, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: ReduceLROnPlateau, cell_expert: torch.nn.Module | None = None):
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
            if cell_expert is not None and 'cell_expert_state' in state:
                try:
                    cell_expert.load_state_dict(state['cell_expert_state'], strict=False)
                except Exception:
                    pass
            if 'optimizer_state' in state:
                try:
                    saved_opt = state['optimizer_state']
                    saved_groups = len(saved_opt.get('param_groups', []))
                    curr_groups = len(optimizer.state_dict().get('param_groups', []))
                    saved_mode = bool(state.get('train_expert_only', False))
                    if saved_groups == curr_groups and saved_mode == TRAIN_EXPERT_ONLY:
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
                if cell_expert is not None and 'cell_expert' in sd:
                    try:
                        cell_expert.load_state_dict(sd['cell_expert'], strict=False)
                    except Exception:
                        pass
            logger.info(f"加载模型检查点: {model_path} (epoch={latest_epoch})")
        if os.path.exists(kb_path):
            kb = torch.load(kb_path, map_location='cpu')
    return latest_epoch, kb, ema_mu


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
    
    unique_cells_train = sorted(train_pairs_df['cell_line'].unique())
    
    # 创建数据集
    train_dataset = PRISMDataset(train_pairs_df, train_e_seqs, train_p_seqs)
    
    logger.info("创建数据加载器...")
    if USE_RANDOM_EP_DATALOADER:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=PRISM_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=prism_collate_fn,
        )
        logger.info("使用随机EP导入 (跨细胞系随机批次)")
    else:
        train_sampler = CellBatchSampler(train_dataset, batch_size=PRISM_BATCH_SIZE, shuffle=True)
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
    xml_path = os.path.join(PROJECT_ROOT, "vocab", "cell_type.xml")
    def load_cell_types(path: str):
        if os.path.exists(path):
            try:
                root = ET.parse(path).getroot()
                names = []
                for node in root.findall(".//type"):
                    name = node.get("name")
                    if name:
                        names.append(name.strip())
                names = [n for n in names if n]
                if names:
                    return names
            except Exception:
                pass
        return []
    fixed_cells = load_cell_types(xml_path)
    if not fixed_cells:
        fixed_cells = unique_cells_train
    label_map = {c: i for i, c in enumerate(fixed_cells)}
    other_id = label_map.get("OTHER", None)
    num_cells = len(fixed_cells)
    cell_expert = CellClassificationExpert(num_classes=num_cells).to(device)
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

    if TRAIN_EXPERT_ONLY:
        for p in model.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(list(cell_expert.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        training_mode = "expert_only"
    elif TRAIN_EP_ONLY:
        for p in cell_expert.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        training_mode = "ep_only"
    else:
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(cell_expert.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        training_mode = "joint"
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
    start_epoch, _kb, _ema_mu = _load_resume_state(PRISM_SAVE_MODEL_DIR, device, model, optimizer, scheduler, cell_expert)
    if start_epoch > 0:
        logger.info(f"从最近权重恢复: epoch {start_epoch}")
    else:
        logger.info("未发现可恢复检查点，将从头开始训练")
    if start_epoch >= EPOCH:
        logger.info("已达到或超过目标训练轮数，无需继续。若需追加训练，请增大EPOCH或删除旧检查点。")
    for epoch_idx in range(start_epoch, EPOCH):
        # 训练
        if TRAIN_EXPERT_ONLY:
            logger.info("模式: 仅训练细胞专家头")
        elif TRAIN_EP_ONLY:
            logger.info("模式: 仅训练EP互作主干")
        else:
            logger.info("模式: 联合训练 (cell=0.35, ep=0.65)")
        model.train(); cell_expert.train()
        total_loss = 0.0; total_cell_acc = 0.0; total_ep_acc = 0.0; n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{EPOCH} [Training]", leave=True, dynamic_ncols=True)
        for batch in pbar:
            enh_ids, pr_ids, cell_lines, labels = batch
            enh_ids = enh_ids.to(device); pr_ids = pr_ids.to(device)
            enh_ids = apply_random_mask(enh_ids)
            pr_ids = apply_random_mask(pr_ids)
            labels = labels.to(device)
            cell_targets = torch.tensor([cell_label_map.get(c, other_id if other_id is not None else 0) for c in cell_lines], device=device, dtype=torch.long)

            if TRAIN_EXPERT_ONLY:
                cell_logits = cell_expert(enh_ids, pr_ids)
                if cell_logits.dim() == 3 and cell_logits.size(1) == 1:
                    cell_logits = cell_logits.squeeze(1)
                cell_loss = F.cross_entropy(cell_logits, cell_targets)
                with torch.no_grad():
                    cell_acc = (cell_logits.argmax(dim=-1) == cell_targets).float().mean().item()
                loss = cell_loss
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(cell_expert.parameters()), BERT_MAX_GRAD_NORM); optimizer.step()
                ep_acc = 0.0
            elif TRAIN_EP_ONLY:
                ep_outputs, adaptive_loss = model(enh_ids, pr_ids, None)
                ep_outputs = ep_outputs.squeeze(-1)
                ep_loss = model.compute_loss(ep_outputs, labels.float(), adaptive_loss)
                with torch.no_grad():
                    ep_preds = (ep_outputs >= 0.5).long()
                    ep_acc = (ep_preds == labels.long()).float().mean().item()
                loss = ep_loss
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()), BERT_MAX_GRAD_NORM); optimizer.step()
                cell_acc = 0.0
            else:
                cell_logits = cell_expert(enh_ids, pr_ids)
                if cell_logits.dim() == 3 and cell_logits.size(1) == 1:
                    cell_logits = cell_logits.squeeze(1)
                cell_loss = F.cross_entropy(cell_logits, cell_targets)
                with torch.no_grad():
                    cell_acc = (cell_logits.argmax(dim=-1) == cell_targets).float().mean().item()
                ep_outputs, adaptive_loss = model(enh_ids, pr_ids, cell_logits)
                ep_outputs = ep_outputs.squeeze(-1)
                ep_loss = model.compute_loss(ep_outputs, labels.float(), adaptive_loss)
                with torch.no_grad():
                    ep_preds = (ep_outputs >= 0.5).long()
                    ep_acc = (ep_preds == labels.long()).float().mean().item()
                loss = PRISM_CELL_LOSS_WEIGHT * cell_loss + PRISM_EP_LOSS_WEIGHT * ep_loss
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(cell_expert.parameters()), BERT_MAX_GRAD_NORM); optimizer.step()

            total_loss += loss.item(); total_cell_acc += cell_acc; total_ep_acc += ep_acc; n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cell_acc': f'{cell_acc:.4f}', 'ep_acc': f'{ep_acc:.4f}'})

        avg_loss = total_loss / max(n_batches, 1)
        avg_cell_acc = total_cell_acc / max(n_batches, 1)
        avg_ep_acc = total_ep_acc / max(n_batches, 1)
        logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Train Loss: {avg_loss:.4f}, Cell Acc: {avg_cell_acc:.4f}, EP Acc: {avg_ep_acc:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_epoch_{epoch_idx+1}.pth")
        torch.save({'backbone': model.state_dict(), 'cell_expert': cell_expert.state_dict()}, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")
        full_state_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_full_epoch_{epoch_idx+1}.pt")
        full_state = {
            'model_state': model.state_dict(),
            'cell_expert_state': cell_expert.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch_idx + 1,
            'train_expert_only': TRAIN_EXPERT_ONLY,
            'train_ep_only': TRAIN_EP_ONLY,
            'use_random_ep_dataloader': USE_RANDOM_EP_DATALOADER,
            'training_mode': training_mode,
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

