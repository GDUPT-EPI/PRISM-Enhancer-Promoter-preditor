#!/usr/bin/env python3
"""
旁路解耦训练脚本（Bypass Decoupling Adversarial Training）

目标：
- 使用 `AuxiliaryModel` 对序列对进行旁路训练，得到 M=[G,F,I]
- 采用特性鉴别损失（F→cell）、共性对抗损失（GRL(G)→cell）、正交约束与一致性约束
- 每个epoch绘制并保存验证图到 `./compete/decouple/`（英文标签）

规范：遵循集中配置；注释中文；不使用命令行参数。
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List

from config import (
    DEVICE,
    DOMAIN_KL_DIR,
    PRISM_BATCH_SIZE,
    BYPASS_BATCH_SIZE,
    BYPASS_EPOCHS,
    BYPASS_LEARNING_RATE,
    BYPASS_WEIGHT_DECAY,
    BYPASS_SPEC_WEIGHT,
    BYPASS_INV_WEIGHT,
    BYPASS_ORTHO_WEIGHT,
    BYPASS_CONSIST_WEIGHT,
    PROJECT_ROOT,
    BYPASS_MAX_BATCHES_PER_EPOCH,
)

from data_loader import load_prism_data, PRISMDataset, RandomBatchSampler
from models.AuxiliaryModel import AuxiliaryModel
from models.layers.footprint import FootprintExpert


def ensure_out_dir() -> str:
    out_dir = os.path.join(PROJECT_ROOT, 'compete', 'decouple')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_cell_label_tensor(cells: List[str], all_cells: List[str]) -> torch.Tensor:
    """将细胞系名称映射为索引张量"""
    name_to_idx = {c: i for i, c in enumerate(all_cells)}
    idxs = [name_to_idx.get(x, 0) for x in cells]
    return torch.tensor(idxs, dtype=torch.long)


def pair_consistency_indices(pairs_df) -> Dict[Tuple[str, str], List[int]]:
    """
    为一致性约束构建索引：同一 (enhancer_seq, promoter_seq) 的不同细胞系样本索引集合。
    """
    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, row in pairs_df.iterrows():
        key = (row['enhancer_name'], row['promoter_name'])
        groups.setdefault(key, []).append(idx)
    return groups


def sample_consistency_pairs(group_map: Dict[Tuple[str, str], List[int]], max_pairs: int) -> List[Tuple[int, int]]:
    """从相同序列组中随机抽取索引对用于一致性损失"""
    pairs: List[Tuple[int, int]] = []
    for _, idxs in group_map.items():
        if len(idxs) >= 2:
            np.random.shuffle(idxs)
            for i in range(0, len(idxs) - 1, 2):
                pairs.append((idxs[i], idxs[i+1]))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def plot_epoch_curves(out_dir: str, history: Dict[str, List[float]], epoch: int) -> None:
    """绘制训练曲线（英文标签）"""
    plt.figure(figsize=(8, 5))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bypass Training Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch:02d}_loss.png'))
    plt.close()


def main():
    # 准备输出目录
    out_dir = ensure_out_dir()

    # 加载数据（ALL训练集）
    pairs_df, e_seqs, p_seqs = load_prism_data('train')
    all_cells = sorted(pairs_df['cell_line'].unique().tolist())

    dataset = PRISMDataset(pairs_df, e_seqs, p_seqs)
    sampler = RandomBatchSampler(dataset, batch_size=BYPASS_BATCH_SIZE, shuffle=True)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)

    # 构建一致性索引映射
    group_map = pair_consistency_indices(pairs_df)

    # 模型与优化器
    model = AuxiliaryModel(num_cell_types=len(all_cells))
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BYPASS_LEARNING_RATE, weight_decay=BYPASS_WEIGHT_DECAY)

    history = { 'total': [], 'spec': [], 'adv': [], 'orth': [], 'consist': [] }

    for epoch in range(BYPASS_EPOCHS):
        model.train()
        total_loss_epoch = 0.0
        spec_loss_epoch = 0.0
        adv_loss_epoch = 0.0
        orth_loss_epoch = 0.0
        consist_loss_epoch = 0.0
        n_batches = 0

        for bi, batch_indices in enumerate(sampler):
            # 组装一个批次
            batch = [dataset[i] for i in batch_indices]
            # 使用与PRISM相同的tokenizer预处理
            from PRISM import prism_collate_fn
            enh_ids, pr_ids, cell_lines, labels_t = prism_collate_fn(batch)

            cell_t = build_cell_label_tensor(cell_lines, all_cells)
            enh_ids = enh_ids.to(DEVICE)
            pr_ids = pr_ids.to(DEVICE)
            cell_t = cell_t.to(DEVICE)

            # 前向得到 z 与分类logits
            optimizer.zero_grad()
            pred_prob, extras = model(enh_ids, pr_ids, cell_labels=cell_t)

            spec_loss = extras['spec_loss']
            adv_loss = extras['adv_loss']

            # 正交约束：来自FootprintExpert
            fp_enh: FootprintExpert = model.fp_enh
            fp_pr: FootprintExpert = model.fp_pr
            orth_loss = fp_enh.orthogonality_loss(extras['zG'], extras['zF'], extras['zI']) 
            orth_loss = orth_loss + fp_pr.orthogonality_loss(extras['zG'], extras['zF'], extras['zI']) * 0.0  # 已合并z，避免重复加重

            # 一致性约束：同序列对在不同细胞系的 z_G/z_I 应接近
            consist_loss = torch.tensor(0.0, device=DEVICE)
            pairs = sample_consistency_pairs(group_map, max_pairs=8)
            if len(pairs) > 0:
                for (i, j) in pairs:
                    xi = dataset[i]
                    xj = dataset[j]
                    # 编码两样本
                    ei_ids, pj_ids, _, _ = prism_collate_fn([xi])
                    ej_ids, ppj_ids, _, _ = prism_collate_fn([xj])
                    ei_ids = ei_ids.to(DEVICE)
                    pj_ids = pj_ids.to(DEVICE)
                    ej_ids = ej_ids.to(DEVICE)
                    ppj_ids = ppj_ids.to(DEVICE)
                    with torch.no_grad():
                        _, ex_i = model(ei_ids, pj_ids)
                        _, ex_j = model(ej_ids, ppj_ids)
                    consist_loss = consist_loss + F.mse_loss(ex_i['zG'], ex_j['zG']) + F.mse_loss(ex_i['zI'], ex_j['zI'])
                consist_loss = consist_loss / len(pairs)

            total_loss = (
                BYPASS_SPEC_WEIGHT * spec_loss +
                BYPASS_INV_WEIGHT * adv_loss +
                BYPASS_ORTHO_WEIGHT * orth_loss +
                BYPASS_CONSIST_WEIGHT * consist_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += total_loss.item()
            spec_loss_epoch += spec_loss.item()
            adv_loss_epoch += adv_loss.item()
            orth_loss_epoch += orth_loss.item()
            consist_loss_epoch += consist_loss.item()
            n_batches += 1
            if bi + 1 >=  BYPASS_MAX_BATCHES_PER_EPOCH:
                break

        # 记录与绘图
        avg_total = total_loss_epoch / max(1, n_batches)
        history['total'].append(avg_total)
        history['spec'].append(spec_loss_epoch / max(1, n_batches))
        history['adv'].append(adv_loss_epoch / max(1, n_batches))
        history['orth'].append(orth_loss_epoch / max(1, n_batches))
        history['consist'].append(consist_loss_epoch / max(1, n_batches))
        plot_epoch_curves(out_dir, history, epoch + 1)

        # 简要保存z的分布可视化（G/F/I范数）
        with torch.no_grad():
            norms = {
                'G_norm': history['orth'][-1],
                'F_loss_proxy': history['spec'][-1],
                'I_consist_proxy': history['consist'][-1],
            }
            plt.figure(figsize=(6,4))
            plt.bar(list(norms.keys()), list(norms.values()))
            plt.title('Bypass Proxies (epoch avg)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'epoch_{epoch+1:02d}_proxies.png'))
            plt.close()


if __name__ == '__main__':
    main()
