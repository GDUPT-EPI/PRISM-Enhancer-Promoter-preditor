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
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

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
    GCN_CENTER_LOSS_W,
    GCN_MARGIN_LOSS_W,
    GCN_SMOOTH_LOSS_W,
    PROJECT_ROOT,
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
    has_any = False
    y_max = 0.0
    for k, v in history.items():
        if not v:
            continue
        arr = np.array(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        x = np.arange(1, arr.size + 1)
        plt.plot(x, arr, label=k, marker='o', linewidth=1.5)
        y_max = max(y_max, float(arr.max()))
        has_any = True
    if not has_any:
        plt.text(0.5, 0.5, 'No data', ha='center')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bypass Training Loss Curves')
    if y_max > 0:
        plt.ylim(0, y_max * 1.1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch:02d}_loss.png'))
    plt.close()

def _safe_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def plot_tsne(out_dir: str, epoch: int, Z: np.ndarray, labels: np.ndarray, title: str) -> None:
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', random_state=42)
    Y = tsne.fit_transform(Z)
    plt.figure(figsize=(7,6))
    unique = np.unique(labels)
    try:
        import matplotlib
        colors = matplotlib.colormaps.get_cmap('tab20')
    except Exception:
        colors = plt.get_cmap('tab20')
    for i, u in enumerate(unique):
        idx = labels == u
        c = colors(i / max(len(unique)-1, 1)) if hasattr(colors, '__call__') else colors(i)
        plt.scatter(Y[idx,0], Y[idx,1], s=8, color=c, label=str(u), alpha=0.8)
    plt.title(title)
    plt.legend(markerscale=2, bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    out_name = f"epoch_{epoch:02d}_{title.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(out_dir, out_name))
    plt.close()

def plot_confusion(out_dir: str, epoch: int, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center', color='black')
    plt.tight_layout()
    out_name = f"epoch_{epoch:02d}_{title.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(out_dir, out_name))
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

    history = { 'total': [], 'spec': [], 'adv': [], 'orth': [], 'consist': [], 'gcn_center': [], 'gcn_margin': [], 'gcn_smooth': [] }

    for epoch in range(BYPASS_EPOCHS):
        model.train()
        total_loss_epoch = 0.0
        spec_loss_epoch = 0.0
        adv_loss_epoch = 0.0
        orth_loss_epoch = 0.0
        consist_loss_epoch = 0.0
        n_batches = 0
        gcn_center_epoch = 0.0
        gcn_margin_epoch = 0.0
        gcn_smooth_epoch = 0.0

        pbar = tqdm(
            sampler,
            total=len(sampler),
            desc=f"Epoch {epoch+1}/{BYPASS_EPOCHS} [Bypass Training]",
            leave=True,
            dynamic_ncols=True,
        )
        for bi, batch_indices in enumerate(pbar):
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
            gcn_center = extras['gcn_center']
            gcn_margin = extras['gcn_margin']
            gcn_smooth = extras['gcn_smooth']
            # 将图平滑视为一致性项（图内一致性替代跨批一致性）
            consist_loss = gcn_smooth

            # 正交约束：来自FootprintExpert
            fp_enh: FootprintExpert = model.fp_enh
            fp_pr: FootprintExpert = model.fp_pr
            orth_loss = fp_enh.orthogonality_loss(extras['zG'], extras['zF'], extras['zI']) 
            orth_loss = orth_loss + fp_pr.orthogonality_loss(extras['zG'], extras['zF'], extras['zI']) * 0.0  # 已合并z，避免重复加重


            total_loss = (
                BYPASS_SPEC_WEIGHT * spec_loss +
                BYPASS_INV_WEIGHT * adv_loss +
                BYPASS_ORTHO_WEIGHT * orth_loss +
                BYPASS_CONSIST_WEIGHT * consist_loss +
                GCN_CENTER_LOSS_W * gcn_center + GCN_MARGIN_LOSS_W * gcn_margin + GCN_SMOOTH_LOSS_W * gcn_smooth
            )
            # 额外一致性信号：在RoPE后池化的序列表征保持一致（跨细胞）
            # 注意：该项权重较小，仅作为补充
            he = extras.get('zG')  # 使用共性子空间代表序列语义
            # 若未来需要，可返回池化后的序列向量进行一致性约束

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tl = total_loss.item()
            sl = spec_loss.item()
            il = adv_loss.item()
            ol = orth_loss.item()
            cl = consist_loss.item()
            total_loss_epoch += tl
            spec_loss_epoch += sl
            adv_loss_epoch += il
            orth_loss_epoch += ol
            consist_loss_epoch += cl
            gcn_center_epoch += float(gcn_center.item())
            gcn_margin_epoch += float(gcn_margin.item())
            gcn_smooth_epoch += float(gcn_smooth.item())
            n_batches += 1
            pbar.set_postfix({
                'total': f"{tl:.4f}",
                'spec': f"{sl:.4f}",
                'adv': f"{il:.4f}",
                'orth': f"{ol:.4f}",
                'cons': f"{cl:.4f}",
                'g_center': f"{gcn_center.item():.4f}",
                'g_margin': f"{gcn_margin.item():.4f}",
                'g_smooth': f"{gcn_smooth.item():.4f}",
            })
            # 使用完整数据集，无批次上限，保持与PRISM一致

        # 记录与绘图
        avg_total = total_loss_epoch / max(1, n_batches)
        history['total'].append(avg_total)
        history['spec'].append(spec_loss_epoch / max(1, n_batches))
        history['adv'].append(adv_loss_epoch / max(1, n_batches))
        history['orth'].append(orth_loss_epoch / max(1, n_batches))
        history['consist'].append(consist_loss_epoch / max(1, n_batches))
        history['gcn_center'].append(gcn_center_epoch / max(1, n_batches))
        history['gcn_margin'].append(gcn_margin_epoch / max(1, n_batches))
        history['gcn_smooth'].append(gcn_smooth_epoch / max(1, n_batches))
        plot_epoch_curves(out_dir, history, epoch + 1)

        # 深度可视化：G/F子空间的t-SNE分布与F分类的混淆矩阵
        with torch.no_grad():
            # 采样若干批次收集 zG/zF 与细胞标签
            zg_list: List[torch.Tensor] = []
            zf_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            steps_collected = 0
            for batch_indices in sampler:
                if steps_collected >= 10:
                    break
                batch = [dataset[i] for i in batch_indices]
                from PRISM import prism_collate_fn
                enh_ids, pr_ids, cell_lines, labels_t = prism_collate_fn(batch)
                cell_t = build_cell_label_tensor(cell_lines, all_cells).to(DEVICE)
                enh_ids = enh_ids.to(DEVICE)
                pr_ids = pr_ids.to(DEVICE)
                _, ex = model(enh_ids, pr_ids, cell_labels=cell_t)
                zg_list.append(ex['zG'])
                zf_list.append(ex['zF'])
                y_list.append(cell_t)
                steps_collected += 1
            if len(zg_list) > 0:
                ZG = _safe_np(torch.cat(zg_list, dim=0))
                ZF = _safe_np(torch.cat(zf_list, dim=0))
                Y = _safe_np(torch.cat(y_list, dim=0))
                # 下采样避免t-SNE过慢
                nG = ZG.shape[0]
                nF = ZF.shape[0]
                selG = np.random.choice(nG, size=min(600, nG), replace=False)
                selF = np.random.choice(nF, size=min(600, nF), replace=False)
                plot_tsne(out_dir, epoch+1, ZG[selG], Y[selG], 't-SNE G (cell)')
                plot_tsne(out_dir, epoch+1, ZF[selF], Y[selF], 't-SNE F (cell)')
            # 使用一批绘制F分类混淆矩阵
            for batch_indices in sampler:
                batch = [dataset[i] for i in batch_indices]
                from PRISM import prism_collate_fn
                enh_ids, pr_ids, cell_lines, labels_t = prism_collate_fn(batch)
                cell_t = build_cell_label_tensor(cell_lines, all_cells).to(DEVICE)
                enh_ids = enh_ids.to(DEVICE)
                pr_ids = pr_ids.to(DEVICE)
                _, ex = model(enh_ids, pr_ids, cell_labels=cell_t)
                y_true = _safe_np(cell_t)
                y_pred = _safe_np(ex['spec_logits'].argmax(dim=-1))
                plot_confusion(out_dir, epoch+1, y_true, y_pred, 'Confusion (cell by F)')
                break


if __name__ == '__main__':
    main()
