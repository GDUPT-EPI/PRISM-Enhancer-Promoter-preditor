"""
旁路解耦训练入口（main）

设计说明：
- 采用 `AuxiliaryModel` 获取子空间 `[zG, zF, zI]` 与整体特征 `M'`；
- 一致性约束来自图平滑（GCN拉普拉斯），替代基于批次的噪声一致性；
- 训练流程遵循《解耦对抗算子.md》，包含特性鉴别、域对抗、正交约束与图一致性；
- 支持从 `save_model/bypass/aux_epoch_5.pth` 导入旁路权重继续训练；
- 严格遵循集中配置与统一词表管理（6-mer，词表大小4100）。
"""
from models.pleat.embedding import KMerTokenizer
from config import *
from config import PRISM_SAVE_MODEL_DIR, PRISM_BATCH_SIZE
from data_loader import load_prism_data, PRISMDataset, RandomBatchSampler
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import re
import xml.etree.ElementTree as ET
from models.AuxiliaryModel import AuxiliaryModel
from models.layers.footprint import FootprintExpert

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


# 配置日志系统
def setup_logging():
    """
    配置日志系统

    Returns:
        logging.Logger: 已初始化的中文日志记录器
    """
    log_filename = os.path.join(LOG_DIR, f"prism_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")  # 日志文件路径
    log_format = '%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    logging.basicConfig(  # 基础配置
        level=logging.INFO,  # 设置日志级别
        format=log_format,  # 设置日志格式
        handlers=[  # 处理器列表
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
    """
    PRISM批处理拼接函数（统一词表管理，6-mer，词表大小4100）

    输入项: `(enhancer_seq, promoter_seq, cell_line, label)`

    Returns:
        `(enh_ids, pr_ids, cell_lines, labels)`
    """

    
    enhancer_seqs = [item[0] for item in batch]  # 增强子序列
    promoter_seqs = [item[1] for item in batch]  # 启动子序列
    cell_lines = [item[2] for item in batch]  # 细胞系名称
    labels = [item[3] for item in batch]  # 标签
    
    # 创建tokenizer (只创建一次)
    if not hasattr(prism_collate_fn, 'tokenizer'):  # 检查是否有tokenizer属性
        prism_collate_fn.tokenizer = KMerTokenizer()  # 创建tokenizer
    
    tokenizer = prism_collate_fn.tokenizer  # 获取tokenizer
    
    # 将DNA序列转换为token IDs
    enhancer_ids_list = [tokenizer.encode(seq) for seq in enhancer_seqs]  # 编码增强子
    promoter_ids_list = [tokenizer.encode(seq) for seq in promoter_seqs]  # 编码启动子

    K = CNN_KERNEL_SIZE  # 卷积核大小
    P = POOL_KERNEL_SIZE  # 池化核大小
    pad_id = DNA_EMBEDDING_PADDING_IDX  # PAD索引
    min_req = K + P - 1  # 最小需求长度

    max_len_en = max(int(x.size(0)) for x in enhancer_ids_list) if enhancer_ids_list else min_req  # 增强子最大长度
    max_len_pr = max(int(x.size(0)) for x in promoter_ids_list) if promoter_ids_list else min_req  # 启动子最大长度

    adj_base_en = max(1, max_len_en - (K - 1))  # 调整增强子基础长度
    adj_base_pr = max(1, max_len_pr - (K - 1))  # 调整启动子基础长度
    target_len_en = (K - 1) + ((adj_base_en + P - 1) // P) * P  # 目标增强子长度
    target_len_pr = (K - 1) + ((adj_base_pr + P - 1) // P) * P  # 目标启动子长度
    target_len_en = max(min_req, min(target_len_en, MAX_ENHANCER_LENGTH))  # 限制增强子长度
    target_len_pr = max(min_req, min(target_len_pr, MAX_PROMOTER_LENGTH))  # 限制启动子长度

    processed_en = []  # 处理后的增强子序列
    for ids in enhancer_ids_list:  # 遍历增强子ID列表
        L = int(ids.size(0))  # 获取长度
        if L > target_len_en:  # 如果长度超过目标
            s = (L - target_len_en) // 2  # 计算起始位置
            ids = ids[s:s + target_len_en]  # 截取序列
        processed_en.append(ids)  # 添加到处理列表
    processed_pr = []  # 处理后的启动子序列
    for ids in promoter_ids_list:  # 遍历启动子ID列表
        L = int(ids.size(0))  # 获取长度
        if L > target_len_pr:  # 如果长度超过目标
            s = (L - target_len_pr) // 2  # 计算起始位置
            ids = ids[s:s + target_len_pr]  # 截取序列
        processed_pr.append(ids)  # 添加到处理列表

    padded_enhancer_ids = pad_sequence(processed_en, batch_first=True, padding_value=pad_id)  # 填充增强子序列
    padded_promoter_ids = pad_sequence(processed_pr, batch_first=True, padding_value=pad_id)  # 填充启动子序列

    if padded_enhancer_ids.size(1) < target_len_en:  # 如果增强子序列长度不足
        padded_enhancer_ids = torch.nn.functional.pad(padded_enhancer_ids, (0, target_len_en - padded_enhancer_ids.size(1)), value=pad_id)  # 填充到目标长度
    if padded_promoter_ids.size(1) < target_len_pr:  # 如果启动子序列长度不足
        padded_promoter_ids = torch.nn.functional.pad(padded_promoter_ids, (0, target_len_pr - padded_promoter_ids.size(1)), value=pad_id)  # 填充到目标长度
    
    labels_tensor = torch.tensor(labels, dtype=torch.float)  # 标签张量
    
    return padded_enhancer_ids, padded_promoter_ids, cell_lines, labels_tensor  # 返回处理后的数据


# 过时随机掩码逻辑已移除


# 过时的MLM训练流程已移除

def _find_latest_epoch(save_dir: str) -> int:  # 查找最新epoch
    if not os.path.exists(save_dir):  # 如果目录不存在
        return 0  # 返回0
    epochs = []  # epoch列表
    for name in os.listdir(save_dir):  # 遍历目录
        m1 = re.match(r"prism_epoch_(\d+)\.pth$", name)  # 匹配模型文件
        m2 = re.match(r"prism_full_epoch_(\d+)\.pt$", name)  # 匹配完整状态文件
        if m1:  # 如果匹配模型文件
            epochs.append(int(m1.group(1)))  # 添加epoch
        elif m2:  # 如果匹配完整状态文件
            epochs.append(int(m2.group(1)))  # 添加epoch
    return max(epochs) if epochs else 0  # 返回最大epoch或0

def _load_resume_state(save_dir: str, device: torch.device, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: ReduceLROnPlateau):  # 加载恢复状态
    latest_epoch = _find_latest_epoch(save_dir)  # 查找最新epoch
    if latest_epoch <= 0:
        return 0
    full_path = os.path.join(save_dir, f"prism_full_epoch_{latest_epoch}.pt")
    model_path = os.path.join(save_dir, f"prism_epoch_{latest_epoch}.pth")
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
    else:
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=device)
            if isinstance(sd, dict):
                if 'backbone' in sd:
                    model.load_state_dict(sd['backbone'], strict=False)
                else:
                    model.load_state_dict(sd, strict=False)
            logger.info(f"加载模型检查点: {model_path} (epoch={latest_epoch})")
    return latest_epoch


def build_cell_label_tensor(cells, all_cells):
    """
    将细胞系名称映射为索引张量

    Args:
        cells (list[str]): 当前批次的细胞系名称
        all_cells (list[str]): 训练集或XML定义的全部细胞系

    Returns:
        torch.Tensor: 细胞标签索引，形状 `[B]`
    """
    name_to_idx = {c: i for i, c in enumerate(all_cells)}
    idxs = [name_to_idx.get(x, 0) for x in cells]
    return torch.tensor(idxs, dtype=torch.long)


def main():  # 主函数
    """
    入口函数：加载旁路模型权重并进行解耦对抗训练

    流程遵循《解耦对抗算子.md》：
    - 使用 `AuxiliaryModel`，得到子空间 `[zG, zF, zI]` 与整体特征 `M'`
    - 采用特性鉴别损失(F→cell)、域对抗损失(GRL(G)→cell)、正交约束与图一致性约束(GCN平滑)
    - 从 `save_model/bypass/aux_epoch_5.pth` 导入权重继续训练
    """
    logger.info("=" * 80)
    logger.info("旁路解耦训练开始 (加载已保存的旁路模型权重)")
    logger.info("=" * 80)

    # 加载PRISM特供数据（作为旁路训练的序列来源）
    logger.info("加载训练数据 (domain-kl)...")
    train_pairs_df, train_e_seqs, train_p_seqs = load_prism_data("train")
    logger.info(f"训练样本数: {len(train_pairs_df)}")
    logger.info(f"训练细胞系: {', '.join(sorted(train_pairs_df['cell_line'].unique()))}")

    unique_cells_train = sorted(train_pairs_df['cell_line'].unique())

    # 创建数据集与加载器
    train_dataset = PRISMDataset(train_pairs_df, train_e_seqs, train_p_seqs)
    train_sampler = RandomBatchSampler(train_dataset, batch_size=BYPASS_BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=prism_collate_fn,
    )

    # 细胞类型集合（优先读取XML，如无则使用训练集出现的细胞）
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
    num_cells = len(fixed_cells)

    # 构建旁路模型并加载权重
    model = AuxiliaryModel(num_cell_types=num_cells).to(device)
    ckpt_dir = os.path.join(PROJECT_ROOT, 'save_model', 'bypass')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'aux_epoch_5.pth')
    start_epoch_offset = 0
    if os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location=device)
            if isinstance(sd, dict) and 'auxiliary' in sd:
                model.load_state_dict(sd['auxiliary'], strict=False)
                logger.info(f"已加载旁路权重: {ckpt_path} (key='auxiliary')")
            elif isinstance(sd, dict):
                model.load_state_dict(sd, strict=False)
                logger.info(f"已加载旁路权重: {ckpt_path} (直接state_dict)")
            else:
                logger.info(f"旁路权重格式非dict，跳过: {ckpt_path}")
            start_epoch_offset = 5
        except Exception as e:
            logger.info(f"加载旁路权重失败: {e}")
    else:
        logger.info("未找到 aux_epoch_5.pth，使用随机初始化继续训练")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=BYPASS_LEARNING_RATE, weight_decay=BYPASS_WEIGHT_DECAY)
    logger.info(f"批量大小: {BYPASS_BATCH_SIZE}")
    logger.info(f"训练轮数: {BYPASS_EPOCHS}")
    logger.info(f"学习率: {BYPASS_LEARNING_RATE}")

    # 训练循环（遵循解耦对抗算子）
    for epoch_idx in range(BYPASS_EPOCHS):
        model.train()
        total_loss_epoch = 0.0
        spec_loss_epoch = 0.0
        adv_loss_epoch = 0.0
        orth_loss_epoch = 0.0
        consist_loss_epoch = 0.0
        gcn_center_epoch = 0.0
        gcn_margin_epoch = 0.0
        gcn_smooth_epoch = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {start_epoch_offset+epoch_idx+1}/{start_epoch_offset+BYPASS_EPOCHS} [Bypass]", leave=True, dynamic_ncols=True)
        for batch in pbar:
            enh_ids, pr_ids, cell_lines, labels_t = batch
            enh_ids = enh_ids.to(device)
            pr_ids = pr_ids.to(device)
            cell_t = build_cell_label_tensor(cell_lines, fixed_cells).to(device)

            optimizer.zero_grad()
            pred_prob, extras = model(enh_ids, pr_ids, cell_labels=cell_t)

            spec_loss = extras['spec_loss']
            adv_loss = extras['adv_loss']
            gcn_center = extras['gcn_center']
            gcn_margin = extras['gcn_margin']
            gcn_smooth = extras['gcn_smooth']
            consist_loss = gcn_smooth  # 图一致性替代批次一致性

            # 正交约束
            fp_enh: FootprintExpert = model.fp_enh
            fp_pr: FootprintExpert = model.fp_pr
            orth_loss = fp_enh.orthogonality_loss(extras['zG'], extras['zF'], extras['zI'])
            orth_loss = orth_loss + fp_pr.orthogonality_loss(extras['zG'], extras['zF'], extras['zI']) * 0.0

            total_loss = (
                BYPASS_SPEC_WEIGHT * spec_loss +
                BYPASS_INV_WEIGHT * adv_loss +
                BYPASS_ORTHO_WEIGHT * orth_loss +
                BYPASS_CONSIST_WEIGHT * consist_loss +
                GCN_CENTER_LOSS_W * gcn_center +
                GCN_MARGIN_LOSS_W * gcn_margin +
                GCN_SMOOTH_LOSS_W * gcn_smooth
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

            tl = float(total_loss.item())
            sl = float(spec_loss.item())
            il = float(adv_loss.item())
            ol = float(orth_loss.item())
            cl = float(consist_loss.item())
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

        # 保存旁路模型权重（延续编号）
        save_dir = os.path.join(PROJECT_ROOT, 'save_model', 'bypass')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"aux_epoch_{start_epoch_offset+epoch_idx+1}.pth")
        torch.save({'auxiliary': model.state_dict()}, save_path)
        logger.info(f"保存旁路权重: {save_path}")

        # 汇总日志
        avg_total = total_loss_epoch / max(1, n_batches)
        avg_spec = spec_loss_epoch / max(1, n_batches)
        avg_adv = adv_loss_epoch / max(1, n_batches)
        avg_orth = orth_loss_epoch / max(1, n_batches)
        avg_cons = consist_loss_epoch / max(1, n_batches)
        avg_gc = gcn_center_epoch / max(1, n_batches)
        avg_gm = gcn_margin_epoch / max(1, n_batches)
        avg_gs = gcn_smooth_epoch / max(1, n_batches)
        logger.info(
            f"Epoch {start_epoch_offset+epoch_idx+1}/{start_epoch_offset+BYPASS_EPOCHS} - "
            f"total={avg_total:.4f}, spec={avg_spec:.4f}, adv={avg_adv:.4f}, orth={avg_orth:.4f}, cons={avg_cons:.4f}, "
            f"g_center={avg_gc:.4f}, g_margin={avg_gm:.4f}, g_smooth={avg_gs:.4f}"
        )

    logger.info("=" * 80)
    logger.info("旁路解耦训练完成")
    logger.info("=" * 80)


if __name__ == "__main__":  # 如果是主程序
    main()  # 执行主函数
