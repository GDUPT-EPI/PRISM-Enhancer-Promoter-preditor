from models.pleat.embedding import KMerTokenizer
from config import *
from config import PRISM_SAVE_MODEL_DIR, PRISM_BATCH_SIZE
from data_loader import load_prism_data, PRISMDataset, RandomBatchSampler
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
    """PRISM批处理拼接函数

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


# 过时的MLM验证流程已移除


# 过时的可分性验证流程已移除


def main():  # 主函数
    """入口函数"""
    logger.info("=" * 80)  # 分隔线
    logger.info("PRISM预训练开始 (Domain-KL数据)")  # 记录日志
    logger.info("=" * 80)  # 分隔线
    
    # 加载PRISM特供数据
    logger.info("加载训练数据 (domain-kl)...")  # 记录日志
    train_pairs_df, train_e_seqs, train_p_seqs = load_prism_data("train")  # 加载训练数据
    logger.info(f"训练样本数: {len(train_pairs_df)}")  # 记录日志
    logger.info(f"训练细胞系: {', '.join(sorted(train_pairs_df['cell_line'].unique()))}")  # 记录日志
    
    unique_cells_train = sorted(train_pairs_df['cell_line'].unique())  # 获取唯一细胞系
    
    # 创建数据集
    train_dataset = PRISMDataset(train_pairs_df, train_e_seqs, train_p_seqs)  # 创建训练数据集
    
    train_sampler = RandomBatchSampler(train_dataset, batch_size=PRISM_BATCH_SIZE, shuffle=True)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")  # 记录日志
    train_loader = DataLoader(  # 创建数据加载器
        dataset=train_dataset,  # 数据集
        batch_sampler=train_sampler,  # 采样器
        num_workers=NUM_WORKERS,  # 工作进程数
        pin_memory=True,  # 固定内存
        collate_fn=prism_collate_fn,  # 批处理函数
    )
    
    val_loader = None  # 验证加载器为空
    
    # 创建模型
    logger.info("创建PRISM模型...")  # 记录日志
    xml_path = os.path.join(PROJECT_ROOT, "vocab", "cell_type.xml")  # 细胞类型XML路径
    def load_cell_types(path: str):  # 加载细胞类型函数
        if os.path.exists(path):  # 如果文件存在
            try:  # 尝试解析
                root = ET.parse(path).getroot()  # 解析XML
                names = []  # 名称列表
                for node in root.findall(".//type"):  # 遍历类型节点
                    name = node.get("name")  # 获取名称属性
                    if name:  # 如果名称不为空
                        names.append(name.strip())  # 去除空白并收集
                names = [n for n in names if n]  # 过滤空名称
                if names:  # 如果有名称
                    return names  # 返回名称列表
            except Exception:  # 捕获异常
                pass  # 忽略
        return []  # 文件不存在或解析失败时返回空列表
    fixed_cells = load_cell_types(xml_path)  # 加载固定细胞类型
    if not fixed_cells:  # 如果没有固定细胞类型
        fixed_cells = unique_cells_train  # 使用训练数据中的唯一细胞类型
    label_map = {c: i for i, c in enumerate(fixed_cells)}  # 创建标签映射
    other_id = label_map.get("OTHER", None)  # 获取OTHER标签ID
    num_cells = len(fixed_cells)  # 细胞类型数量
    model = PRISMBackbone(num_classes=num_cells).to(device)  # 创建模型
    model = model.to(device)  # 移动到设备
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())  # 总参数数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数数
    logger.info(f"模型总参数: {total_params:,}")  # 记录日志
    logger.info(f"可训练参数: {trainable_params:,}")  # 记录日志
    logger.info(f"GPU可用: {torch.cuda.is_available()}")  # 记录日志
    logger.info(f"模型在GPU上: {next(model.parameters()).is_cuda}")  # 记录日志
    
    # 创建优化器和调度器
    cell_label_map = label_map  # 细胞标签映射

    start_epoch = 0  # 起始epoch

    optimizer = torch.optim.AdamW(list(model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # 创建优化器
    total_steps = len(train_loader) * EPOCH  # 总步数
    scheduler = None  # 调度器为空
    
    logger.info(f"批量大小: {PRISM_BATCH_SIZE} (纯随机批次)")
    logger.info(f"训练轮数: {EPOCH}")  # 记录日志
    logger.info(f"学习率: {LEARNING_RATE}")  # 记录日志
    logger.info(f"总训练步数: {total_steps}")  # 记录日志
    
    # 训练循环
    logger.info("=" * 80)  # 分隔线
    logger.info("开始训练")  # 记录日志
    logger.info("=" * 80)  # 分隔线
    
    os.makedirs(PRISM_SAVE_MODEL_DIR, exist_ok=True)  # 创建模型保存目录
    start_epoch = _load_resume_state(PRISM_SAVE_MODEL_DIR, device, model, optimizer, scheduler)  # 加载恢复状态
    if start_epoch > 0:  # 如果有起始epoch
        logger.info(f"从最近权重恢复: epoch {start_epoch}")  # 记录日志
    else:  # 如果没有起始epoch
        logger.info("未发现可恢复检查点，将从头开始训练")  # 记录日志
    if start_epoch >= EPOCH:  # 如果起始epoch大于等于目标epoch
        logger.info("已达到或超过目标训练轮数，无需继续。若需追加训练，请增大EPOCH或删除旧检查点。")  # 记录日志
    for epoch_idx in range(start_epoch, EPOCH):  # 遍历epoch
        # 训练
        logger.info("Loss Weights: ep=1.0")  # 记录日志
        model.train()  # 设置为训练模式
        total_loss = 0.0; total_ep_acc = 0.0; n_batches = 0  # 初始化统计变量
        total_tp = 0; total_fp = 0; total_fn = 0  # 初始化TP、FP、FN
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{EPOCH} [Training]", leave=True, dynamic_ncols=True)  # 创建进度条
        for batch in pbar:  # 遍历批次
            enh_ids, pr_ids, cell_lines, labels = batch  # 获取批次数据
            enh_ids = enh_ids.to(device); pr_ids = pr_ids.to(device)  # 移动到设备
            # 暂停随机PAD掩码，避免训练初期数值不稳
            # enh_ids = apply_random_mask(enh_ids)
            # pr_ids = apply_random_mask(pr_ids)
            labels = labels.to(device)  # 移动到设备
            precision = 0.0; recall = 0.0; f1 = 0.0  # 初始化精确率、召回率、F1
            ep_outputs, adaptive_loss = model(enh_ids, pr_ids)  # 前向传播
            ep_outputs = ep_outputs.squeeze(-1)  # 压缩维度
            ep_loss, loss_details = model.compute_loss(ep_outputs, labels.float(), adaptive_loss, return_details=True)  # 计算损失
            with torch.no_grad():  # 不计算梯度
                ep_preds = (ep_outputs >= 0.5).long()  # 预测结果
                ep_acc = (ep_preds == labels.long()).float().mean().item()  # 准确率
                tp = int(((ep_preds == 1) & (labels.long() == 1)).sum().item())  # 真正例
                fp = int(((ep_preds == 1) & (labels.long() == 0)).sum().item())  # 假正例
                fn = int(((ep_preds == 0) & (labels.long() == 1)).sum().item())  # 假负例
                total_tp += tp; total_fp += fp; total_fn += fn  # 累加
                precision = (tp / max(tp + fp, 1)) if (tp + fp) > 0 else 0.0  # 计算精确率
                recall = (tp / max(tp + fn, 1)) if (tp + fn) > 0 else 0.0  # 计算召回率
                f1 = (2 * precision * recall / max(precision + recall, 1e-6)) if (precision + recall) > 0 else 0.0  # 计算F1
            loss = ep_loss  # 损失
            optimizer.zero_grad();  # 清空梯度
            loss.backward();  # 反向传播
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=GRAD_CLIP_MAX_NORM);  # 梯度裁剪
            optimizer.step();  # 更新参数

            total_loss += loss.item(); total_ep_acc += ep_acc; n_batches += 1  # 累加统计
            pbar.set_postfix({  # 设置进度条后缀
                'loss': f'{loss.item():.4f}',  # 损失
                # 'base': f"{loss_details['base']:.4f}",  # 基础损失
                # 'adaptive': f"{loss_details['adaptive']:.4f}",  # 自适应损失
                # 'penalty': f"{loss_details['penalty']:.4f}",  # 惩罚损失
                'ep_acc': f'{ep_acc:.4f}',  # 准确率
                'prec': f'{precision:.4f}',  # 精确率
                'rec': f'{recall:.4f}',  # 召回率
                'f1': f'{f1:.4f}',  # F1
            })

        avg_loss = total_loss / max(n_batches, 1)  # 平均损失
        avg_ep_acc = total_ep_acc / max(n_batches, 1)  # 平均准确率
        epoch_precision = (total_tp / max(total_tp + total_fp, 1)) if (total_tp + total_fp) > 0 else 0.0  # epoch精确率
        epoch_recall = (total_tp / max(total_tp + total_fn, 1)) if (total_tp + total_fn) > 0 else 0.0  # epoch召回率
        epoch_f1 = (2 * epoch_precision * epoch_recall / max(epoch_precision + epoch_recall, 1e-6)) if (epoch_precision + epoch_recall) > 0 else 0.0  # epoch F1
        logger.info(f"Epoch {epoch_idx+1}/{EPOCH} - Train Loss: {avg_loss:.4f}, EP Acc: {avg_ep_acc:.4f}, Prec: {epoch_precision:.4f}, Rec: {epoch_recall:.4f}, F1: {epoch_f1:.4f}")  # 记录日志
        
        # 保存检查点
        checkpoint_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_epoch_{epoch_idx+1}.pth")  # 检查点路径
        torch.save({'backbone': model.state_dict()}, checkpoint_path)  # 保存模型
        logger.info(f"保存检查点: {checkpoint_path}")  # 记录日志
        full_state_path = os.path.join(PRISM_SAVE_MODEL_DIR, f"prism_full_epoch_{epoch_idx+1}.pt")  # 完整状态路径
        full_state = {  # 完整状态
            'model_state': model.state_dict(),  # 模型状态
            'optimizer_state': optimizer.state_dict(),  # 优化器状态
            'epoch': epoch_idx + 1,  # epoch
        }
        if scheduler is not None:  # 如果调度器不为空
            full_state['scheduler_state'] = scheduler.state_dict()  # 添加调度器状态
        torch.save(full_state, full_state_path)  # 保存完整状态
        logger.info(f"保存完整状态: {full_state_path}")  # 记录日志

        # 移除验证与知识库保存流程
    
    logger.info("=" * 80)  # 分隔线
    logger.info("PRISM预训练完成")  # 记录日志
    logger.info("=" * 80)  # 分隔线


if __name__ == "__main__":  # 如果是主程序
    main()  # 执行主函数
