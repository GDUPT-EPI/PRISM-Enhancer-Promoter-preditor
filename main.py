#!/usr/bin/env python3

# 导入配置文件
from config import *
# 导入新的数据加载模块
from data_loader import load_all_val_data, load_all_train_data

# 导入日志模块
import logging
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from data_loader import MyDataset

# 导入优化的数据预处理模块
from models.pleat.optimized_pre import (
    create_optimized_dataset
)
# 导入词表管理模块
from models.pleat.vocab_utils import (
    get_token_to_idx,
    get_vocab_hash,
    VocabManager
)
from models.EPIModel import EPIModel
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR
import torch
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# 使用配置文件中的参数
# BATCH_SIZE = 32  # 已在config.py中定义
# epoch = 15  # 已在config.py中定义为EPOCH
# pre_train_epoch = 10  # 已在config.py中定义为PRE_TRAIN_EPOCH

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# 配置日志系统
def setup_logging():
    """
    配置日志系统，同时输出到控制台和文件
    """
    # 创建日志文件名，包含时间戳
    log_filename = os.path.join(LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# 设置日志
logger = setup_logging()
logger.info("日志系统已初始化，日志文件将保存到: {}".format(LOG_DIR))
logger.info("预处理配置: 使用 {} 个线程".format(PREPROCESS_NUM_THREADS))



def get_num_correct(preds, labels):
    """
    计算预测正确的样本数量
    
    Args:
        preds: 模型预测值
        labels: 真实标签
        
    Returns:
        预测正确的样本数量
    """
    predictions = (preds >= 0.5).float()
    return (predictions == labels).sum().item()


def load_dna_sequences(file_path):
    """
    从文件加载DNA序列数据
    
    Args:
        file_path: DNA序列文件路径
        
    Returns:
        DNA序列的NumPy数组
    """
    # 读取DNA序列文件
    with open(file_path, 'r') as file:
        # 逐行读取文件并保存为字符串列表
        sequences = [line.strip() for line in file]

    # 将字符串列表转换为NumPy数组
    sequences_array = np.array(sequences)

    return sequences_array


def load_labels(file_path):
    """
    从文件加载标签数据
    
    Args:
        file_path: 标签文件路径
        
    Returns:
        标签的NumPy数组
    """
    # 读取标签文件
    with open(file_path, 'r') as file:
        # 逐行读取文件并保存为整数列表
        labels = [int(line.strip()) for line in file]

    # 将整数列表转换为NumPy数组
    labels_array = np.array(labels)

    return labels_array


def simple_collate_fn(batch):
    enhancer_sequences = [item[0] for item in batch]
    promoter_sequences = [item[1] for item in batch]
    enhancer_features = [item[2] for item in batch]
    promoter_features = [item[3] for item in batch]
    labels = [item[4] for item in batch]

    K = CNN_KERNEL_SIZE
    P = POOL_KERNEL_SIZE
    pad_id = DNA_EMBEDDING_PADDING_IDX
    min_req = K + P - 1

    max_len_en = max(int(x.size(0)) for x in enhancer_sequences) if enhancer_sequences else min_req
    max_len_pr = max(int(x.size(0)) for x in promoter_sequences) if promoter_sequences else min_req

    adj_base_en = max(1, max_len_en - (K - 1))
    adj_base_pr = max(1, max_len_pr - (K - 1))
    target_len_en = (K - 1) + ((adj_base_en + P - 1) // P) * P
    target_len_pr = (K - 1) + ((adj_base_pr + P - 1) // P) * P
    target_len_en = max(min_req, min(target_len_en, MAX_ENHANCER_LENGTH))
    target_len_pr = max(min_req, min(target_len_pr, MAX_PROMOTER_LENGTH))

    processed_en = []
    for ids in enhancer_sequences:
        L = int(ids.size(0))
        if L > target_len_en:
            s = (L - target_len_en) // 2
            ids = ids[s:s + target_len_en]
        processed_en.append(ids)
    processed_pr = []
    for ids in promoter_sequences:
        L = int(ids.size(0))
        if L > target_len_pr:
            s = (L - target_len_pr) // 2
            ids = ids[s:s + target_len_pr]
        processed_pr.append(ids)

    padded_enhancer_sequences = pad_sequence(processed_en, batch_first=True, padding_value=pad_id)
    padded_promoter_sequences = pad_sequence(processed_pr, batch_first=True, padding_value=pad_id)

    if padded_enhancer_sequences.size(1) < target_len_en:
        padded_enhancer_sequences = torch.nn.functional.pad(
            padded_enhancer_sequences, (0, target_len_en - padded_enhancer_sequences.size(1)), value=pad_id
        )
    if padded_promoter_sequences.size(1) < target_len_pr:
        padded_promoter_sequences = torch.nn.functional.pad(
            padded_promoter_sequences, (0, target_len_pr - padded_promoter_sequences.size(1)), value=pad_id
        )

    padded_enhancer_features = torch.stack(enhancer_features)
    padded_promoter_features = torch.stack(promoter_features)
    labels = torch.tensor(labels, dtype=torch.float)

    return padded_enhancer_sequences, padded_promoter_sequences, padded_enhancer_features, padded_promoter_features, labels


class OptimizedCombinedDataset(Dataset):
    """
    优化的组合数据集类，使用新的预处理模块
    """
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


def val_forwrd(model, dataloader, cell_name=""):
    """
    模型验证函数，计算验证集上的损失和指标
    
    Args:
        model: 要验证的模型
        dataloader: 验证数据加载器
        cell_name: 细胞系名称，用于进度条显示
        
    Returns:
        tuple: 包含三个元素的元组
            - test_epoch_loss: 验证集总损失
            - test_epoch_aupr: 验证集AUPR值
            - test_epoch_auc: 验证集AUC值
    """
    model.eval()
    test_epoch_loss = 0.0
    test_epoch_correct = 0
    test_epoch_preds = torch.tensor([]).to(device)
    test_epoch_target = torch.tensor([]).to(device)
    
    # 使用tqdm显示验证进度
    val_pbar = tqdm(dataloader, desc=f"Validation [{cell_name}]", 
                   leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for data in val_pbar:
            enhancer_ids, promoter_ids, enhancer_features, promoter_features, labels = data
            enhancer_ids = enhancer_ids.to(device, non_blocking=True)
            promoter_ids = promoter_ids.to(device, non_blocking=True)
            enhancer_features = enhancer_features.to(device, non_blocking=True)
            promoter_features = promoter_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
                
            outputs, _, _ = model(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
            labels = labels.unsqueeze(1).float()
            test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

            if labels.shape == torch.Size([1, 1]):
                labels = torch.reshape(labels, (1,))
                
            loss = model.criterion(outputs, labels)
            test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
            test_epoch_loss += loss.item()
            test_epoch_correct += get_num_correct(outputs, labels)
            
            # 更新进度条显示当前损失
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
    test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    test_epoch_acc = test_epoch_correct / max(1, test_epoch_target.numel())
    return test_epoch_loss, test_epoch_aupr, test_epoch_auc, test_epoch_acc


#得到的是序列数据
#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

train_data = load_all_train_data()
val_data = load_all_val_data()

# 检查模型保存路径是否存在
if not os.path.exists(SAVE_MODEL_DIR):
    os.makedirs(SAVE_MODEL_DIR)

# 创建新的模型实例（不使用预训练权重）
epimodel = EPIModel()
epimodel = epimodel.to(device)
print("创建新的模型实例（不使用预训练权重，使用CBAT模块）")

# 创建优化器
optimizer = torch.optim.Adam(epimodel.parameters(), lr=LEARNING_RATE)
epimodel.optimizer = optimizer

# 设置学习率调度器
scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)

# 检查GPU是否可用
all_params_on_gpu = all(param.is_cuda for param in epimodel.parameters())
print("GPU是否可用：", torch.cuda.is_available())
print("模型是否在GPU上：", all_params_on_gpu)
print(f"训练细胞系列表：{', '.join(sorted(train_data.keys()))}")
print(f"验证细胞系列表：{', '.join(sorted(val_data.keys()))}")
print(f"批量大小：{BATCH_SIZE}")
print(f"数据加载器工作进程数：{NUM_WORKERS}")
    

# 创建优化的数据集 - 使用新的预处理模块
print("创建训练数据集加载器...")
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
        cache_dir=os.path.join(CACHE_DIR, f"train_cache_{cell}"),
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

# 确保总是创建 ALL 数据加载器，即使 train_folds 为空
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
else:
    # 如果没有训练数据，创建一个空的数据加载器以避免 KeyError
    print("警告：没有找到训练数据，创建空的 ALL 数据加载器")
    empty_dataset = MyDataset(np.array([]), np.array([]), np.array([]))
    train_loaders["ALL"] = DataLoader(
        dataset=empty_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 没有数据时不需要工作进程
        pin_memory=False,
        collate_fn=simple_collate_fn,
    )
# 创建验证数据集
def create_optimized_validation_dataset(enhancers, promoters, labels, cell_name):
    """创建优化的验证数据集"""
    return OptimizedCombinedDataset(
        enhancers=enhancers,
        promoters=promoters,
        labels=labels,
        cache_dir=os.path.join(CACHE_DIR, f"{cell_name}_cache"),
        use_cache=True
    )

val_loaders = {}
for cell, (enhancers_val, promoters_val, labels_val) in val_data.items():
    val_dataset = MyDataset(enhancers_val, promoters_val, labels_val)
    enh_raw = [val_dataset[i][0] for i in range(len(val_dataset))]
    prom_raw = [val_dataset[i][1] for i in range(len(val_dataset))]
    labels_raw = [val_dataset[i][2] for i in range(len(val_dataset))]
    val_fold = create_optimized_validation_dataset(enh_raw, prom_raw, labels_raw, cell)
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





# 开始训练循环
for i in range(EPOCH):
    epimodel.train()
    train_epoch_loss = 0.0
    train_epoch_correct = 0
    train_epoch_preds = torch.tensor([]).to(device)
    train_epoch_target = torch.tensor([]).to(device)
    
    train_pbar = tqdm(train_loaders[TRAIN_CELL_LINE], desc=f"Epoch {i+1}/{EPOCH} [Training]", 
                      leave=True, dynamic_ncols=True)
    
    # 遍历训练数据
    for data in train_pbar:
        enhancer_ids, promoter_ids, enhancer_features, promoter_features, labels = data
        enhancer_ids = enhancer_ids.to(device, non_blocking=True)
        promoter_ids = promoter_ids.to(device, non_blocking=True)
        enhancer_features = enhancer_features.to(device, non_blocking=True)
        promoter_features = promoter_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs, emd, total_adaptive_loss = epimodel(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
        
        labels = labels.unsqueeze(1).float()
        
        train_epoch_preds = torch.cat((train_epoch_preds, outputs.view(-1)))
        train_epoch_target = torch.cat((train_epoch_target, labels.view(-1)))
        
        if labels.shape == torch.Size([1, 1]):
            labels = torch.reshape(labels, (1,))

        # 确保adaptive_loss是一个标量
        if isinstance(total_adaptive_loss, torch.Tensor) and total_adaptive_loss.numel() == 1:
            adaptive_loss_val = total_adaptive_loss.item()
        else:
            adaptive_loss_val = total_adaptive_loss

        # 使用新的损失计算方式
        loss, loss_details = epimodel.compute_loss(outputs, labels, adaptive_loss_val, return_details=True)
        
        train_epoch_loss += loss.item()
        train_epoch_correct += get_num_correct(outputs, labels)

        loss.backward()
        
        # 已移除PGD/FGM对抗训练逻辑

        epimodel.optimizer.step()
        epimodel.optimizer.zero_grad()
        
        # 更新进度条显示当前损失详情
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'immax': f'{loss_details["immax_loss"]:.4f}',
            'adaptive': f'{loss_details["adaptive_loss"]:.4f}',
            'alpha': f'{loss_details["alpha"]:.3f}'
        })
        
    scheduler.step()
    logger.info("learning_rate: {}".format(scheduler.get_last_lr()))
    
    # 记录当前epoch的损失详情和α值
    current_alpha = epimodel.get_loss_alpha()
    train_acc = train_epoch_correct / max(1, train_epoch_target.numel())
    logger.info("Epoch {}: 训练完成，当前α值为: {:.4f}, train_acc: {:.4f}".format(i + 1, current_alpha, train_acc))
    
    train_epoch_aupr = average_precision_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
    train_epoch_auc = roc_auc_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
    

    torch.save(epimodel.state_dict(), os.path.join(SAVE_MODEL_DIR, f"epimodel_{cell}_{i+1}.pth"))

    if i % VALIDATION_INTERVAL == 0 or i == EPOCH - 1:
        for cell, loader in val_loaders.items():
            val_loss, val_aupr, val_auc, val_acc = val_forwrd(epimodel, loader, cell)
            logger.info(
                "validate {} epoch: {}, val_loss：{}, val_aupr：{}, val_auc:{}, val_acc:{}".format(
                    cell, i + 1, val_loss, val_aupr, val_auc, val_acc
                )
            )
        # 保存模型
