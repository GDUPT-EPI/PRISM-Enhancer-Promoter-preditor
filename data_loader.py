#!/usr/bin/env python3
"""
数据加载模块 - 用于从新的数据结构中加载增强子-启动子配对数据
符合项目规范：集中配置管理、相对路径、简洁高效
"""
from config import DOMAIN_KL_DIR
from torch.utils.data import Dataset, Sampler
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
from config import DATA_DIR as CONFIG_DATA_DIR, TEST_CELL_LINES

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(CONFIG_DATA_DIR)

# 细胞系列表来源于配置
CELL_LINES = TEST_CELL_LINES

# 数据集类型
DATA_TYPES = ["train", "val", "test"]


def load_sequence_data(seq_file: Path) -> Dict[str, str]:
    """
    从序列文件中加载序列数据
    
    Args:
        seq_file: 序列文件路径(e_seq.csv或p_seq.csv)
        
    Returns:
        区域名称到序列的映射字典
    """
    df = pd.read_csv(seq_file)
    return dict(zip(df['region'], df['sequence']))


def load_pairs_data(pairs_file: Path) -> pd.DataFrame:
    """
    从配对文件中加载增强子-启动子配对数据
    
    Args:
        pairs_file: 配对文件路径(pairs_hg38.csv)
        
    Returns:
        包含enhancer_name, promoter_name和label的数据框
    """
    return pd.read_csv(pairs_file)


def prepare_cell_data(cell_line: str, data_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    准备指定细胞系和数据类型的数据
    
    Args:
        cell_line: 细胞系名称(如GM12878)
        data_type: 数据类型(train/val/test)
        
    Returns:
        增强子序列数组、启动子序列数组、标签数组的元组
    """
    # 构建文件路径
    cell_dir = DATA_DIR / data_type / cell_line
    pairs_file = cell_dir / "pairs_hg38.csv"
    e_seq_file = cell_dir / "e_seq.csv"
    p_seq_file = cell_dir / "p_seq.csv"
    
    # 加载数据
    pairs_df = load_pairs_data(pairs_file)
    e_sequences = load_sequence_data(e_seq_file)
    p_sequences = load_sequence_data(p_seq_file)
    
    # 提取序列和标签
    enhancers = [e_sequences[enhancer] for enhancer in pairs_df['enhancer_name']]
    promoters = [p_sequences[promoter] for promoter in pairs_df['promoter_name']]
    labels = pairs_df['label'].values
    
    return np.array(enhancers), np.array(promoters), labels


def create_datasets_for_cell(cell_line: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    为指定细胞系创建所有数据集(train/val/test)
    
    Args:
        cell_line: 细胞系名称(如GM12878)
        
    Returns:
        包含train/val/test数据集的字典，每个值为(enhancers, promoters, labels)元组
    """
    datasets = {}
    for data_type in DATA_TYPES:
        datasets[data_type] = prepare_cell_data(cell_line, data_type)
    
    return datasets


def get_available_cells() -> List[str]:
    """
    获取数据目录中可用的细胞系列表
    
    Returns:
        可用细胞系名称列表
    """
    available_cells = []
    test_dir = DATA_DIR / "test"
    
    if test_dir.exists():
        for item in test_dir.iterdir():
            if item.is_dir() and (item / "pairs_hg38.csv").exists():
                available_cells.append(item.name)
    
    return sorted(available_cells)


def load_all_test_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    加载所有细胞系的测试数据
    
    Returns:
        细胞系名称到测试数据(enhancers, promoters, labels)的映射字典
    """
    test_data = {}
    for cell_line in CELL_LINES:
        if (DATA_DIR / "test" / cell_line).exists():
            test_data[cell_line] = prepare_cell_data(cell_line, "test")
    return test_data


def load_all_val_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    加载所有细胞系的验证数据
    
    Returns:
        细胞系名称到验证数据(enhancers, promoters, labels)的映射字典
    """
    val_data = {}
    for cell_line in CELL_LINES:
        if (DATA_DIR / "val" / cell_line).exists():
            val_data[cell_line] = prepare_cell_data(cell_line, "val")
    return val_data

def load_all_train_data() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_data = {}
    for cell_line in CELL_LINES:
        if (DATA_DIR / "train" / cell_line).exists():
            train_data[cell_line] = prepare_cell_data(cell_line, "train")
    return train_data


def load_train_data(cell_line: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载指定细胞系的训练数据
    
    Args:
        cell_line: 细胞系名称，默认为HUVEC
        
    Returns:
        增强子序列数组、启动子序列数组、标签数组的元组
    """
    return prepare_cell_data(cell_line, "train")

# 从embedding.py中复制的MyDataset类
class MyDataset():
    
    def __init__(self, enhancers, promoters, labels):
        self.enhancers = enhancers
        self.promoters = promoters
        self.labels = labels
        
        self.enhancer_count = len(enhancers)
        self.promoter_count = len(promoters)
        self.label_count = len(labels)
        
    def __getitem__(self, idx):
        enhancer_sequence = str(self.enhancers[idx])
        promoter_sequence = str(self.promoters[idx])
        
        label = int(self.labels[idx])
        return enhancer_sequence, promoter_sequence, label
    
    def __len__(self):
        return self.label_count
    
    def count_lines_in_txt(self, file_path):
        with open(file_path, "r") as file:
            line_count = len(file.readlines())
        
        return line_count


# ####################### PRISM.py 特供数据导入 ########################

def load_prism_data(data_type: str = "train") -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    加载PRISM特供数据 (domain-kl目录)
    
    Args:
        data_type: 数据类型 (train/val)
        
    Returns:
        pairs_df: 包含enhancer_name, promoter_name, label, cell_line的数据框
        e_sequences: enhancer序列字典
        p_sequences: promoter序列字典
    """
    data_dir = Path(DOMAIN_KL_DIR) / data_type / "ALL"
    
    pairs_file = data_dir / "pairs_hg19.csv"
    e_seq_file = data_dir / "e_seq.csv"
    p_seq_file = data_dir / "p_seq.csv"
    
    # 加载数据
    pairs_df = pd.read_csv(pairs_file)
    e_sequences = load_sequence_data(e_seq_file)
    p_sequences = load_sequence_data(p_seq_file)
    
    return pairs_df, e_sequences, p_sequences


class PRISMDataset(Dataset):
    """
    PRISM特供数据集 - 支持按细胞系分组
    """
    def __init__(self, pairs_df: pd.DataFrame, e_sequences: Dict[str, str], p_sequences: Dict[str, str]):
        self.pairs_df = pairs_df
        self.e_sequences = e_sequences
        self.p_sequences = p_sequences
        
        # 按细胞系分组索引
        self.cell_line_groups = {}
        for idx, row in pairs_df.iterrows():
            cell_line = row['cell_line']
            if cell_line not in self.cell_line_groups:
                self.cell_line_groups[cell_line] = []
            self.cell_line_groups[cell_line].append(idx)
        
        self.cell_lines = list(self.cell_line_groups.keys())
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        enhancer_seq = self.e_sequences[row['enhancer_name']]
        promoter_seq = self.p_sequences[row['promoter_name']]
        cell_line = row['cell_line']
        label = int(row['label'])
        
        return enhancer_seq, promoter_seq, cell_line, label


class PRISMContrastiveSampler(Sampler):
    """
    PRISM对比采样器
    每个batch包含:
    - batch_size//2 个来自同一细胞系的样本
    - batch_size//2 个来自其他细胞系的样本 (均匀采样，不放回)
    """
    def __init__(self, dataset: PRISMDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 确保batch_size是偶数
        if batch_size % 2 != 0:
            raise ValueError(f"PRISM batch_size必须是偶数，当前为{batch_size}")
        
        self.half_batch = batch_size // 2
        self.cell_lines = dataset.cell_lines
        self.cell_line_groups = dataset.cell_line_groups
        
        # 计算总batch数
        total_samples = len(dataset)
        self.num_batches = total_samples // batch_size
    
    def __iter__(self):
        # 为每个细胞系创建打乱的索引列表
        cell_line_indices = {}
        for cell_line in self.cell_lines:
            indices = self.cell_line_groups[cell_line].copy()
            if self.shuffle:
                random.shuffle(indices)
            cell_line_indices[cell_line] = indices
        
        # 生成batch
        for _ in range(self.num_batches):
            # 随机选择一个目标细胞系
            target_cell = random.choice(self.cell_lines)
            
            # 从目标细胞系采样 half_batch 个样本
            same_cell_indices = []
            target_indices = cell_line_indices[target_cell]
            
            for _ in range(self.half_batch):
                if len(target_indices) == 0:
                    # 如果用完了，重新打乱
                    target_indices = self.cell_line_groups[target_cell].copy()
                    if self.shuffle:
                        random.shuffle(target_indices)
                    cell_line_indices[target_cell] = target_indices
                same_cell_indices.append(target_indices.pop())
            
            # 从其他细胞系均匀采样 half_batch 个样本
            other_cells = [c for c in self.cell_lines if c != target_cell]
            diff_cell_indices = []
            
            if len(other_cells) > 0:
                # 计算每个其他细胞系应该采样多少个
                samples_per_cell = self.half_batch // len(other_cells)
                remaining = self.half_batch % len(other_cells)
                
                for i, other_cell in enumerate(other_cells):
                    num_samples = samples_per_cell + (1 if i < remaining else 0)
                    other_indices = cell_line_indices[other_cell]
                    
                    for _ in range(num_samples):
                        if len(other_indices) == 0:
                            # 如果用完了，重新打乱
                            other_indices = self.cell_line_groups[other_cell].copy()
                            if self.shuffle:
                                random.shuffle(other_indices)
                            cell_line_indices[other_cell] = other_indices
                        diff_cell_indices.append(other_indices.pop())
            else:
                # 如果只有一个细胞系，从同一细胞系采样
                for _ in range(self.half_batch):
                    if len(target_indices) == 0:
                        target_indices = self.cell_line_groups[target_cell].copy()
                        if self.shuffle:
                            random.shuffle(target_indices)
                        cell_line_indices[target_cell] = target_indices
                    diff_cell_indices.append(target_indices.pop())
            
            # 合并并打乱batch内的顺序
            batch_indices = same_cell_indices + diff_cell_indices
            if self.shuffle:
                random.shuffle(batch_indices)
            
            # 返回整个batch作为一个列表
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


class CellBatchSampler(Sampler):
    """
    细胞系纯批次采样器
    每个batch仅包含同一细胞系的样本，用于批次级细胞系分类
    """
    def __init__(self, dataset: PRISMDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cell_lines = dataset.cell_lines
        self.cell_line_groups = {c: g.copy() for c, g in dataset.cell_line_groups.items()}
        self.total_samples = len(dataset)
        self.num_batches = max(1, self.total_samples // max(1, batch_size))

    def __iter__(self):
        cell_line_indices = {}
        for cell_line, indices in self.cell_line_groups.items():
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)
            cell_line_indices[cell_line] = indices

        # 轮询细胞系，生成纯细胞批次
        cell_idx = 0
        for _ in range(self.num_batches):
            cell = self.cell_lines[cell_idx % len(self.cell_lines)]
            pool = cell_line_indices[cell]
            batch = []
            while len(batch) < self.batch_size:
                if len(pool) == 0:
                    pool = self.cell_line_groups[cell].copy()
                    if self.shuffle:
                        random.shuffle(pool)
                    cell_line_indices[cell] = pool
                batch.append(pool.pop())
            cell_idx += 1
            yield batch

    def __len__(self):
        return self.num_batches
