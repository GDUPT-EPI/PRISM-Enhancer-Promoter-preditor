#!/usr/bin/env python3
"""
数据加载模块 - 用于从新的数据结构中加载增强子-启动子配对数据
符合项目规范：集中配置管理、相对路径、简洁高效
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os

# 集中配置路径管理
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"

# 细胞系列表
CELL_LINES = ["GM12878", "HUVEC", "HeLa-S3", "IMR90", "K562", "NHEK"]

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


def load_train_data(cell_line: str = "HUVEC") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载指定细胞系的训练数据
    
    Args:
        cell_line: 细胞系名称，默认为HUVEC
        
    Returns:
        增强子序列数组、启动子序列数组、标签数组的元组
    """
    return prepare_cell_data(cell_line, "train")