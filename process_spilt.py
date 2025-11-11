#!/usr/bin/env python3
"""
数据集切分脚本 - 处理pairs_hg38.csv及其关联的序列文件

按照9:1的比例将data目录中的pairs_hg38.csv文件切分为训练集和验证集，
并保存到dataset目录中，按照dataset/train/细胞系/的结构组织。
同时，根据切分结果提取对应的增强子(e_seq.csv)和启动子(p_seq.csv)序列文件。

使用方法:
    conda activate your-env  # 请根据实际情况激活环境
    python split.py
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import random

# 集中配置参数
DATA_DIR = Path("data")  # 原始数据目录
OUTPUT_DIR = Path("dataset")  # 输出目录
TRAIN_RATIO = 0.75  # 训练集比例
VAL_RATIO = 0.15  # 验证集比例
TEST_RATIO = 0.1  # 测试集比例
RANDOM_SEED = 42  # 随机种子，确保可重复性
KEEP_FILE = ["e_seq.csv", "p_seq.csv"]  # 需要保留的文件列表，在随机切分后读取对应的增强子/启动子序列

# 细胞系列表
CELL_LINES = ["GM12878", "HUVEC", "HeLa-S3", "IMR90", "K562", "NHEK"]


def create_output_directories() -> None:
    """创建输出目录结构 - 按数据集类型组织"""
    for split in ["train", "val", "test"]:
        for cell_line in CELL_LINES:
            (OUTPUT_DIR / split / cell_line).mkdir(parents=True, exist_ok=True)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按照指定比例切分数据集
    
    Args:
        df: 原始数据框，包含enhancer_name, promoter_name和label列
        
    Returns:
        训练集、验证集和测试集的数据框元组
    """
    # 打乱数据
    df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # 计算切分点
    n = len(df_shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    # 切分数据
    train_df = df_shuffled.iloc[:train_end]
    val_df = df_shuffled.iloc[train_end:val_end]
    test_df = df_shuffled.iloc[val_end:]
    
    return train_df, val_df, test_df


def process_cell_line(cell_line: str) -> None:
    """
    处理单个细胞系的数据
    
    Args:
        cell_line: 细胞系名称
    """
    # 输入文件路径
    input_file = DATA_DIR / cell_line / "pairs_hg38.csv"
    
    # 检查文件是否存在
    if not input_file.exists():
        print(f"警告: 文件 {input_file} 不存在，跳过")
        return
    
    # 读取数据
    print(f"处理 {cell_line} 的数据...")
    df = pd.read_csv(input_file)
    print(f"  原始数据大小: {len(df)}")
    
    # 切分数据
    train_df, val_df, test_df = split_data(df)
    print(f"  训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
    
    # 保存切分后的数据 - 按数据集类型组织
    train_df.to_csv(OUTPUT_DIR / "train" / cell_line / "pairs_hg38.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val" / cell_line / "pairs_hg38.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test" / cell_line / "pairs_hg38.csv", index=False)
    
    # 处理需要保留的文件
    for file_name in KEEP_FILE:
        # 读取原始文件
        original_file = DATA_DIR / cell_line / file_name
        if not original_file.exists():
            print(f"警告: {original_file} 不存在，跳过...")
            continue
            
        original_df = pd.read_csv(original_file)
        
        # 根据文件类型确定列名
        if file_name == "e_seq.csv":
            # 增强子文件，根据enhancer_name匹配
            key_column = "enhancer_name"
            original_key_column = "region"
        elif file_name == "p_seq.csv":
            # 启动子文件，根据promoter_name匹配
            key_column = "promoter_name"
            original_key_column = "region"
        else:
            print(f"警告: 未知文件类型 {file_name}，跳过...")
            continue
        
        # 提取训练集中需要的增强子/启动子
        train_keys = train_df[key_column].unique()
        train_seq_df = original_df[original_df[original_key_column].isin(train_keys)]
        train_seq_df.to_csv(OUTPUT_DIR / "train" / cell_line / file_name, index=False)
        
        # 提取验证集中需要的增强子/启动子
        val_keys = val_df[key_column].unique()
        val_seq_df = original_df[original_df[original_key_column].isin(val_keys)]
        val_seq_df.to_csv(OUTPUT_DIR / "val" / cell_line / file_name, index=False)
        
        # 提取测试集中需要的增强子/启动子
        test_keys = test_df[key_column].unique()
        test_seq_df = original_df[original_df[original_key_column].isin(test_keys)]
        test_seq_df.to_csv(OUTPUT_DIR / "test" / cell_line / file_name, index=False)


def main() -> None:
    """主函数"""
    print("开始切分数据集...")
    
    # 创建输出目录
    create_output_directories()
    
    # 处理每个细胞系
    for cell_line in CELL_LINES:
        process_cell_line(cell_line)
    
    print("数据集切分完成!")


if __name__ == "__main__":
    main()