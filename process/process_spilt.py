#!/usr/bin/env python3
"""
数据集切分脚本 - 处理pairs_hg38.csv及其关联的序列文件
对抗数据偏倚版本：实现1:20正负样本比例筛选和距离分布对齐

主要功能：
1. 保留所有正样本（label=1）
2. 基于距离分布对齐选择负样本，实现1:20正负样本比例
3. 确保负样本的启动子、增强子距离分布与正样本一致

使用方法:
    conda activate your-env  # 请根据实际情况激活环境
    python process_spilt copy.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import random
import re
from collections import defaultdict

# 集中配置参数 - 集中式路径配置模式
DATA_DIR = Path("data")  # 原始数据目录，使用相对路径
OUTPUT_DIR = Path("dataset")  # 输出目录，使用相对路径
TRAIN_RATIO = 0.75  # 训练集比例 (75%)
VAL_RATIO = 0.15  # 验证集比例 (15%)  
TEST_RATIO = 0.1  # 测试集比例 (10%)
RANDOM_SEED = 42  # 随机种子，确保结果可重复
KEEP_FILE = ["e_seq.csv", "p_seq.csv"]  # 需要保留的序列文件列表
PAIRS_NAME = "pairs_hg38.csv"  # pairs文件名称

# 1:20 正负样本比例配置
# 修复：每1个正样本对应20个负样本
NEG_PER_POS = 20  # 每个正样本对应的负样本数量，实现1:20正负比例

# 距离分箱数量配置 - 基于数据集规模的动态策略
# 推荐配置：小数据集(≤20k): 20-30, 中等数据集(20-40k): 50, 大数据集(≥40k): 100
# 当前数据集规模: GM12878(32k), HUVEC(23k), HeLa-S3(27k), IMR90(19k), K562(30k), NHEK(20k)
# 综合考虑各细胞系规模，统一使用50分箱是合理选择
DISTANCE_BINS = 50  # 距离分箱数量，用于距离分布对齐

# 细胞系列表
CELL_LINES = ["GM12878", "HUVEC", "HeLa-S3", "IMR90", "K562", "NHEK"]  # 支持的细胞系列表
IS_ALL_CELL_LINES = True  # 是否启用ALL细胞系模式：True=生成ALL目录(合并所有细胞系), False=不生成

def parse_genomic_coord(coord: str) -> Tuple[str, int, int]:
    """
    解析基因组坐标字符串
    
    Args:
        coord: 基因组坐标字符串，如 'chr1:9685722-9686400'
        
    Returns:
        (染色体, 起始位置, 结束位置) 的元组
    """
    match = re.match(r'^(chr[^:]+):(\d+)-(\d+)$', coord)
    if not match:
        raise ValueError(f"无效的基因组坐标格式: {coord}")
    
    chrom, start, end = match.groups()
    return chrom, int(start), int(end)


def calculate_distance(enhancer_coord: str, promoter_coord: str) -> int:
    """
    计算增强子和启动子之间的绝对距离
    
    Args:
        enhancer_coord: 增强子坐标字符串
        promoter_coord: 启动子坐标字符串
        
    Returns:
        增强子和启动子中心点之间的绝对距离
    """
    enh_chrom, enh_start, enh_end = parse_genomic_coord(enhancer_coord)
    prom_chrom, prom_start, prom_end = parse_genomic_coord(promoter_coord)
    
    # 确保在同一条染色体上
    if enh_chrom != prom_chrom:
        raise ValueError(f"增强子和启动子不在同一条染色体上: {enhancer_coord} vs {promoter_coord}")
    
    # 计算中心点
    enh_center = (enh_start + enh_end) / 2
    prom_center = (prom_start + prom_end) / 2
    
    return int(abs(enh_center - prom_center))


def create_distance_bins(distances: np.ndarray, num_bins: int = 50) -> np.ndarray:
    """
    创建距离分箱边界
    
    Args:
        distances: 距离数组
        num_bins: 分箱数量
        
    Returns:
        分箱边界数组
    """
    # 使用对数分箱来处理大范围的距离分布
    log_distances = np.log10(distances + 1)  # +1避免log(0)
    bins = np.linspace(log_distances.min(), log_distances.max(), num_bins + 1)
    return np.power(10, bins) - 1  # 转换回原始尺度


def get_sample_ratios_by_distance(df: pd.DataFrame, distance_bins: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    根据距离分布计算各分箱的样本比例
    
    Args:
        df: 包含距离信息的数据框
        distance_bins: 距离分箱边界
        
    Returns:
        分箱到样本比例的映射
    """
    # 为每个样本分配分箱
    bin_indices = np.digitize(df['distance'], distance_bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(distance_bins) - 2)  # 确保索引不越界
    
    # 计算每个分箱中正样本的总数
    positive_counts = np.bincount(bin_indices[df['label'] == 1], minlength=len(distance_bins) - 1)
    total_positive = positive_counts.sum()
    
    # 计算每个分箱的目标负样本数量（保持1:20比例）
    target_negative_ratios = {}
    for i, count in enumerate(positive_counts):
        if count > 0:
            # 每个正样本对应20个负样本
            target_negative_ratios[(i, i+1)] = min(count * 20, len(distance_bins) - 1 - i) / len(df)
        else:
            target_negative_ratios[(i, i+1)] = 0
    
    return target_negative_ratios


def select_balanced_negative_samples(df: pd.DataFrame, target_ratio: float = 1.0) -> pd.DataFrame:
    """
    基于距离分布对齐选择平衡的负样本
    
    Args:
        df: 原始数据框，包含所有样本
        target_ratio: 正负样本目标比例（正:负）
        
    Returns:
        平衡后的数据框
    """
    # 分离正负样本
    positive_samples = df[df['label'] == 1].copy()
    negative_samples = df[df['label'] == 0].copy()
    
    print(f"  原始数据: 正样本={len(positive_samples)}, 负样本={len(negative_samples)}")
    
    # 如果没有负样本，返回所有正样本
    if len(negative_samples) == 0:
        return positive_samples
    
    # 计算所有样本的距离
    all_distances = []
    for _, row in df.iterrows():
        try:
            distance = calculate_distance(row['enhancer_name'], row['promoter_name'])
            all_distances.append(distance)
        except ValueError as e:
            print(f"  警告: 跳过无效坐标配对 {row['enhancer_name']} - {row['promoter_name']}: {e}")
            continue
    
    if len(all_distances) == 0:
        print("  警告: 无法计算有效距离，返回所有正样本")
        return positive_samples
    
    # 创建距离分箱
    distance_bins = create_distance_bins(np.array(all_distances), DISTANCE_BINS)
    
    # 为正样本计算距离
    positive_distances = []
    for _, row in positive_samples.iterrows():
        distance = calculate_distance(row['enhancer_name'], row['promoter_name'])
        positive_distances.append(distance)
    
    # 为负样本计算距离
    negative_distances = []
    valid_negative_indices = []
    for idx, row in negative_samples.iterrows():
        try:
            distance = calculate_distance(row['enhancer_name'], row['promoter_name'])
            negative_distances.append(distance)
            valid_negative_indices.append(idx)
        except ValueError as e:
            print(f"  警告: 跳过无效负样本坐标 {row['enhancer_name']} - {row['promoter_name']}: {e}")
            continue
    
    if len(negative_distances) == 0:
        print("  警告: 没有有效的负样本，返回所有正样本")
        return positive_samples
    
    # 转换为numpy数组
    positive_distances = np.array(positive_distances)
    negative_distances = np.array(negative_distances)
    
    # 为正负样本分配分箱
    positive_bin_indices = np.digitize(positive_distances, distance_bins) - 1
    negative_bin_indices = np.digitize(negative_distances, distance_bins) - 1
    
    # 确保索引不越界
    positive_bin_indices = np.clip(positive_bin_indices, 0, len(distance_bins) - 2)
    negative_bin_indices = np.clip(negative_bin_indices, 0, len(distance_bins) - 2)
    
    # 计算每个分箱中正负样本的数量
    positive_bin_counts = np.bincount(positive_bin_indices, minlength=len(distance_bins) - 1)
    
    # 选择负样本：每个分箱选择指定倍数于该分箱正样本数量的负样本
    selected_negative_indices = []
    for bin_idx in range(len(positive_bin_counts)):
        pos_count = positive_bin_counts[bin_idx]
        if pos_count == 0:
            continue
        
        # 该分箱中负样本的候选索引
        bin_negative_indices = np.where(negative_bin_indices == bin_idx)[0]
        
        if len(bin_negative_indices) == 0:
            continue
        
        # 该分箱目标负样本数量
        target_negative_count = min(pos_count * target_ratio, len(bin_negative_indices))
        
        # 随机选择
        if target_negative_count <= len(bin_negative_indices):
            selected_indices = np.random.choice(bin_negative_indices, int(target_negative_count), replace=False)
        else:
            selected_indices = bin_negative_indices  # 如果候选数量不足，全部选择
        
        selected_negative_indices.extend([valid_negative_indices[i] for i in selected_indices])
    
    # 如果选择的负样本不足，使用全局随机选择补充
    if len(selected_negative_indices) < int(len(positive_samples) * target_ratio):
        remaining_negative = set(negative_samples.index) - set(selected_negative_indices)
        additional_count = int(len(positive_samples) * target_ratio) - len(selected_negative_indices)
        additional_count = min(additional_count, len(remaining_negative))
        
        if additional_count > 0:
            additional_indices = np.random.choice(list(remaining_negative), additional_count, replace=False)
            selected_negative_indices.extend(additional_indices)
    
    # 确保不超过原始负样本数量
    selected_negative_indices = selected_negative_indices[:len(negative_samples)]
    
    # 获取选中的负样本
    selected_negative_samples = negative_samples.loc[selected_negative_indices]
    
    print(f"  平衡后: 正样本={len(positive_samples)}, 负样本={len(selected_negative_samples)}")
    print(f"  实际正负样本比例: 1:{len(selected_negative_samples)/len(positive_samples):.1f} (目标比例: 1:{target_ratio})")
    
    # 合并正样本和选中的负样本
    balanced_df = pd.concat([positive_samples, selected_negative_samples], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    return balanced_df


def create_output_directories() -> None:
    """创建输出目录结构 - 按数据集类型组织"""
    dataset_splits = ['train', 'val', 'test']
    
    # 创建主要的输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 创建子目录结构
    for split in dataset_splits:
        for cell_line in CELL_LINES:
            split_dir = OUTPUT_DIR / split / cell_line
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果启用ALL细胞系，创建ALL目录
        if IS_ALL_CELL_LINES:
            all_dir = OUTPUT_DIR / split / "ALL"
            all_dir.mkdir(parents=True, exist_ok=True)


def split_data_for_cell_line(cell_line: str) -> None:
    """
    为单个细胞系处理数据切分
    
    Args:
        cell_line: 细胞系名称
    """
    print(f"\n处理细胞系: {cell_line}")
    
    # 构建文件路径
    pairs_file = DATA_DIR / cell_line / PAIRS_NAME
    
    if not pairs_file.exists():
        print(f"  警告: 文件 {pairs_file} 不存在，跳过该细胞系")
        return
    
    try:
        # 读取pairs数据
        df = pd.read_csv(pairs_file)
        print(f"  原始数据shape: {df.shape}")
        
        # 数据验证
        required_columns = ['enhancer_name', 'promoter_name', 'label']
        if not all(col in df.columns for col in required_columns):
            print(f"  错误: 文件缺少必要列 {required_columns}")
            return
        
        # 验证label列的值
        unique_labels = df['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            print(f"  警告: 发现无效的label值 {unique_labels}，应仅包含0和1")
        
        print(f"  原始数据统计: 正样本={(df['label']==1).sum()}, 负样本={(df['label']==0).sum()}")
        
        # 应用距离分布对齐的负样本选择
        balanced_df = select_balanced_negative_samples(df, target_ratio=NEG_PER_POS)
        
        print(f"  平衡后数据shape: {balanced_df.shape}")
        
        # 进行训练/验证/测试集切分
        train_data, val_data, test_data = split_data_into_sets(balanced_df)
        
        # 创建输出目录
        cell_output_dir = OUTPUT_DIR / "train" / cell_line
        cell_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存切分后的数据
        train_data.to_csv(cell_output_dir / PAIRS_NAME, index=False)
        (OUTPUT_DIR / "val" / cell_line).mkdir(parents=True, exist_ok=True)
        val_data.to_csv(OUTPUT_DIR / "val" / cell_line / PAIRS_NAME, index=False)
        (OUTPUT_DIR / "test" / cell_line).mkdir(parents=True, exist_ok=True)
        test_data.to_csv(OUTPUT_DIR / "test" / cell_line / PAIRS_NAME, index=False)
        
        print(f"  数据已保存到: {cell_output_dir}")
        
        # 复制序列文件
        copy_associated_files(cell_line)
        
    except Exception as e:
        print(f"  错误: 处理细胞系 {cell_line} 时发生异常: {e}")
        import traceback
        traceback.print_exc()


def split_data_into_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将数据切分为训练集、验证集和测试集
    
    Args:
        df: 输入数据框
        
    Returns:
        (训练集, 验证集, 测试集) 的元组
    """
    # 设置随机种子以确保可重复性
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # 计算每个集合的大小
    total_size = len(df)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size
    
    # 随机打乱数据
    shuffled_indices = np.random.permutation(total_size)
    
    # 切分数据
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    
    train_data = df.iloc[train_indices].copy()
    val_data = df.iloc[val_indices].copy()
    test_data = df.iloc[test_indices].copy()
    
    print(f"    训练集: {len(train_data)} 样本 (正:{(train_data['label']==1).sum()}, 负:{(train_data['label']==0).sum()})")
    print(f"    验证集: {len(val_data)} 样本 (正:{(val_data['label']==1).sum()}, 负:{(val_data['label']==0).sum()})")
    print(f"    测试集: {len(test_data)} 样本 (正:{(test_data['label']==1).sum()}, 负:{(test_data['label']==0).sum()})")
    
    return train_data, val_data, test_data


def copy_associated_files(cell_line: str) -> None:
    """
    复制与细胞系相关的序列文件，根据对应的配对数据筛选相应的序列
    
    Args:
        cell_line: 细胞系名称
    """
    source_dir = DATA_DIR / cell_line
    
    # 读取已保存的数据集
    train_data = pd.read_csv(OUTPUT_DIR / "train" / cell_line / PAIRS_NAME)
    val_data = pd.read_csv(OUTPUT_DIR / "val" / cell_line / PAIRS_NAME)
    test_data = pd.read_csv(OUTPUT_DIR / "test" / cell_line / PAIRS_NAME)
    
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for dataset_name, dataset_df in datasets.items():
        target_dir = OUTPUT_DIR / dataset_name / cell_line
        
        for file_name in KEEP_FILE:
            source_file = source_dir / file_name
            if not source_file.exists():
                print(f"    警告: 文件 {source_file} 不存在，跳过")
                continue
            
            # 读取原始序列文件
            try:
                seq_df = pd.read_csv(source_file)
            except Exception as e:
                print(f"    错误: 读取 {source_file} 失败: {e}")
                continue
            
            # 根据文件类型确定匹配逻辑
            if file_name == "e_seq.csv":
                # 增强子文件，根据enhancer_name匹配
                key_column = "enhancer_name"
                original_key_column = "region"
            elif file_name == "p_seq.csv":
                # 启动子文件，根据promoter_name匹配
                key_column = "promoter_name"
                original_key_column = "region"
            else:
                print(f"    警告: 未知文件类型 {file_name}，跳过...")
                continue
            
            # 提取对应的序列
            required_keys = dataset_df[key_column].unique()
            filtered_seq_df = seq_df[seq_df[original_key_column].isin(required_keys)]
            
            # 保存筛选后的序列
            target_file = target_dir / file_name
            filtered_seq_df.to_csv(target_file, index=False)
            print(f"    保存: {file_name} -> {target_dir} ({len(filtered_seq_df)} 条记录)")
            
            # 验证数据一致性
            if len(filtered_seq_df) != len(required_keys):
                print(f"    警告: {file_name} 中找到 {len(filtered_seq_df)} 条记录，但期望 {len(required_keys)} 条")
                missing_keys = set(required_keys) - set(filtered_seq_df[original_key_column])
                if missing_keys:
                    print(f"    缺失的键: {list(missing_keys)[:5]}{'...' if len(missing_keys) > 5 else ''}")


def process_all_cell_lines() -> None:
    """
    处理所有细胞系数据，合并为ALL细胞系
    
    合并所有细胞系的数据，进行距离分布对齐和负样本选择，
    然后进行训练/验证/测试切分
    """
    print(f"\n处理所有细胞系: ALL")
    
    # 收集所有细胞系的数据
    all_dfs = []
    processed_cell_lines = []
    
    for cell_line in CELL_LINES:
        pairs_file = DATA_DIR / cell_line / PAIRS_NAME
        if not pairs_file.exists():
            print(f"  警告: 跳过不存在的细胞系 {cell_line}")
            continue
        
        try:
            df = pd.read_csv(pairs_file)
            # 添加细胞系标识列
            df['cell_line'] = cell_line
            all_dfs.append(df)
            processed_cell_lines.append(cell_line)
            print(f"  加载 {cell_line}: {df.shape} 样本")
        except Exception as e:
            print(f"  错误: 加载细胞系 {cell_line} 失败: {e}")
            continue
    
    if not all_dfs:
        print("  错误: 没有可用的细胞系数据")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  合并后总数据: {combined_df.shape}")
    print(f"  细胞系分布: {combined_df['cell_line'].value_counts().to_dict()}")
    
    # 验证数据
    required_columns = ['enhancer_name', 'promoter_name', 'label', 'cell_line']
    if not all(col in combined_df.columns for col in required_columns):
        print(f"  错误: 合并数据缺少必要列 {required_columns}")
        return
    
    # 验证label列的值
    unique_labels = combined_df['label'].unique()
    if not all(label in [0, 1] for label in unique_labels):
        print(f"  警告: 发现无效的label值 {unique_labels}，应仅包含0和1")
    
    print(f"  原始数据统计: 正样本={(combined_df['label']==1).sum()}, 负样本={(combined_df['label']==0).sum()}")
    
    # 应用距离分布对齐的负样本选择
    balanced_df = select_balanced_negative_samples(combined_df, target_ratio=NEG_PER_POS)
    
    print(f"  平衡后数据shape: {balanced_df.shape}")
    
    # 进行训练/验证/测试集切分
    train_data, val_data, test_data = split_data_into_sets(balanced_df)
    
    # 保存切分后的数据到ALL目录
    all_output_dir = OUTPUT_DIR / "train" / "ALL"
    all_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存切分后的数据
    train_data.to_csv(all_output_dir / PAIRS_NAME, index=False)
    (OUTPUT_DIR / "val" / "ALL").mkdir(parents=True, exist_ok=True)
    val_data.to_csv(OUTPUT_DIR / "val" / "ALL" / PAIRS_NAME, index=False)
    (OUTPUT_DIR / "test" / "ALL").mkdir(parents=True, exist_ok=True)
    test_data.to_csv(OUTPUT_DIR / "test" / "ALL" / PAIRS_NAME, index=False)
    
    print(f"  数据已保存到: {all_output_dir}")
    
    # 复制ALL细胞系的序列文件
    copy_associated_files_for_all(train_data, val_data, test_data)


def copy_associated_files_for_all(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    为ALL细胞系复制和筛选序列文件
    
    Args:
        train_data: 训练集数据
        val_data: 验证集数据
        test_data: 测试集数据
    """
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for dataset_name, dataset_df in datasets.items():
        target_dir = OUTPUT_DIR / dataset_name / "ALL"
        
        for file_name in KEEP_FILE:
            # 收集所有细胞系的对应文件
            all_seq_dfs = []
            
            for cell_line in CELL_LINES:
                source_file = DATA_DIR / cell_line / file_name
                if not source_file.exists():
                    continue
                
                try:
                    seq_df = pd.read_csv(source_file)
                    all_seq_dfs.append(seq_df)
                except Exception as e:
                    print(f"    警告: 读取 {cell_line}/{file_name} 失败: {e}")
                    continue
            
            if not all_seq_dfs:
                print(f"    警告: 没有找到 {file_name} 文件")
                continue
            
            # 合并所有细胞系的序列数据
            combined_seq_df = pd.concat(all_seq_dfs, ignore_index=True)
            
            # 根据文件类型确定匹配逻辑
            if file_name == "e_seq.csv":
                # 增强子文件，根据enhancer_name匹配
                key_column = "enhancer_name"
                original_key_column = "region"
            elif file_name == "p_seq.csv":
                # 启动子文件，根据promoter_name匹配
                key_column = "promoter_name"
                original_key_column = "region"
            else:
                print(f"    警告: 未知文件类型 {file_name}，跳过...")
                continue
            
            # 提取对应的序列
            required_keys = dataset_df[key_column].unique()
            filtered_seq_df = combined_seq_df[combined_seq_df[original_key_column].isin(required_keys)]
            
            # 去重（防止不同细胞系有相同的序列）
            if len(filtered_seq_df) > len(required_keys):
                filtered_seq_df = filtered_seq_df.drop_duplicates(subset=[original_key_column])
            
            # 保存筛选后的序列
            target_file = target_dir / file_name
            filtered_seq_df.to_csv(target_file, index=False)
            print(f"    保存: {file_name} -> {target_dir} ({len(filtered_seq_df)} 条记录)")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("数据集切分脚本 - 对抗数据偏倚版本")
    print("=" * 60)
    print(f"配置参数:")
    print(f"  数据源目录: {DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  切分比例: 训练集 {TRAIN_RATIO*100:.0f}%, 验证集 {VAL_RATIO*100:.0f}%, 测试集 {TEST_RATIO*100:.0f}%")
    print(f"  正负样本比例: 1:{NEG_PER_POS}")
    print(f"  距离分箱数: {DISTANCE_BINS}")
    print(f"  随机种子: {RANDOM_SEED}")
    print(f"  启用ALL细胞系: {IS_ALL_CELL_LINES}")
    print()
    
    # 创建输出目录
    create_output_directories()
    print(f"输出目录创建完成: {OUTPUT_DIR}")
    
    # 处理每个细胞系
    processed_count = 0
    for cell_line in CELL_LINES:
        cell_dir = DATA_DIR / cell_line
        if cell_dir.exists():
            split_data_for_cell_line(cell_line)
            processed_count += 1
        else:
            print(f"警告: 细胞系目录 {cell_dir} 不存在，跳过")
    
    # 如果启用ALL细胞系，合并处理所有细胞系数据
    if IS_ALL_CELL_LINES and processed_count > 0:
        process_all_cell_lines()
        print(f"ALL细胞系处理完成")
    
    print(f"\n处理完成! 共处理 {processed_count} 个细胞系")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if IS_ALL_CELL_LINES:
        print("  同时生成了合并的ALL细胞系数据")


if __name__ == "__main__":
    main()