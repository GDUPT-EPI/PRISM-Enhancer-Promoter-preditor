#!/usr/bin/env python3
"""
处理失败序列条目的脚本 - 从各个细胞类型目录中读取failed.txt文件，
更新map.csv，并输出到pairs_hg38.csv
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional

# 集中管理配置参数
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"

# 文件名配置
MAP_FILENAME = "map.csv"  # 映射文件名
E_SEQ_FAILED_FILENAME = "e_seq.failed.txt"  # enhancer序列失败文件名
P_SEQ_FAILED_FILENAME = "p_seq.failed.txt"  # promoter序列失败文件名
FILTERED_MAP_FILENAME = "pairs_hg38.csv"  # 过滤后的映射文件名

def find_cell_types() -> List[str]:
    """
    查找所有细胞类型目录
    
    Returns:
        细胞类型名称列表
    """
    if not DATA_DIR.exists():
        print(f"数据目录不存在: {DATA_DIR}")
        return []
    
    cell_types = []
    for item in DATA_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            cell_types.append(item.name)
    
    return sorted(cell_types)

def read_failed_regions(cell_type: str) -> Set[str]:
    """
    读取单个细胞类型的failed.txt文件，获取失败的区域
    
    Args:
        cell_type: 细胞类型名称
        
    Returns:
        失败区域集合
    """
    cell_type_dir = DATA_DIR / cell_type
    failed_regions = set()
    
    # 读取enhancer失败文件
    e_seq_failed_file = cell_type_dir / E_SEQ_FAILED_FILENAME
    if e_seq_failed_file.exists():
        with open(e_seq_failed_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    failed_regions.add(line)
    
    # 读取promoter失败文件
    p_seq_failed_file = cell_type_dir / P_SEQ_FAILED_FILENAME
    if p_seq_failed_file.exists():
        with open(p_seq_failed_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    failed_regions.add(line)
    
    return failed_regions

def read_map_file(cell_type: str) -> Optional[pd.DataFrame]:
    """
    读取单个细胞类型的map.csv文件
    
    Args:
        cell_type: 细胞类型名称
        
    Returns:
        map.csv的DataFrame，如果文件不存在则返回None
    """
    cell_type_dir = DATA_DIR / cell_type
    map_file = cell_type_dir / MAP_FILENAME
    
    if not map_file.exists():
        print(f"未找到 {cell_type} 的{MAP_FILENAME}文件")
        return None
    
    try:
        df = pd.read_csv(map_file)
        return df
    except Exception as e:
        print(f"读取 {cell_type} 的{MAP_FILENAME}文件时出错: {str(e)}")
        return None

def filter_map_data(df: pd.DataFrame, failed_regions: Set[str]) -> pd.DataFrame:
    """
    从map数据中移除失败的条目
    
    Args:
        df: 原始map数据的DataFrame
        failed_regions: 失败区域集合
        
    Returns:
        过滤后的DataFrame
    """
    # 检查必要的列是否存在
    if 'enhancer_name' not in df.columns or 'promoter_name' not in df.columns:
        print("map.csv文件缺少必要的列")
        return df
    
    # 过滤掉enhancer或promoter在失败列表中的条目
    filtered_df = df[
        ~df['enhancer_name'].isin(failed_regions) & 
        ~df['promoter_name'].isin(failed_regions)
    ].copy()
    
    return filtered_df

def process_cell_type(cell_type: str) -> bool:
    """
    处理单个细胞类型的数据，保存过滤后的文件到该细胞类型目录
    
    Args:
        cell_type: 细胞类型名称
        
    Returns:
        处理是否成功
    """
    print(f"处理细胞类型: {cell_type}")
    
    # 读取失败的区域
    failed_regions = read_failed_regions(cell_type)
    print(f"找到 {len(failed_regions)} 个失败区域")
    
    # 读取map文件
    map_df = read_map_file(cell_type)
    if map_df is None:
        return False
    
    print(f"原始map数据包含 {len(map_df)} 条记录")
    
    # 过滤数据
    filtered_df = filter_map_data(map_df, failed_regions)
    print(f"过滤后数据包含 {len(filtered_df)} 条记录")
    
    # 保存过滤后的文件到细胞类型目录
    cell_type_dir = DATA_DIR / cell_type
    output_file = cell_type_dir / FILTERED_MAP_FILENAME
    filtered_df.to_csv(output_file, index=False)
    print(f"过滤后的数据已保存到 {output_file}")
    
    return True

def main() -> None:
    """
    主函数：处理所有细胞类型的数据，为每个细胞类型生成过滤后的文件
    """
    print("开始处理失败序列条目...")
    
    # 查找所有细胞类型
    cell_types = find_cell_types()
    if not cell_types:
        print("未找到任何细胞类型目录")
        return
    
    print(f"找到 {len(cell_types)} 个细胞类型: {', '.join(cell_types)}")
    
    # 处理每个细胞类型
    success_count = 0
    for cell_type in cell_types:
        print(f"\n处理细胞类型: {cell_type}")
        if process_cell_type(cell_type):
            success_count += 1
    
    print(f"\n成功处理了 {success_count}/{len(cell_types)} 个细胞类型")
    print("处理完成!")

if __name__ == "__main__":
    main()