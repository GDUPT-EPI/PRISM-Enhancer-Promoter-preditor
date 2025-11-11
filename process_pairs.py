#!/usr/bin/env python3
"""
处理所有./data目录中的pairs.csv文件，提取特定列并处理数据
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# 配置常量
DATA_DIR = "./data"
OUTPUT_FILENAME = "index.csv"
TARGET_COLUMNS = ["enhancer_name", "promoter_name", "label"]


def find_pairs_csv_files(data_dir: str) -> List[str]:
    """
    查找所有data目录下的pairs.csv文件
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        pairs.csv文件路径列表
    """
    pairs_files = []
    data_path = Path(data_dir)
    
    # 遍历所有子目录，查找pairs.csv文件
    for cell_type_dir in data_path.iterdir():
        if cell_type_dir.is_dir():
            pairs_file = cell_type_dir / "pairs.csv"
            if pairs_file.exists():
                pairs_files.append(str(pairs_file))
    
    return pairs_files


def process_name(name: str) -> str:
    """
    处理名称，移除细胞系前缀，确保以chr开头
    
    Args:
        name: 原始名称，格式如"K562|chr1:6454864-6455189"
        
    Returns:
        处理后的名称，格式如"chr1:6454864-6455189"
    """
    # 如果包含|，则取|之后的部分
    if "|" in name:
        name = name.split("|", 1)[1]
    
    # 双重检查确保名称以chr开头
    if not name.startswith("chr"):
        raise ValueError(f"处理后的名称不以chr开头: {name}")
    
    return name


def process_pairs_file(file_path: str) -> pd.DataFrame:
    """
    处理单个pairs.csv文件，提取并处理指定列
    
    Args:
        file_path: pairs.csv文件路径
        
    Returns:
        处理后的DataFrame，包含enhancer_name, promoter_name, label列
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取目标列
    result_df = df[TARGET_COLUMNS].copy()
    
    # 处理enhancer_name和promoter_name列
    result_df["enhancer_name"] = result_df["enhancer_name"].apply(process_name)
    result_df["promoter_name"] = result_df["promoter_name"].apply(process_name)
    
    return result_df


def save_to_output_dir(df: pd.DataFrame, source_file_path: str) -> None:
    """
    将处理后的DataFrame保存到源文件同目录的index.csv
    
    Args:
        df: 处理后的DataFrame
        source_file_path: 源文件路径
    """
    # 获取源文件所在目录
    source_dir = os.path.dirname(source_file_path)
    output_path = os.path.join(source_dir, OUTPUT_FILENAME)
    
    # 保存到CSV
    df.to_csv(output_path, index=False)
    print(f"已保存处理后的文件到: {output_path}")


def main():
    """主函数，处理所有pairs.csv文件"""
    print("开始处理pairs.csv文件...")
    
    # 查找所有pairs.csv文件
    pairs_files = find_pairs_csv_files(DATA_DIR)
    print(f"找到{len(pairs_files)}个pairs.csv文件")
    
    # 处理每个文件
    for file_path in pairs_files:
        print(f"正在处理: {file_path}")
        try:
            # 处理文件
            processed_df = process_pairs_file(file_path)
            
            # 保存结果
            save_to_output_dir(processed_df, file_path)
            
            print(f"成功处理文件: {file_path}")
        except Exception as e:
            print(f"处理文件{file_path}时出错: {str(e)}")
    
    print("所有文件处理完成!")


if __name__ == "__main__":
    main()