#!/usr/bin/env python3
"""
读取所有cell类型的index.csv文件中的基因位置信息，生成BED格式文件
输出格式: chr4 100000 100001 (0-based BED标准格式)
"""

import os
import pandas as pd
from pathlib import Path

def extract_gene_positions(csv_file):
    """
    从index.csv文件中提取基因位置信息
    
    参数:
        csv_file: index.csv文件路径
        
    返回:
        set: 包含所有基因位置的集合
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 提取enhancer和promoter位置
    positions = set()
    
    # 从enhancer_name列提取位置
    for pos in df['enhancer_name']:
        positions.add(pos)
    
    # 从promoter_name列提取位置
    for pos in df['promoter_name']:
        positions.add(pos)
    
    return positions

def convert_to_bed_format(positions):
    """
    将位置信息转换为BED格式 (0-based)
    
    参数:
        positions: 包含位置信息的集合
        
    返回:
        list: BED格式的位置列表 (格式: chr4 100000 100001)
    """
    bed_positions = []
    
    for pos in positions:
        # 位置格式如 "chr1:9685722-9686400"
        # 转换为 "chr1 9685722 9686400" (0-based, BED标准格式)
        parts = pos.split(':')
        chr_name = parts[0]
        start_end = parts[1].split('-')
        
        # BED格式使用0-based起始坐标，1-based结束坐标
        start = int(start_end[0])  # 保持0-based
        end = int(start_end[1])    # 保持1-based结束
        
        # 组合成标准BED格式
        bed_pos = f"{chr_name} {start} {end}"
        bed_positions.append(bed_pos)
    
    # 排序
    bed_positions.sort()
    
    return bed_positions

def main():
    # 数据目录
    data_dir = Path("./data")
    
    # 输出文件
    output_file = "./data/all_genes.bed"
    
    # 收集所有位置
    all_positions = set()
    
    # 遍历所有cell类型的目录
    for cell_dir in data_dir.iterdir():
        if cell_dir.is_dir() and (cell_dir / "index.csv").exists():
            print(f"处理 {cell_dir.name} 的数据...")
            positions = extract_gene_positions(cell_dir / "index.csv")
            all_positions.update(positions)
            print(f"  找到 {len(positions)} 个位置")
    
    print(f"总共找到 {len(all_positions)} 个唯一位置")
    
    # 转换为BED格式
    bed_positions = convert_to_bed_format(all_positions)
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        for pos in bed_positions:
            f.write(pos + '\n')
    
    print(f"结果已保存到 {output_file}")
    print(f"共生成 {len(bed_positions)} 行BED格式数据 (0-based格式: chr4 100000 100001)")

if __name__ == "__main__":
    main()