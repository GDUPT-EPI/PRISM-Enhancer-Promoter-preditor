#!/usr/bin/env python3
"""
本地序列处理工具 - 使用samtools从本地FASTA文件获取DNA序列
处理所有细胞类型的map.csv文件，提取enhancer和promoter序列
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures
import time
from typing import Dict, List, Tuple, Optional
import re

# 集中管理配置参数
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR
FASTA_FILE = DATA_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
CHROMOSOME_LENGTH_FILE = DATA_DIR / "chromosome_lengths.txt"

# 文件名配置
INDEX_FILENAME = "map.csv"  # 细胞类型索引文件名
ENHANCER_SEQ_FILENAME = "e_seq.csv"  # enhancer序列输出文件名
PROMOTER_SEQ_FILENAME = "p_seq.csv"  # promoter序列输出文件名

# 多线程配置
MAX_WORKERS = 10  # 最大并发线程数

# DNA序列验证正则表达式
DNA_PATTERN = re.compile(r'^[ATCGN]*$', re.IGNORECASE)

def validate_dna_sequence(sequence: str) -> str:
    """
    验证DNA序列，确保只包含ATCGN字符
    
    Args:
        sequence: 待验证的DNA序列
        
    Returns:
        验证后的DNA序列（大写）
        
    Raises:
        ValueError: 当序列包含非ATCGN字符时
    """
    if not sequence:
        return ""
    
    # 转换为大写
    sequence = sequence.upper()
    
    # 验证序列是否只包含ATCGN
    if not DNA_PATTERN.match(sequence):
        # 找出所有非ATCGN字符
        invalid_chars = set(re.sub(r'[ATCGN]', '', sequence))
        raise ValueError(f"DNA序列包含无效字符: {invalid_chars}")
    
    return sequence

def fetch_sequence(chromosome: str, start: int, end: int) -> str:
    """
    使用samtools从本地FASTA文件获取指定区域的DNA序列
    
    Args:
        chromosome: 染色体名称 (如 "1", "X", "Y")
        start: 起始位置 (1-based)
        end: 结束位置 (1-based)
        
    Returns:
        DNA序列字符串
        
    Raises:
        RuntimeError: 当samtools命令执行失败时
    """
    # 检查FASTA文件是否存在
    if not FASTA_FILE.exists():
        raise FileNotFoundError(f"FASTA文件不存在: {FASTA_FILE}")
    
    # 构建samtools命令
    # 注意: samtools使用1-based坐标系统
    cmd = [
        "samtools", "faidx", 
        str(FASTA_FILE),
        f"{chromosome}:{start}-{end}"
    ]
    
    try:
        # 执行samtools命令
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # 解析输出，移除标题行
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return ""
            
        # 合并所有序列行
        sequence = ''.join(lines[1:])
        # 验证DNA序列
        return validate_dna_sequence(sequence)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"samtools命令执行失败: {e.stderr}") from e
    except Exception as e:
        raise RuntimeError(f"获取序列时发生错误: {str(e)}") from e

def process_region(region: str) -> Tuple[str, str]:
    """
    处理单个区域，获取其DNA序列
    
    Args:
        region: 区域字符串，格式为 "chr:start-end"
        
    Returns:
        元组 (region, sequence)
    """
    try:
        # 解析区域字符串
        parts = region.split(':')
        if len(parts) != 2:
            return region, ""
            
        chromosome = parts[0]
        if chromosome.startswith('chr'):
            chromosome = chromosome[3:]  # 移除'chr'前缀
            
        positions = parts[1].split('-')
        if len(positions) != 2:
            return region, ""
            
        start = int(positions[0])
        end = int(positions[1])
        
        # 获取序列
        sequence = fetch_sequence(chromosome, start, end)
        return region, sequence
        
    except Exception as e:
        print(f"处理区域 {region} 时出错: {str(e)}")
        return region, ""

def process_regions_batch(regions: List[str], output_file: Path) -> None:
    """
    批量处理区域并保存结果
    
    Args:
        regions: 区域列表
        output_file: 输出文件路径
    """
    print(f"开始处理 {len(regions)} 个区域...")
    
    results = {}
    failed_regions = []
    
    # 使用多线程处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_region = {
            executor.submit(process_region, region): region 
            for region in regions
        }
        
        # 收集结果
        completed = 0
        for future in concurrent.futures.as_completed(future_to_region):
            region, sequence = future.result()
            if sequence:
                results[region] = sequence
            else:
                failed_regions.append(region)
                
            completed += 1
            if completed % 100 == 0:
                print(f"已完成 {completed}/{len(regions)} 个区域")
    
    # 保存结果
    save_sequences(results, output_file)
    
    # 保存失败区域
    if failed_regions:
        failed_file = output_file.with_suffix('.failed.txt')
        with open(failed_file, 'w') as f:
            for region in failed_regions:
                f.write(f"{region}\n")
        print(f"有 {len(failed_regions)} 个区域处理失败，已保存到 {failed_file}")
    
    print(f"处理完成，成功获取 {len(results)} 个序列")

def save_sequences(sequences: Dict[str, str], output_file: Path) -> None:
    """
    保存序列到文件
    
    Args:
        sequences: 序列字典 {region: sequence}
        output_file: 输出文件路径
    """
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为CSV格式
    df = pd.DataFrame([
        {"region": region, "sequence": sequence}
        for region, sequence in sequences.items()
    ])
    df.to_csv(output_file, index=False)
    
    # 同时保存为FASTA格式
    fasta_file = output_file.with_suffix('.fasta')
    with open(fasta_file, 'w') as f:
        for region, sequence in sequences.items():
            f.write(f">{region}\n")
            # 每行写80个字符
            for i in range(0, len(sequence), 80):
                f.write(f"{sequence[i:i+80]}\n")
    
    print(f"序列已保存到 {output_file} 和 {fasta_file}")

def find_cell_types() -> List[str]:
    """
    查找所有细胞类型目录
    
    Returns:
        细胞类型名称列表
    """
    if not RESULTS_DIR.exists():
        print(f"结果目录不存在: {RESULTS_DIR}")
        return []
    
    cell_types = []
    for item in RESULTS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            cell_types.append(item.name)
    
    return sorted(cell_types)

def process_cell_type(cell_type: str) -> None:
    """
    处理单个细胞类型的{INDEX_FILENAME}文件，提取enhancer和promoter序列
    
    Args:
        cell_type: 细胞类型名称
    """
    cell_type_dir = RESULTS_DIR / cell_type
    if not cell_type_dir.exists():
        print(f"细胞类型目录不存在: {cell_type_dir}")
        return
    
    # 查找map.csv文件
    index_file = cell_type_dir / INDEX_FILENAME
    if not index_file.exists():
        print(f"未找到 {cell_type} 的{INDEX_FILENAME}文件")
        return
    
    print(f"处理 {cell_type} 的{INDEX_FILENAME}文件: {index_file}")
    
    try:
        # 读取map.csv文件
        df = pd.read_csv(index_file)
        
        # 检查必要的列是否存在
        if 'enhancer_name' not in df.columns or 'promoter_name' not in df.columns:
            print(f"{cell_type} 的{INDEX_FILENAME}文件缺少必要的列")
            return
        
        # 提取enhancer和promoter区域
        enhancer_regions = df['enhancer_name'].dropna().unique().tolist()
        promoter_regions = df['promoter_name'].dropna().unique().tolist()
        
        print(f"找到 {len(enhancer_regions)} 个enhancer区域和 {len(promoter_regions)} 个promoter区域")
        
        # 处理enhancer序列
        if enhancer_regions:
            enhancer_output = cell_type_dir / ENHANCER_SEQ_FILENAME
            process_regions_batch(enhancer_regions, enhancer_output)
        
        # 处理promoter序列
        if promoter_regions:
            promoter_output = cell_type_dir / PROMOTER_SEQ_FILENAME
            process_regions_batch(promoter_regions, promoter_output)
            
    except Exception as e:
        print(f"处理 {cell_type} 的{INDEX_FILENAME}文件时出错: {str(e)}")

def main() -> None:
    """
    主函数：处理所有细胞类型的{INDEX_FILENAME}文件
    """
    print("开始本地序列处理...")
    
    # 检查FASTA文件
    if not FASTA_FILE.exists():
        print(f"错误: FASTA文件不存在: {FASTA_FILE}")
        return
    
    # 查找所有细胞类型
    cell_types = find_cell_types()
    if not cell_types:
        print("未找到任何细胞类型目录")
        return
    
    print(f"找到 {len(cell_types)} 个细胞类型: {', '.join(cell_types)}")
    
    # 处理每个细胞类型
    for cell_type in cell_types:
        print(f"\n处理细胞类型: {cell_type}")
        process_cell_type(cell_type)
    
    print("\n所有处理完成!")

if __name__ == "__main__":
    main()