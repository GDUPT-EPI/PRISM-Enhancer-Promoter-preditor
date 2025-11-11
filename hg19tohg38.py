#!/usr/bin/env python3
"""
基因组坐标映射工具 - 将hg19坐标映射到hg38坐标
读取各细胞类型的index.csv文件，使用BED映射文件进行坐标转换，生成map.csv文件
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# 集中管理配置参数
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
BED_MAPPING_FILE = DATA_DIR / "hglft_genome_df908_dc6340.bed"

# 坐标解析正则表达式
COORD_PATTERN = re.compile(r'^chr(\w+):(\d+)-(\d+)$', re.IGNORECASE)

def parse_coordinate(coord_str: str) -> Tuple[str, int, int]:
    """
    解析坐标字符串，提取染色体、起始和结束位置
    
    Args:
        coord_str: 坐标字符串，格式为 "chr1:1000-2000"
        
    Returns:
        元组 (chromosome, start, end)
        
    Raises:
        ValueError: 当坐标格式不正确时
    """
    match = COORD_PATTERN.match(coord_str.strip())
    if not match:
        raise ValueError(f"无效的坐标格式: {coord_str}")
    
    chromosome = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    
    return chromosome, start, end

def format_coordinate(chromosome: str, start: int, end: int) -> str:
    """
    格式化坐标为标准字符串
    
    Args:
        chromosome: 染色体名称
        start: 起始位置
        end: 结束位置
        
    Returns:
        格式化的坐标字符串，格式为 "chr1:1000-2000"
    """
    return f"chr{chromosome}:{start}-{end}"

def load_bed_mapping(bed_file: Path) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    加载BED映射文件，构建hg19到hg38的坐标映射字典
    
    Args:
        bed_file: BED映射文件路径
        
    Returns:
        嵌套字典，格式为 {hg19_coord: {"chromosome": chr, "start": start, "end": end}}
        其中hg19_coord是"chr1:1000-2000"格式的字符串
    """
    mapping_dict = {}
    
    print(f"加载BED映射文件: {bed_file}")
    
    # 读取BED文件
    with open(bed_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  已处理 {line_num} 行...")
                
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
                
            # BED格式: chr1 1064620 1064820 chr1:1000001-1000200 1
            hg38_chrom = parts[0]
            hg38_start = int(parts[1])
            hg38_end = int(parts[2])
            hg19_coord = parts[3]  # 这是hg19坐标，格式为"chr1:1000001-1000200"
            
            # 存储映射关系
            mapping_dict[hg19_coord] = {
                "chromosome": hg38_chrom,
                "start": hg38_start,
                "end": hg38_end
            }
    
    print(f"映射字典构建完成，共 {len(mapping_dict)} 个映射条目")
    return mapping_dict

def map_coordinate(coord_str: str, mapping_dict: Dict[str, Dict[str, Tuple[int, int]]]) -> Optional[str]:
    """
    将hg19坐标映射到hg38坐标
    
    Args:
        coord_str: hg19坐标字符串，格式为 "chr1:1000-2000"
        mapping_dict: 坐标映射字典
        
    Returns:
        hg38坐标字符串，格式为 "chr1:1000-2000"，如果找不到映射则返回None
    """
    try:
        # 解析坐标
        chrom, start, end = parse_coordinate(coord_str)
        
        # 构建hg19坐标键
        hg19_key = format_coordinate(chrom, start, end)
        
        # 查找映射
        if hg19_key in mapping_dict:
            hg38_data = mapping_dict[hg19_key]
            return format_coordinate(
                hg38_data["chromosome"].replace("chr", ""),  # 移除chr前缀
                hg38_data["start"],
                hg38_data["end"]
            )
        else:
            return None
            
    except ValueError as e:
        print(f"坐标解析错误: {e}")
        return None

def process_cell_type(cell_type: str, mapping_dict: Dict[str, Dict[str, Tuple[int, int]]]) -> None:
    """
    处理单个细胞类型的index.csv文件，生成映射后的map.csv文件
    
    Args:
        cell_type: 细胞类型名称
        mapping_dict: 坐标映射字典
    """
    cell_type_dir = DATA_DIR / cell_type
    if not cell_type_dir.exists():
        print(f"细胞类型目录不存在: {cell_type_dir}")
        return
    
    # 查找index.csv文件
    index_file = cell_type_dir / "index.csv"
    if not index_file.exists():
        print(f"未找到 {cell_type} 的index.csv文件")
        return
    
    # 输出文件
    map_file = cell_type_dir / "map.csv"
    
    print(f"处理 {cell_type} 的index.csv文件: {index_file}")
    
    try:
        # 读取index.csv文件
        df = pd.read_csv(index_file)
        
        # 检查必要的列是否存在
        if 'enhancer_name' not in df.columns or 'promoter_name' not in df.columns:
            print(f"{cell_type} 的index.csv文件缺少必要的列")
            return
        
        # 创建新的DataFrame用于存储映射结果
        mapped_data = []
        unmapped_count = 0
        
        # 处理每一行数据
        for _, row in df.iterrows():
            enhancer_hg19 = row['enhancer_name']
            promoter_hg19 = row['promoter_name']
            label = row['label']
            
            # 映射enhancer坐标
            enhancer_hg38 = map_coordinate(enhancer_hg19, mapping_dict)
            if enhancer_hg38 is None:
                unmapped_count += 1
                
            # 映射promoter坐标
            promoter_hg38 = map_coordinate(promoter_hg19, mapping_dict)
            if promoter_hg38 is None:
                unmapped_count += 1
            
            # 添加到映射数据
            mapped_data.append({
                'enhancer_name': enhancer_hg38 if enhancer_hg38 else enhancer_hg19,
                'promoter_name': promoter_hg38 if promoter_hg38 else promoter_hg19,
                'label': label
            })
        
        # 创建新的DataFrame
        mapped_df = pd.DataFrame(mapped_data)
        
        # 保存映射结果
        mapped_df.to_csv(map_file, index=False)
        
        print(f"  映射完成，共处理 {len(df)} 行")
        print(f"  未成功映射的坐标数: {unmapped_count}")
        print(f"  结果已保存到: {map_file}")
        
    except Exception as e:
        print(f"处理 {cell_type} 的index.csv文件时出错: {str(e)}")

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
        if item.is_dir() and not item.name.startswith('.') and item.name != 'output':
            # 检查是否存在index.csv文件
            if (item / "index.csv").exists():
                cell_types.append(item.name)
    
    return sorted(cell_types)

def main() -> None:
    """
    主函数：处理所有细胞类型的index.csv文件，进行hg19到hg38的坐标映射
    """
    print("开始基因组坐标映射...")
    
    # 检查BED映射文件
    if not BED_MAPPING_FILE.exists():
        print(f"错误: BED映射文件不存在: {BED_MAPPING_FILE}")
        return
    
    # 加载BED映射文件
    mapping_dict = load_bed_mapping(BED_MAPPING_FILE)
    
    # 查找所有细胞类型
    cell_types = find_cell_types()
    if not cell_types:
        print("未找到任何细胞类型目录")
        return
    
    print(f"找到 {len(cell_types)} 个细胞类型: {', '.join(cell_types)}")
    
    # 处理每个细胞类型
    for cell_type in cell_types:
        print(f"\n处理细胞类型: {cell_type}")
        process_cell_type(cell_type, mapping_dict)
    
    print("\n所有映射处理完成!")

if __name__ == "__main__":
    main()