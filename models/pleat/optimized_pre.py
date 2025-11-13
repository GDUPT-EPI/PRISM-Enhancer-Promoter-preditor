#!/usr/bin/env python3
"""
优化的数据预处理模块
解决原有预处理流程中的性能瓶颈，提高数据处理效率
"""

import torch
import numpy as np
import itertools
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import pickle
from pathlib import Path

# 导入配置
from config import (
    CACHE_DIR,
    MAX_ENHANCER_LENGTH, 
    MAX_PROMOTER_LENGTH,
    ENHANCER_FEATURE_DIM,
    PROMOTER_FEATURE_DIM,
    PROJECT_ROOT,
    CACHE_DIR,
    KMER_SIZE,
    KMER_OVERLAP
)

# 全局tokenizer缓存
_TOKENIZER_CACHE = None
_TOKENIZER_CACHE_PATH = os.path.join(CACHE_DIR, "tokenizer_cache.pkl")

def get_tokenizer(force_recreate: bool = False) -> Dict[str, int]:
    """
    获取全局tokenizer字典，使用缓存避免重复计算
    与embedding.py中的KMerTokenizer保持一致
    
    Args:
        force_recreate: 是否强制重新创建tokenizer
        
    Returns:
        token到索引的映射字典
    """
    global _TOKENIZER_CACHE
    
    # 如果缓存存在且不强制重新创建，直接返回
    if _TOKENIZER_CACHE is not None and not force_recreate:
        return _TOKENIZER_CACHE
    
    # 尝试从文件加载缓存
    cache_dir = os.path.dirname(_TOKENIZER_CACHE_PATH)
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(_TOKENIZER_CACHE_PATH) and not force_recreate:
        try:
            with open(_TOKENIZER_CACHE_PATH, 'rb') as f:
                _TOKENIZER_CACHE = pickle.load(f)
            print(f"从缓存加载tokenizer: {_TOKENIZER_CACHE_PATH}")
            return _TOKENIZER_CACHE
        except Exception as e:
            print(f"加载tokenizer缓存失败: {e}，将重新创建")
    
    # 创建新的tokenizer，与embedding.py中的KMerTokenizer保持一致
    print("创建tokenizer字典...")
    bases = ['A', 'C', 'G', 'T']
    k = KMER_SIZE  # 从config.py导入的6-mer配置
    
    # 使用itertools.product生成所有可能的6-mer组合
    products = itertools.product(bases, repeat=k)
    tokens = [''.join(p) for p in products]
    
    # 添加null token
    tokens.append('null')
    
    # 创建token到索引的映射字典
    token_dict = {token: idx for idx, token in enumerate(tokens)}  # null token的索引为4096
    
    _TOKENIZER_CACHE = token_dict
    
    # 保存到缓存文件
    try:
        with open(_TOKENIZER_CACHE_PATH, 'wb') as f:
            pickle.dump(_TOKENIZER_CACHE, f)
        print(f"保存tokenizer到缓存: {_TOKENIZER_CACHE_PATH}")
    except Exception as e:
        print(f"保存tokenizer缓存失败: {e}")
    
    return _TOKENIZER_CACHE


def sequence_to_tokens_fast(sequence: str, k: int = KMER_SIZE) -> List[str]:
    """
    高效地将DNA序列转换为k-mer tokens
    使用overlapping方式，每次滑动k-1个碱基
    
    Args:
        sequence: DNA序列字符串
        k: k-mer长度，默认使用config.py中的KMER_SIZE
        
    Returns:
        k-mer tokens列表
    """
    # 预分配列表大小，提高性能
    seq_len = len(sequence)
    if seq_len < k:
        return ['null']
    
    # 使用overlapping方式生成k-mers，每次滑动k-1个碱基
    tokens = []
    for i in range(seq_len - k + 1):
        kmer = sequence[i:i+k]
        # 检查是否包含未知碱基N
        if 'N' in kmer:
            tokens.append('null')
        else:
            tokens.append(kmer)
    
    return tokens


def tokens_to_ids_fast(tokens: List[str], tokenizer: Dict[str, int]) -> torch.Tensor:
    """
    高效地将tokens转换为ID张量
    
    Args:
        tokens: token列表
        tokenizer: tokenizer字典
        
    Returns:
        token ID张量
    """
    # 使用列表推导式和字典查找，比循环更高效
    token_ids = [tokenizer.get(token, 0) for token in tokens]
    return torch.tensor(token_ids, dtype=torch.long)


class OptimizedSequenceDataset(Dataset):
    """
    优化的序列数据集类，采用延迟预处理和缓存策略
    
    Args:
        enhancers: 增强子序列列表
        promoters: 启动子序列列表
        labels: 标签列表
        cache_dir: 缓存目录，用于存储预处理后的数据
        use_cache: 是否使用缓存
    """
    
    def __init__(
        self, 
        enhancers: List[str], 
        promoters: List[str], 
        labels: List[int],
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        self.enhancers = enhancers
        self.promoters = promoters
        self.labels = labels
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # 验证数据长度一致性
        assert len(enhancers) == len(promoters) == len(labels), \
            "增强子、启动子和标签的长度必须一致"
        
        self.length = len(labels)
        
        # 获取tokenizer
        self.tokenizer = get_tokenizer()
        
        # 创建缓存目录
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # 预计算特征张量，避免重复创建
        self.enhancer_features = torch.zeros(*ENHANCER_FEATURE_DIM)
        self.promoter_features = torch.zeros(*PROMOTER_FEATURE_DIM)
        
        # 初始化缓存索引
        self._cache_index = set()
        if self.use_cache and self.cache_dir:
            self._load_cache_index()
    
    def _load_cache_index(self):
        """加载缓存索引"""
        index_file = os.path.join(self.cache_dir, "cache_index.pkl")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'rb') as f:
                    self._cache_index = pickle.load(f)
            except Exception as e:
                print(f"加载缓存索引失败: {e}")
                self._cache_index = set()
    
    def _save_cache_index(self):
        """保存缓存索引"""
        if not self.use_cache or not self.cache_dir:
            return
            
        index_file = os.path.join(self.cache_dir, "cache_index.pkl")
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self._cache_index, f)
        except Exception as e:
            print(f"保存缓存索引失败: {e}")
    
    def _get_cache_path(self, idx: int) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"data_{idx}.pt")
    
    def _process_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        处理单个数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            处理后的数据元组 (enhancer_ids, promoter_ids, enhancer_features, promoter_features, label)
        """
        # 获取原始数据
        enhancer_seq = self.enhancers[idx]
        promoter_seq = self.promoters[idx]
        label = self.labels[idx]
        
        # 转换为tokens
        enhancer_tokens = sequence_to_tokens_fast(enhancer_seq)
        promoter_tokens = sequence_to_tokens_fast(promoter_seq)
        
        # 转换为ID张量
        enhancer_ids = tokens_to_ids_fast(enhancer_tokens, self.tokenizer)
        promoter_ids = tokens_to_ids_fast(promoter_tokens, self.tokenizer)
        
        # 截断过长的序列
        if len(enhancer_ids) > MAX_ENHANCER_LENGTH:
            enhancer_ids = enhancer_ids[:MAX_ENHANCER_LENGTH]
        if len(promoter_ids) > MAX_PROMOTER_LENGTH:
            promoter_ids = promoter_ids[:MAX_PROMOTER_LENGTH]
        
        # 克隆特征张量，避免所有样本共享同一个张量
        enhancer_features = self.enhancer_features.clone()
        promoter_features = self.promoter_features.clone()
        
        return enhancer_ids, promoter_ids, enhancer_features, promoter_features, label
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        获取数据项，使用缓存提高性能
        
        Args:
            idx: 数据索引
            
        Returns:
            数据元组 (enhancer_ids, promoter_ids, enhancer_features, promoter_features, label)
        """
        # 如果使用缓存且缓存中存在该数据，直接加载
        if self.use_cache and self.cache_dir and idx in self._cache_index:
            cache_path = self._get_cache_path(idx)
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    print(f"加载缓存数据失败: {e}")
        
        # 处理数据
        data = self._process_item(idx)
        
        # 保存到缓存
        if self.use_cache and self.cache_dir:
            cache_path = self._get_cache_path(idx)
            try:
                torch.save(data, cache_path)
                self._cache_index.add(idx)
                # 每100个样本保存一次索引，减少IO操作
                if idx % 100 == 0:
                    self._save_cache_index()
            except Exception as e:
                print(f"保存缓存数据失败: {e}")
        
        return data
    
    def __len__(self) -> int:
        return self.length


def create_optimized_dataset(
    enhancers: List[str], 
    promoters: List[str], 
    labels: List[int],
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    num_workers: int = 4
) -> OptimizedSequenceDataset:
    """
    创建优化的数据集，支持并行预处理
    
    Args:
        enhancers: 增强子序列列表
        promoters: 启动子序列列表
        labels: 标签列表
        cache_dir: 缓存目录
        use_cache: 是否使用缓存
        num_workers: 预处理工作进程数
        
    Returns:
        优化后的数据集
    """
    # 如果没有指定缓存目录，创建默认缓存目录
    if cache_dir is None and use_cache:
        cache_dir = os.path.join(CACHE_DIR, "dataset_cache")
    
    # 创建数据集
    dataset = OptimizedSequenceDataset(
        enhancers=enhancers,
        promoters=promoters,
        labels=labels,
        cache_dir=cache_dir,
        use_cache=use_cache
    )
    
    return dataset


def clear_tokenizer_cache():
    """清除tokenizer缓存"""
    global _TOKENIZER_CACHE
    _TOKENIZER_CACHE = None
    if os.path.exists(_TOKENIZER_CACHE_PATH):
        os.remove(_TOKENIZER_CACHE_PATH)
        print(f"已删除tokenizer缓存: {_TOKENIZER_CACHE_PATH}")


def warmup_cache(dataset: OptimizedSequenceDataset, num_samples: int = 100):
    """
    预热缓存，提前处理一部分数据
    
    Args:
        dataset: 数据集
        num_samples: 预处理的样本数量
    """
    print(f"预热缓存，处理前{num_samples}个样本...")
    for i in tqdm(range(min(num_samples, len(dataset))), desc="预热缓存"):
        _ = dataset[i]
    print("缓存预热完成")