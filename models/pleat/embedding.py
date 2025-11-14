#!/usr/bin/env python3
"""
6-mer重叠嵌入层模块
实现基于DNA序列的6-mer tokenization和嵌入向量生成
"""

import torch
import torch.nn as nn
import itertools
from typing import Dict, List, Tuple
import numpy as np

# 导入配置参数
from config import KMER_SIZE, EMBEDDING_DIM


class KMerTokenizer:
    """
    6-mer分词器类，用于将DNA序列转换为k-mer tokens
    
    Attributes:
        token_to_idx (Dict[str, int]): token到索引的映射字典
        idx_to_token (Dict[int, str]): 索引到token的映射字典
        vocab_size (int): 词汇表大小
    """
    
    def __init__(self, k: int = KMER_SIZE):
        """
        初始化k-mer分词器
        
        Args:
            k: k-mer长度，默认使用配置文件中的KMER_SIZE
        """
        self.k = k
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab_size = 0
        self._build_vocab()
    
    def _build_vocab(self) -> None:
        """
        构建k-mer词汇表
        包含所有可能的k-mer组合以及null token
        """
        # DNA碱基
        bases = ['A', 'C', 'G', 'T']
        
        # 生成所有可能的k-mer组合
        products = itertools.product(bases, repeat=self.k)
        tokens = [''.join(p) for p in products]
        
        # 添加null token用于填充和未知碱基
        tokens.insert(0, 'null')
        
        # 创建token到索引的映射字典
        self.token_to_idx = {token: idx for idx, token in enumerate(tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(tokens)
        
        print(f"构建了{self.vocab_size}大小的{self.k}-mer词汇表")
    
    def tokenize(self, sequence: str) -> List[str]:
        """
        将DNA序列转换为k-mer tokens列表
        使用overlapping方式，每次滑动k-1个碱基
        
        Args:
            sequence: DNA序列字符串
            
        Returns:
            k-mer tokens列表
        """
        # 检查序列长度
        if len(sequence) < self.k:
            return ['null']
        
        # 使用overlapping方式生成k-mers，每次滑动k-1个碱基
        tokens = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            # 检查是否包含未知碱基N
            if 'N' in kmer:
                tokens.append('null')
            else:
                tokens.append(kmer)
        
        return tokens
    
    def encode(self, sequence: str) -> torch.Tensor:
        """
        将DNA序列编码为token索引张量
        
        Args:
            sequence: DNA序列字符串
            
        Returns:
            token索引张量
        """
        tokens = self.tokenize(sequence)
        indices = [self.token_to_idx.get(token, 0) for token in tokens]  # 默认为null的索引
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> List[str]:
        """
        将token索引张量解码为k-mer tokens列表
        
        Args:
            indices: token索引张量
            
        Returns:
            k-mer tokens列表
        """
        return [self.idx_to_token.get(idx.item(), 'null') for idx in indices]


class DNAEmbedding(nn.Module):
    """
    DNA序列嵌入层
    将DNA序列转换为密集向量表示
    
    Attributes:
        tokenizer (KMerTokenizer): k-mer分词器
        embedding (nn.Embedding): PyTorch嵌入层
    """
    
    def __init__(self, vocab_size: int = None, embed_dim: int = EMBEDDING_DIM, k: int = KMER_SIZE, 
                 padding_idx: int = 0, init_std: float = 0.1):
        """
        初始化DNA嵌入层
        
        Args:
            vocab_size: 词汇表大小，如果为None则使用k-mer生成的词汇表大小
            embed_dim: 嵌入维度，默认使用配置文件中的EMBEDDING_DIM
            k: k-mer长度，默认使用配置文件中的KMER_SIZE
            padding_idx: padding token的索引
            init_std: 嵌入权重初始化标准差
        """
        super(DNAEmbedding, self).__init__()
        
        # 创建k-mer分词器
        self.tokenizer = KMerTokenizer(k=k)
        
        # 如果未指定词汇表大小，使用分词器的词汇表大小
        if vocab_size is None:
            vocab_size = self.tokenizer.vocab_size
        
        # 创建嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)  # null token作为padding
        
        # 初始化嵌入权重
        self._init_weights(init_std)
    
    def _init_weights(self, init_std: float = 0.1) -> None:
        """
        初始化嵌入权重
        使用正态分布初始化，均值为0，标准差为init_std
        
        Args:
            init_std: 嵌入权重初始化标准差
        """
        nn.init.normal_(self.embedding.weight, mean=0, std=init_std)
        # 保持padding token的权重为0
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
    
    def forward(self, sequences) -> torch.Tensor:
        """
        前向传播
        
        Args:
            sequences: 可以是DNA序列字符串列表或token索引张量
            
        Returns:
            嵌入向量张量，形状为(batch_size, seq_len, embed_dim)
        """
        # 检查输入类型
        if isinstance(sequences, list) and all(isinstance(seq, str) for seq in sequences):
            # 如果输入是字符串列表，先进行tokenize
            batch_indices = []
            max_len = 0
            
            # 编码所有序列并找到最大长度
            for seq in sequences:
                indices = self.tokenizer.encode(seq)
                batch_indices.append(indices)
                max_len = max(max_len, len(indices))
            
            # 填充序列到相同长度
            padded_indices = []
            for indices in batch_indices:
                # 计算需要填充的长度
                pad_len = max_len - len(indices)
                # 创建填充张量
                if pad_len > 0:
                    padding = torch.zeros(pad_len, dtype=torch.long)
                    padded = torch.cat([indices, padding])
                else:
                    padded = indices
                padded_indices.append(padded)
            
            # 堆叠为批次张量
            batch_tensor = torch.stack(padded_indices)
        else:
            # 如果输入已经是张量（token索引），直接使用
            batch_tensor = sequences
        
        # 通过嵌入层
        embedded = self.embedding(batch_tensor)
        
        return embedded


def create_dna_embedding_layer(vocab_size: int = None, embed_dim: int = EMBEDDING_DIM, 
                               padding_idx: int = 0, init_std: float = 0.1) -> DNAEmbedding:
    """
    创建DNA嵌入层的工厂函数
    
    Args:
        vocab_size: 词汇表大小
        embed_dim: 嵌入维度
        padding_idx: padding token的索引
        init_std: 嵌入权重初始化标准差
        
    Returns:
        DNA嵌入层实例
    """
    return DNAEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, padding_idx=padding_idx, init_std=init_std)


# # 测试代码
# if __name__ == "__main__":
#     # 计时开始
#     import time
#     start_time = time.time()
    
#     # 构建词汇表
#     tokenizer = KMerTokenizer()
#     tokenizer._build_vocab()
    
#     # 计时结束
#     end_time = time.time()
#     print(f"词汇表构建时间: {end_time - start_time:.4f} 秒")
