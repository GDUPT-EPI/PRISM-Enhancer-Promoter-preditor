#!/usr/bin/env python3
"""
词表管理模块
提供唯一、确定的词表生成和管理功能
"""

import os
import hashlib
import pickle
import itertools
from typing import Dict, Tuple, Optional

# 导入配置
from config import KMER_SIZE, PROJECT_ROOT, SPECIAL_TOKENS_ORDER


class VocabManager:
    """
    词表管理器，确保词表的唯一性和一致性
    """
    _instance = None
    _vocab = None
    _vocab_hash = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VocabManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 确保词表目录存在
        self.vocab_dir = os.path.join(PROJECT_ROOT, "vocab")
        os.makedirs(self.vocab_dir, exist_ok=True)
        
        # 词表和哈希文件路径
        self.tokenizer_path = os.path.join(self.vocab_dir, "tokenizer.pkl")
        self.hash_path = os.path.join(self.vocab_dir, "hash.key")
    
    def _generate_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        生成确定性的DNA k-mer词表，并在 4^K + 1(null) 基础上追加特殊token
        保持固定顺序和索引，以确保哈希验证稳定可靠。

        Returns:
            token_to_idx: token到索引的映射字典
            idx_to_token: 索引到token的映射字典
        """
        # 基础核苷酸集合，固定顺序保证确定性
        bases = ['A', 'C', 'G', 'T']

        # 生成所有k-mer，按字典序排列（itertools.product 已保证固定迭代顺序）
        products = itertools.product(bases, repeat=KMER_SIZE)
        core_tokens = [''.join(p) for p in products]

        # 将 null token 固定插入索引0
        tokens = ['null'] + core_tokens

        # 追加特殊token，采用集中配置的顺序 SPECIAL_TOKENS_ORDER
        for name in SPECIAL_TOKENS_ORDER:
            tokens.append(f'<{name}>')

        # 创建映射
        token_to_idx = {token: idx for idx, token in enumerate(tokens)}
        idx_to_token = {idx: token for token, idx in token_to_idx.items()}

        return token_to_idx, idx_to_token
    
    def _calculate_vocab_hash(self, token_to_idx: Dict[str, int]) -> str:
        """
        计算词表的哈希值
        
        Args:
            token_to_idx: token到索引的映射字典
            
        Returns:
            词表的MD5哈希值
        """
        # 哈希内容包含：按token排序的映射 + k值 + 特殊token顺序
        sorted_items = sorted(token_to_idx.items())
        vocab_str = str({'items': sorted_items, 'k': KMER_SIZE, 'specials': SPECIAL_TOKENS_ORDER})
        
        # 计算MD5哈希
        return hashlib.md5(vocab_str.encode()).hexdigest()
    
    def _save_vocab(self, token_to_idx: Dict[str, int], idx_to_token: Dict[int, str]) -> None:
        """
        保存词表到文件
        
        Args:
            token_to_idx: token到索引的映射字典
            idx_to_token: 索引到token的映射字典
        """
        # 保存词表（包含特殊token顺序以便校验）
        vocab_data = {
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'kmer_size': KMER_SIZE,
            'special_tokens_order': SPECIAL_TOKENS_ORDER,
        }
        
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        # 计算并保存哈希值
        vocab_hash = self._calculate_vocab_hash(token_to_idx)
        with open(self.hash_path, 'w') as f:
            f.write(vocab_hash)
        
        print(f"词表已保存到: {self.tokenizer_path}")
        print(f"词表哈希值已保存到: {self.hash_path}")
        print(f"词表哈希值: {vocab_hash}")
    
    def _load_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        从文件加载词表
        
        Returns:
            token_to_idx: token到索引的映射字典
            idx_to_token: 索引到token的映射字典
        """
        with open(self.tokenizer_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        token_to_idx = vocab_data['token_to_idx']
        idx_to_token = vocab_data['idx_to_token']
        
        print(f"词表已从文件加载: {self.tokenizer_path}")
        
        return token_to_idx, idx_to_token
    
    def _load_hash(self) -> str:
        """
        从文件加载哈希值
        
        Returns:
            词表的哈希值
        """
        with open(self.hash_path, 'r') as f:
            vocab_hash = f.read().strip()
        
        print(f"词表哈希值已从文件加载: {self.hash_path}")
        print(f"词表哈希值: {vocab_hash}")
        
        return vocab_hash
    
    def get_vocab(self, force_recreate: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        获取词表，优先从缓存加载，如果不存在或哈希不匹配则重新生成
        
        Args:
            force_recreate: 是否强制重新创建词表
            
        Returns:
            token_to_idx: token到索引的映射字典
            idx_to_token: 索引到token的映射字典
        """
        # 如果已加载且不强制重新创建，直接返回
        if self._vocab is not None and not force_recreate:
            return self._vocab
        
        # 检查词表文件和哈希文件是否存在
        tokenizer_exists = os.path.exists(self.tokenizer_path)
        hash_exists = os.path.exists(self.hash_path)
        
        # 如果哈希文件存在但词表文件不存在，报错
        if hash_exists and not tokenizer_exists:
            raise FileNotFoundError(f"哈希文件存在但词表文件不存在: {self.hash_path} vs {self.tokenizer_path}")
        
        # 如果两个文件都存在，尝试加载
        if tokenizer_exists and hash_exists and not force_recreate:
            try:
                # 加载词表
                token_to_idx, idx_to_token = self._load_vocab()
                
                # 加载哈希值
                saved_hash = self._load_hash()
                
                # 计算当前词表的哈希值
                current_hash = self._calculate_vocab_hash(token_to_idx)
                
                # 验证哈希值是否匹配
                if saved_hash == current_hash:
                    self._vocab = (token_to_idx, idx_to_token)
                    self._vocab_hash = saved_hash
                    return self._vocab
                else:
                    print(f"警告: 词表哈希值不匹配! 已保存: {saved_hash}, 当前: {current_hash}")
                    print("将重新生成词表...")
            except Exception as e:
                print(f"加载词表失败: {e}")
                print("将重新生成词表...")
        
        # 生成新的词表
        print("生成新的词表...")
        token_to_idx, idx_to_token = self._generate_vocab()
        
        # 如果哈希文件不存在，保存新的词表和哈希
        if not hash_exists:
            self._save_vocab(token_to_idx, idx_to_token)
        else:
            # 哈希文件存在但词表文件不存在或哈希不匹配
            print("警告: 哈希文件已存在，不能覆盖。使用内存中的词表。")
            print("如需重新生成词表，请删除哈希文件后重试。")
        
        self._vocab = (token_to_idx, idx_to_token)
        self._vocab_hash = self._calculate_vocab_hash(token_to_idx)
        
        return self._vocab
    
    def get_vocab_hash(self) -> str:
        """
        获取当前词表的哈希值
        
        Returns:
            词表的哈希值
        """
        if self._vocab_hash is None:
            if self._vocab is None:
                self.get_vocab()  # 确保词表已加载
            else:
                token_to_idx, _ = self._vocab
                self._vocab_hash = self._calculate_vocab_hash(token_to_idx)
        
        return self._vocab_hash


# 全局词表管理器实例
_vocab_manager = None


def get_vocab_manager() -> VocabManager:
    """
    获取全局词表管理器实例
    
    Returns:
        词表管理器实例
    """
    global _vocab_manager
    if _vocab_manager is None:
        _vocab_manager = VocabManager()
    return _vocab_manager


def get_vocab(force_recreate: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    获取词表，优先从缓存加载
    
    Args:
        force_recreate: 是否强制重新创建词表
        
    Returns:
        token_to_idx: token到索引的映射字典
        idx_to_token: 索引到token的映射字典
    """
    return get_vocab_manager().get_vocab(force_recreate)


def get_token_to_idx(force_recreate: bool = False) -> Dict[str, int]:
    """
    获取token到索引的映射字典
    
    Args:
        force_recreate: 是否强制重新创建词表
        
    Returns:
        token到索引的映射字典
    """
    token_to_idx, _ = get_vocab(force_recreate)
    return token_to_idx


def get_idx_to_token(force_recreate: bool = False) -> Dict[int, str]:
    """
    获取索引到token的映射字典
    
    Args:
        force_recreate: 是否强制重新创建词表
        
    Returns:
        索引到token的映射字典
    """
    _, idx_to_token = get_vocab(force_recreate)
    return idx_to_token


def get_vocab_hash() -> str:
    """
    获取词表的哈希值
    
    Returns:
        词表的哈希值
    """
    return get_vocab_manager().get_vocab_hash()
