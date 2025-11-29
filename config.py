"""
集中配置文件
统一管理所有超参数和路径，符合项目规则中的"配置集中管理"要求
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
DOMAIN_KL_DIR = os.path.join(PROJECT_ROOT, "domain-kl")  # PRISM特供数据目录
# MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, "_cache")
SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, "save_model/CBAT")
PRISM_SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, "save_model/prism")
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

# 确保目录存在
for dir_path in [DATA_DIR, DOMAIN_KL_DIR, CACHE_DIR, SAVE_MODEL_DIR, PRISM_SAVE_MODEL_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 训练参数配置
BATCH_SIZE = 32  # 从8增加到32，提高GPU利用率
PRISM_BATCH_SIZE = 16  # PRISM特供batch size (会被分为8个同细胞系 + 8个不同细胞系)
EPOCH = 20
LEARNING_RATE = 2e-4
VALIDATION_INTERVAL = 1  # 每隔多少个epoch进行一次验证

# 数据加载器配置
NUM_WORKERS = 4  # 从16减少到4，降低CPU上下文切换开销
PREFETCH_FACTOR = 2  # 从32减少到2，降低内存占用
PERSISTENT_WORKERS = True  # 避免worker重复创建

# 模型参数配置
NUMBER_WORDS = 4097
NUMBER_POS = 70
EMBEDDING_DIM = 768
CNN_KERNEL_SIZE = 40
POOL_KERNEL_SIZE = 20
OUT_CHANNELS = 64

# Transformer相关参数 - 集中管理避免冲突
TRANSFORMER_LAYERS = 2  # 每个enhancer/promoter的transformer层数
TRANSFORMER_HEADS = 8   # 多头注意力头数
TRANSFORMER_FF_DIM = 256  # 前馈网络维度

# Stochastic Depth配置
STOCHASTIC_DEPTH_RATE = 0.2  # 最大drop概率

# CNN和分类头参数 - 集中管理避免冲突
CNN_DROPOUT = 0.2      # CNN层dropout率
CLASSIFIER_HIDDEN_SIZE = 128  # 分类头中间层大小
CLASSIFIER_DROPOUT = 0.33     # 分类头dropout率

# 优化器配置
WEIGHT_DECAY = 0.001

# 序列长度配置
MAX_ENHANCER_LENGTH = 1000  # 固定增强子序列长度
MAX_PROMOTER_LENGTH = 4000  # 固定启动子序列长度

# 特征维度配置
ENHANCER_FEATURE_DIM = (5, 3)  # 5个特征，每个特征3维
PROMOTER_FEATURE_DIM = (5, 4)  # 5个特征，每个特征4维

# 对抗训练参数
PGD_EPSILON = 1.0
PGD_ALPHA = 0.3
FGM_EPSILON = 1.0
ENABLE_ADVERSARIAL_TRAINING = True  # 是否启用对抗训练
K = 3  # PGD攻击迭代次数

# 细胞系配置
TRAIN_CELL_LINE = "ALL"  # 选择单一细胞系或全选
TEST_CELL_LINES = ["GM12878", "IMR90", "HeLa-S3", "HUVEC", "K562", "NHEK"]

# 可视化配置
TSNE_PERPLEXITY = 10
TSNE_RANDOM_STATE = 42

# 设备配置
DEVICE = "cuda"  # 优先使用GPU

# 调试和日志配置
DEBUG_MODE = False
SAVE_ATTENTION_OUTPUTS = False

# K-mer配置
KMER_SIZE = 6  # 6-mer tokenization
KMER_OVERLAP = 5  # overlapping k-mer (k-1)

# DNA嵌入层配置
DNA_EMBEDDING_VOCAB_SIZE = 4097  # 6-mer词汇表大小 (4^6 + 1 null token)
DNA_EMBEDDING_DIM = 768  # DNA嵌入维度，与EMBEDDING_DIM保持一致
DNA_EMBEDDING_PADDING_IDX = 0  # padding token的索引
DNA_EMBEDDING_INIT_STD = 0.1  # 嵌入权重初始化标准差

# 预处理优化配置
PREPROCESS_NUM_WORKERS = max(1, os.cpu_count() - 1)  # 使用所有核心-1
PREPROCESS_BATCH_SIZE = 2000  # 批处理大小，根据内存调整
USE_LAZY_LOADING = False  # 是否使用懒加载模式

PREPROCESS_NUM_THREADS = 12  # 根据你的CPU调整

# 批处理大小（用于进度显示，不影响内存）
PREPROCESS_BATCH_SIZE = 2000

# 是否启用预处理优化
ENABLE_FAST_PREPROCESSING = True

# 预处理配置信息将通过日志系统输出（在main文件中初始化）

# ============================================================================
# BERT预训练配置
# ============================================================================
BERT_MASK_PROB = 0.15  # 15%的token被mask
BERT_MASK_TOKEN_PROB = 0.8  # 80%替换为[MASK]
BERT_RANDOM_TOKEN_PROB = 0.1  # 10%替换为随机token
BERT_UNCHANGED_PROB = 0.1  # 10%保持不变

# BERT特殊token - 注意vocab_size=4097，有效索引是0-4096
BERT_MASK_TOKEN_ID = 4096  # [MASK] token ID (vocab最后一个位置)
BERT_PAD_TOKEN_ID = 0  # [PAD] token ID
BERT_CLS_TOKEN_ID = 4095  # [CLS] token ID (可选)
BERT_SEP_TOKEN_ID = 4094  # [SEP] token ID (可选)

# BERT训练参数
BERT_LEARNING_RATE = 1e-4
BERT_WARMUP_STEPS = 1000
BERT_MAX_GRAD_NORM = 1.0

# PRISM模型配置（简化版本）
PRISM_USE_CROSS_ATTENTION = True  # 是否使用交叉注意力
PRISM_POOLING_TYPE = "mean"  # 池化方式: "mean", "max", "cls"