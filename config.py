"""
集中配置文件
统一管理所有超参数和路径，符合项目规则中的"配置集中管理"要求
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
# MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, "_cache")
SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, "save_model/test-skip")
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

# 确保目录存在
for dir_path in [DATA_DIR, CACHE_DIR, SAVE_MODEL_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 训练参数配置
BATCH_SIZE = 32  # 从8增加到32，提高GPU利用率
EPOCH = 20
PRE_TRAIN_EPOCH = 10
LEARNING_RATE = 2e-4
VALIDATION_INTERVAL = 2  # 每隔多少个epoch进行一次验证

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

# 文件路径配置
DNABERT_VERSION = "6"  # 版本号
EMBEDDING_MATRIX_PATH = os.path.join(PROJECT_ROOT, "dnabert{}_matrix.npy".format(DNABERT_VERSION))
PRETRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, "premodel_DNABERT{}_model_pgd_gene_new_19.pt".format(DNABERT_VERSION))

# 细胞系配置
TRAIN_CELL_LINE = "HUVEC"  # 选择单一细胞系或全选
TEST_CELL_LINES = ["GM12878", "IMR90", "HeLa-S3", "HUVEC", "K562", "NHEK"]

# 可视化配置
TSNE_PERPLEXITY = 10
TSNE_RANDOM_STATE = 42

# 设备配置
DEVICE = "cuda"  # 优先使用GPU

# 调试和日志配置
DEBUG_MODE = False
SAVE_ATTENTION_OUTPUTS = False


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