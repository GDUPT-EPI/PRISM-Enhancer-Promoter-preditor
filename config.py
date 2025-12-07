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
PRISM_SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, "save_model/f")
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

# 确保目录存在
for dir_path in [DATA_DIR, DOMAIN_KL_DIR, CACHE_DIR, SAVE_MODEL_DIR, PRISM_SAVE_MODEL_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 训练参数配置
BATCH_SIZE = 256  # 默认训练批量大小
PRISM_BATCH_SIZE = 64  # PRISM模型批量大小
EPOCH = 20  # 训练轮数
LEARNING_RATE = 5e-5  # 学习率
GRAD_CLIP_MAX_NORM = 1.2  # 梯度裁剪上限

# 数据加载器配置
NUM_WORKERS = 4  # 从16减少到4，降低CPU上下文切换开销
PREFETCH_FACTOR = 2  # 从32减少到2，降低内存占用
PERSISTENT_WORKERS = True  # 避免worker重复创建

# 模型参数配置
EMBEDDING_DIM = 768  # 嵌入维度
CNN_KERNEL_SIZE = 40  # 卷积核大小
POOL_KERNEL_SIZE = 20  # 池化核大小
OUT_CHANNELS = 64  # 卷积输出通道数

# Transformer相关参数 - 集中管理避免冲突
TRANSFORMER_LAYERS = 4  # 每个enhancer/promoter的transformer层数
TRANSFORMER_HEADS = 8   # 多头注意力头数
TRANSFORMER_FF_DIM = 128  # 前馈网络维度 (从128修改为256以匹配checkpoint)

# CNN和分类头参数 - 集中管理避免冲突
CNN_DROPOUT = 0.35      # CNN层dropout率

# 交叉注意力和序列池化后的Dropout参数
CROSS_ATTN_DROPOUT = 0.2  # 交叉注意力层后的dropout率
SEQ_POOL_DROPOUT = 0.15   # 序列池化后的dropout率
CLASSIFIER_DROPOUT = 0.1  # 分类器前的dropout率

# 优化器配置
WEIGHT_DECAY = 0.001

# 序列长度配置
MAX_ENHANCER_LENGTH = 1000  # 固定增强子序列长度
MAX_PROMOTER_LENGTH = 4000  # 固定启动子序列长度

# 特征维度配置
ENHANCER_FEATURE_DIM = (5, 3)  # 5个特征，每个特征3维
PROMOTER_FEATURE_DIM = (5, 4)  # 5个特征，每个特征4维


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

# DNA嵌入层与词表配置
# 采用 4^K + 1(null) + n 的方式扩展词表，以支持随机掩码与特殊符号
SPECIAL_TOKENS_ORDER = ["MASK", "CLS", "SEP"]
NUM_EXTRA_TOKENS = len(SPECIAL_TOKENS_ORDER)
DNA_EMBEDDING_VOCAB_SIZE = (4 ** KMER_SIZE) + 1 + NUM_EXTRA_TOKENS
DNA_EMBEDDING_DIM = 768  # DNA嵌入维度，与EMBEDDING_DIM保持一致
DNA_EMBEDDING_PADDING_IDX = 0  # padding token的索引（null 保持为0）
DNA_EMBEDDING_INIT_STD = 0.1  # 嵌入权重初始化标准差

# 约定：特殊token按顺序追加在 4^K + 1(null) 之后，便于索引推导
MASK_TOKEN_ID = (4 ** KMER_SIZE) + 1
CLS_TOKEN_ID = MASK_TOKEN_ID + 1
SEP_TOKEN_ID = MASK_TOKEN_ID + 2

# 预处理优化配置
PREPROCESS_NUM_WORKERS = max(1, os.cpu_count() - 1)  # 使用所有核心-1
PREPROCESS_BATCH_SIZE = 2000  # 批处理大小，根据内存调整
USE_LAZY_LOADING = False  # 是否使用懒加载模式

PREPROCESS_NUM_THREADS = 12  # 根据CPU调整

# 批处理大小（用于进度显示，不影响内存）
PREPROCESS_BATCH_SIZE = 2000

# 是否启用预处理优化
ENABLE_FAST_PREPROCESSING = True

# 预处理配置信息将通过日志系统输出（在main文件中初始化）

# =========================================================================
# PRISM模型配置
# =========================================================================
PRISM_IMG_SIZE = 16  # CBAT模块图像尺寸

# =========================================================================
# 旁路解耦模块配置
# =========================================================================
# 不使用命令行参数，所有超参数集中在此处
BYPASS_LATENT_DIM = 1024  # 旁路潜变量总维度 z = [G,F,I]
BYPASS_SPEC_WEIGHT = 1.0  # 特性鉴别损失权重
BYPASS_INV_WEIGHT = 1.0   # 共性对抗损失权重
BYPASS_ORTHO_WEIGHT = 0.2 # 正交性约束权重
BYPASS_CONSIST_WEIGHT = 0.2 # 一致性约束权重
BYPASS_GRL_ALPHA = 1.0    # 梯度反转层系数
BYPASS_ROPE_LAYERS = 4    # RoPE自注意层数
BYPASS_ROPE_HEADS = TRANSFORMER_HEADS  # 复用主干的头数
BYPASS_LEARNING_RATE = 1e-4  # 旁路优化器学习率
BYPASS_WEIGHT_DECAY = 1e-4   # 旁路优化器权重衰减
BYPASS_EPOCHS = 5            # 旁路训练轮数（快速验证）
BYPASS_BATCH_SIZE = 16       # 旁路训练批量大小（降低以提升迭代速度）
BYPASS_MAX_BATCHES_PER_EPOCH = None  # 每轮最多迭代批次数（None表示不限制）
BYPASS_TSNE_POINTS = 600      # 每个epoch用于t-SNE的采样点数（G/F各）
BYPASS_CONSIST_PAIRS = 64     # 每个epoch一致性对数量（相同序列）
BYPASS_RANDOM_PAIRS = 64      # 每个epoch随机对数量（对比分布）
BYPASS_PLOT_TSNE = True       # 是否绘制t-SNE
BYPASS_PLOT_CONFUSION = True  # 是否绘制混淆矩阵

# 图上下文与GCN配置
GCN_PROTOS_PER_CELL = 4     # 每个细胞系的图卷积原型数量(调大增加模型容量但易过拟合，调小降低表达能力)
GCN_HIDDEN_DIM = 32          # GCN隐藏层维度(调大增强特征表达能力但增加计算量，调小可能导致欠拟合)
GCN_LAYERS = 2               # GCN层数(调大增强特征抽象层次但易梯度消失，调小降低模型深度)
GCN_MOMENTUM = 0.1           # 原型更新动量系数(调大增强原型稳定性但降低适应性，调小提高响应速度但易震荡)
GCN_SIM_TAU = 0.5            # 相似度温度参数(调大使相似度分布更平滑，调小使分布更尖锐)
GCN_KNN_K = 64                # K近邻图的K值(调大增加图连接密度但可能引入噪声，调小使图更稀疏)
GCN_SMOOTH_LOSS_W = 0.2      # 平滑损失权重(调大增强特征平滑性但可能过度平滑，调小降低约束强度)
GCN_CENTER_LOSS_W = 0.5      # 中心损失权重(调大增强类内紧凑性但可能过度压缩，调小降低聚类效果)
GCN_MARGIN = 1.0             # 边际损失阈值(调大增加类间距离要求，调小降低分离要求)
GCN_MARGIN_LOSS_W = 0.5      # 边际损失权重(调大增强类间分离性但可能过度分离，调小降低分离效果)

# ISAB模块配置
ISAB_NUM_INDUCING = 32  # 诱导点数量
ISAB_DROPOUT = 0.1      # ISAB dropout

# 批级上下文规模与上下文自注意层数
CONTEXT_BATCH_SIZE = 128
CONTEXT_ATTENTION_LAYERS = 2

# 随机掩码配置（位置依赖，平滑过渡）
MASK_ENABLED = True
MASK_PROB = 0.08  # 全局平均掩码概率（约8%）
MASK_CENTER_MAX_PROB = 0.10  # 中心区域最大掩码概率上限
MASK_CENTER_REGION_RATIO = 0.5  # 中心区域相对于序列长度的比例（50%）
MASK_EDGE_SIGMA = 0.25  # 边缘高斯宽度（相对化坐标系的sigma）
