#!/usr/bin/env python3

# 导入配置文件
from config import *

# 导入日志模块
import logging
from datetime import datetime

from torch.utils.data import DataLoader
from dataset_embedding import MyDataset
from dataset_embedding import IDSDataset

# 导入优化的数据预处理模块
from optimized_data_preprocessing import (
    create_optimized_dataset,
    get_tokenizer,
    warmup_cache,
    clear_tokenizer_cache
)
from models.EPIModel import EPIModel
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR
import torch
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# 使用配置文件中的参数
# BATCH_SIZE = 32  # 已在config.py中定义
# epoch = 15  # 已在config.py中定义为EPOCH
# pre_train_epoch = 10  # 已在config.py中定义为PRE_TRAIN_EPOCH

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# 配置日志系统
def setup_logging():
    """
    配置日志系统，同时输出到控制台和文件
    """
    # 创建日志文件名，包含时间戳
    log_filename = os.path.join(LOG_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# 设置日志
logger = setup_logging()
logger.info("日志系统已初始化，日志文件将保存到: {}".format(LOG_DIR))
logger.info("预处理配置: 使用 {} 个线程".format(PREPROCESS_NUM_THREADS))



def get_num_correct(preds, labels):
    """
    计算预测正确的样本数量
    
    Args:
        preds: 模型预测值
        labels: 真实标签
        
    Returns:
        预测正确的样本数量
    """
    predictions = (preds >= 0.5).float()
    return (predictions == labels).sum().item()


def load_dna_sequences(file_path):
    """
    从文件加载DNA序列数据
    
    Args:
        file_path: DNA序列文件路径
        
    Returns:
        DNA序列的NumPy数组
    """
    # 读取DNA序列文件
    with open(file_path, 'r') as file:
        # 逐行读取文件并保存为字符串列表
        sequences = [line.strip() for line in file]

    # 将字符串列表转换为NumPy数组
    sequences_array = np.array(sequences)

    return sequences_array


def load_labels(file_path):
    """
    从文件加载标签数据
    
    Args:
        file_path: 标签文件路径
        
    Returns:
        标签的NumPy数组
    """
    # 读取标签文件
    with open(file_path, 'r') as file:
        # 逐行读取文件并保存为整数列表
        labels = [int(line.strip()) for line in file]

    # 将整数列表转换为NumPy数组
    labels_array = np.array(labels)

    return labels_array


def simple_collate_fn(batch):
    """
    简化的collate函数，因为数据已在Dataset中预处理
    
    Args:
        batch: 包含多个样本的列表，每个样本是一个元组
              (enhancer_sequence, promoter_sequence, enhancer_features, promoter_features, label)
              
    Returns:
        tuple: 包含五个元素的元组
            - enhancer_sequences: 填充后的增强子序列张量
            - promoter_sequences: 填充后的启动子序列张量
            - enhancer_features: 增强子特征张量
            - promoter_features: 启动子特征张量
            - labels: 标签张量
    """
    # 分离各个组件
    enhancer_sequences = [item[0] for item in batch]
    promoter_sequences = [item[1] for item in batch]
    enhancer_features = [item[2] for item in batch]
    promoter_features = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    
    # 使用pad_sequence填充序列
    padded_enhancer_sequences = pad_sequence(enhancer_sequences, batch_first=True, padding_value=0)
    padded_promoter_sequences = pad_sequence(promoter_sequences, batch_first=True, padding_value=0)
    
    # 如果填充后长度仍小于最大长度，继续填充
    if padded_enhancer_sequences.size(1) < MAX_ENHANCER_LENGTH:
        padding_size = MAX_ENHANCER_LENGTH - padded_enhancer_sequences.size(1)
        padded_enhancer_sequences = torch.nn.functional.pad(
            padded_enhancer_sequences, (0, padding_size), mode='constant', value=0
        )
    
    if padded_promoter_sequences.size(1) < MAX_PROMOTER_LENGTH:
        padding_size = MAX_PROMOTER_LENGTH - padded_promoter_sequences.size(1)
        padded_promoter_sequences = torch.nn.functional.pad(
            padded_promoter_sequences, (0, padding_size), mode='constant', value=0
        )
    
    # 直接堆叠特征张量和标签
    padded_enhancer_features = torch.stack(enhancer_features)
    padded_promoter_features = torch.stack(promoter_features)
    labels = torch.tensor(labels, dtype=torch.float)
    
    return padded_enhancer_sequences, padded_promoter_sequences, padded_enhancer_features, padded_promoter_features, labels


# 导入优化的数据预处理模块
from optimized_data_preprocessing import (
    create_optimized_dataset,
    get_tokenizer,
    warmup_cache,
    clear_tokenizer_cache
)

# 原有的数据集类保留作为备用
class OptimizedCombinedDataset(Dataset):
    """
    优化的组合数据集类，使用新的预处理模块
    """
    def __init__(self, enhancers, promoters, labels, cache_dir=None, use_cache=True):
        self.dataset = create_optimized_dataset(
            enhancers=enhancers,
            promoters=promoters,
            labels=labels,
            cache_dir=cache_dir,
            use_cache=use_cache
        )
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


class PGD():
    """
    PGD对抗训练类，用于实现Projected Gradient Descent对抗攻击
    
    Args:
        model: 要进行对抗训练的模型
    """
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name1='embedding_en.', emb_name2='embedding_pr', is_first_attack=False):
        """
        执行PGD对抗攻击
        
        Args:
            epsilon: 扰动的最大范数
            alpha: 每步扰动的步长
            emb_name1: 增强子嵌入层的参数名前缀
            emb_name2: 启动子嵌入层的参数名前缀
            is_first_attack: 是否是第一次攻击
        """
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name1='embedding_en', emb_name2='embedding_pr'):
        """
        恢复嵌入层的参数
        
        Args:
            emb_name1: 增强子嵌入层的参数名前缀
            emb_name2: 启动子嵌入层的参数名前缀
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        """
        将扰动投影到指定范围内
        
        Args:
            param_name: 参数名
            param_data: 参数数据
            epsilon: 扰动的最大范数
            
        Returns:
            投影后的参数数据
        """
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        """备份梯度"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        """恢复梯度"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FGM():
    """
    FGM对抗训练类，用于实现Fast Gradient Method对抗攻击
    
    Args:
        model: 要进行对抗训练的模型
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='embedding_en', emb_name2='embedding_pr'):
        """
        执行FGM对抗攻击
        
        Args:
            epsilon: 扰动的最大范数
            emb_name1: 增强子嵌入层的参数名前缀
            emb_name2: 启动子嵌入层的参数名前缀
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name1='embedding_en', emb_name2='embedding_pr'):
        """
        恢复嵌入层的参数
        
        Args:
            emb_name1: 增强子嵌入层的参数名前缀
            emb_name2: 启动子嵌入层的参数名前缀
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name1 in name or emb_name2 in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def visualize_with_tsne(self_attention_output, labels, file_path):
    """
    使用t-SNE可视化自注意力输出
    
    Args:
        self_attention_output: 模型的自注意力输出
        labels: 样本标签
        file_path: 保存图片的路径
    """
    # 将 self_attention_output 转换为 NumPy 数组
    self_attention_np = self_attention_output.cpu().detach().numpy()

    # 将 labels 转换为 NumPy 数组，并将其调整为一维数组
    labels_np = labels.view(-1).cpu().detach().numpy()

    # 创建 t-SNE 模型
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)

    # 使用 t-SNE 对 self_attention_output 进行降维
    embedded_points = tsne.fit_transform(self_attention_np)

    # 根据标签分割正样本和负样本的坐标
    positive_points = embedded_points[labels_np == 1]
    negative_points = embedded_points[labels_np == 0]

    # 绘制散点图，使用不同颜色表示正样本和负样本
    plt.scatter(positive_points[:, 0], positive_points[:, 1], c='red', label='Positive')
    plt.scatter(negative_points[:, 0], negative_points[:, 1], c='blue', label='Negative')

    # 添加图例和标题
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.savefig(file_path)
    # 显示图形
    plt.show()


def val_forwrd(model, dataloader, cell_name=""):
    """
    模型验证函数，计算验证集上的损失和指标
    
    Args:
        model: 要验证的模型
        dataloader: 验证数据加载器
        cell_name: 细胞系名称，用于进度条显示
        
    Returns:
        tuple: 包含三个元素的元组
            - test_epoch_loss: 验证集总损失
            - test_epoch_aupr: 验证集AUPR值
            - test_epoch_auc: 验证集AUC值
    """
    model.eval()
    test_epoch_loss = 0.0
    test_epoch_correct = 0
    test_epoch_preds = torch.tensor([]).to(device)
    test_epoch_target = torch.tensor([]).to(device)
    
    # 使用tqdm显示验证进度
    val_pbar = tqdm(dataloader, desc=f"Validation [{cell_name}]", 
                   leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for data in val_pbar:
            enhancer_ids, promoter_ids, enhancer_features, promoter_features, labels = data
            enhancer_ids = enhancer_ids.to(device, non_blocking=True)
            promoter_ids = promoter_ids.to(device, non_blocking=True)
            enhancer_features = enhancer_features.to(device, non_blocking=True)
            promoter_features = promoter_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
                
            outputs, _ = model(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
            labels = labels.unsqueeze(1).float()
            test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

            if labels.shape == torch.Size([1, 1]):
                labels = torch.reshape(labels, (1,))
                
            loss = model.criterion(outputs, labels)
            test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
            test_epoch_loss += loss.item()
            test_epoch_correct += get_num_correct(outputs, labels)
            
            # 更新进度条显示当前损失
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
    test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    return test_epoch_loss, test_epoch_aupr, test_epoch_auc


#得到的是序列数据
#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# 导入新的数据加载模块
from data_loader import load_train_data, load_all_test_data

# 加载训练数据 (使用配置文件中指定的细胞系)
enhancers_train, promoters_train, Labels_train = load_train_data(TRAIN_CELL_LINE)

# 加载所有细胞系的测试数据
test_data = load_all_test_data()
enhancers_test_GM12878, promoters_test_GM12878, Labels_test_GM12878 = test_data["GM12878"]
enhancers_test_IMR90, promoters_test_IMR90, Labels_test_IMR90 = test_data["IMR90"]
enhancers_test_HeLa, promoters_test_HeLa, Labels_test_HeLa = test_data["HeLa-S3"]
enhancers_test_HUVEC, promoters_test_HUVEC, Labels_test_HUVEC = test_data["HUVEC"]
enhancers_test_K562, promoters_test_K562, Labels_test_K562 = test_data["K562"]
enhancers_test_NHEK, promoters_test_NHEK, Labels_test_NHEK = test_data["NHEK"]

# 检查模型保存路径是否存在
if not os.path.exists(SAVE_MODEL_DIR):
    os.makedirs(SAVE_MODEL_DIR)

# 创建新的模型实例（不使用预训练权重）
epimodel = EPIModel()
epimodel = epimodel.to(device)
print("创建新的模型实例（不使用预训练权重）")

# 创建优化器
optimizer = torch.optim.Adam(epimodel.parameters(), lr=LEARNING_RATE)
epimodel.optimizer = optimizer

# 设置学习率调度器
scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)
pgd = PGD(epimodel)
K = 3

# 检查GPU是否可用
all_params_on_gpu = all(param.is_cuda for param in epimodel.parameters())
print("GPU是否可用：", torch.cuda.is_available())
print("模型是否在GPU上：", all_params_on_gpu)
print(f"使用训练数据：{TRAIN_CELL_LINE}细胞系")
print(f"使用测试数据：{', '.join(TEST_CELL_LINES)}细胞系")
print(f"批量大小：{BATCH_SIZE}")
print(f"数据加载器工作进程数：{NUM_WORKERS}")
    

# 创建优化的数据集 - 使用新的预处理模块
print("创建优化的数据集...")
train_dataset = MyDataset(enhancers_train, promoters_train, Labels_train)
# 提取原始序列数据
enhancers_train_raw = [train_dataset[i][0] for i in range(len(train_dataset))]
promoters_train_raw = [train_dataset[i][1] for i in range(len(train_dataset))]
labels_train_raw = [train_dataset[i][2] for i in range(len(train_dataset))]

# 创建优化的训练数据集
train_fold = OptimizedCombinedDataset(
    enhancers=enhancers_train_raw,
    promoters=promoters_train_raw,
    labels=labels_train_raw,
    cache_dir=os.path.join(CACHE_DIR, "train_cache"),
    use_cache=True
)

# 预热缓存
warmup_cache(train_fold, num_samples=100)

# 创建验证数据集
def create_optimized_validation_dataset(enhancers, promoters, labels, cell_name):
    """创建优化的验证数据集"""
    return OptimizedCombinedDataset(
        enhancers=enhancers,
        promoters=promoters,
        labels=labels,
        cache_dir=os.path.join(CACHE_DIR, f"{cell_name}_cache"),
        use_cache=True
    )

# GM12878验证集
val_dataset_GM12878 = MyDataset(enhancers_test_GM12878, promoters_test_GM12878, Labels_test_GM12878)
enhancers_test_GM12878_raw = [val_dataset_GM12878[i][0] for i in range(len(val_dataset_GM12878))]
promoters_test_GM12878_raw = [val_dataset_GM12878[i][1] for i in range(len(val_dataset_GM12878))]
labels_test_GM12878_raw = [val_dataset_GM12878[i][2] for i in range(len(val_dataset_GM12878))]
val_fold_GM12878 = create_optimized_validation_dataset(
    enhancers_test_GM12878_raw, promoters_test_GM12878_raw, labels_test_GM12878_raw, "GM12878"
)

# IMR90验证集
val_dataset_IMR90 = MyDataset(enhancers_test_IMR90, promoters_test_IMR90, Labels_test_IMR90)
enhancers_test_IMR90_raw = [val_dataset_IMR90[i][0] for i in range(len(val_dataset_IMR90))]
promoters_test_IMR90_raw = [val_dataset_IMR90[i][1] for i in range(len(val_dataset_IMR90))]
labels_test_IMR90_raw = [val_dataset_IMR90[i][2] for i in range(len(val_dataset_IMR90))]
val_fold_IMR90 = create_optimized_validation_dataset(
    enhancers_test_IMR90_raw, promoters_test_IMR90_raw, labels_test_IMR90_raw, "IMR90"
)

# HeLa验证集
val_dataset_HeLa = MyDataset(enhancers_test_HeLa, promoters_test_HeLa, Labels_test_HeLa)
enhancers_test_HeLa_raw = [val_dataset_HeLa[i][0] for i in range(len(val_dataset_HeLa))]
promoters_test_HeLa_raw = [val_dataset_HeLa[i][1] for i in range(len(val_dataset_HeLa))]
labels_test_HeLa_raw = [val_dataset_HeLa[i][2] for i in range(len(val_dataset_HeLa))]
val_fold_HeLa = create_optimized_validation_dataset(
    enhancers_test_HeLa_raw, promoters_test_HeLa_raw, labels_test_HeLa_raw, "HeLa"
)

# HUVEC验证集
val_dataset_HUVEC = MyDataset(enhancers_test_HUVEC, promoters_test_HUVEC, Labels_test_HUVEC)
enhancers_test_HUVEC_raw = [val_dataset_HUVEC[i][0] for i in range(len(val_dataset_HUVEC))]
promoters_test_HUVEC_raw = [val_dataset_HUVEC[i][1] for i in range(len(val_dataset_HUVEC))]
labels_test_HUVEC_raw = [val_dataset_HUVEC[i][2] for i in range(len(val_dataset_HUVEC))]
val_fold_HUVEC = create_optimized_validation_dataset(
    enhancers_test_HUVEC_raw, promoters_test_HUVEC_raw, labels_test_HUVEC_raw, "HUVEC"
)

# K562验证集
val_dataset_K562 = MyDataset(enhancers_test_K562, promoters_test_K562, Labels_test_K562)
enhancers_test_K562_raw = [val_dataset_K562[i][0] for i in range(len(val_dataset_K562))]
promoters_test_K562_raw = [val_dataset_K562[i][1] for i in range(len(val_dataset_K562))]
labels_test_K562_raw = [val_dataset_K562[i][2] for i in range(len(val_dataset_K562))]
val_fold_K562 = create_optimized_validation_dataset(
    enhancers_test_K562_raw, promoters_test_K562_raw, labels_test_K562_raw, "K562"
)

# NHEK验证集
val_dataset_NHEK = MyDataset(enhancers_test_NHEK, promoters_test_NHEK, Labels_test_NHEK)
enhancers_test_NHEK_raw = [val_dataset_NHEK[i][0] for i in range(len(val_dataset_NHEK))]
promoters_test_NHEK_raw = [val_dataset_NHEK[i][1] for i in range(len(val_dataset_NHEK))]
labels_test_NHEK_raw = [val_dataset_NHEK[i][2] for i in range(len(val_dataset_NHEK))]
val_fold_NHEK = create_optimized_validation_dataset(
    enhancers_test_NHEK_raw, promoters_test_NHEK_raw, labels_test_NHEK_raw, "NHEK"
)


# 创建优化的数据加载器
train_loader = DataLoader(
    dataset=train_fold, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=simple_collate_fn
)

# 创建各细胞系的验证数据加载器
val_loader_GM12878 = DataLoader(
    dataset=val_fold_GM12878, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=simple_collate_fn
)

val_loader_IMR90 = DataLoader(
    dataset=val_fold_IMR90, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=simple_collate_fn
)

val_loader_HeLa = DataLoader(
    dataset=val_fold_HeLa, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=simple_collate_fn
)

val_loader_HUVEC = DataLoader(
    dataset=val_fold_HUVEC, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=simple_collate_fn
)

val_loader_K562 = DataLoader(
    dataset=val_fold_K562, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=simple_collate_fn
)

val_loader_NHEK = DataLoader(
    dataset=val_fold_NHEK, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=simple_collate_fn
)



# 开始训练循环
for i in range(EPOCH):
    epimodel.train()
    train_epoch_loss = 0.0
    train_epoch_correct = 0
    train_epoch_preds = torch.tensor([]).to(device)
    train_epoch_target = torch.tensor([]).to(device)
    
    # 使用tqdm显示训练进度
    train_pbar = tqdm(train_loader, desc=f"Epoch {i+1}/{EPOCH} [Training]", 
                     leave=True, dynamic_ncols=True)
    
    # 遍历训练数据
    for data in train_pbar:
        enhancer_ids, promoter_ids, enhancer_features, promoter_features, labels = data
        enhancer_ids = enhancer_ids.to(device, non_blocking=True)
        promoter_ids = promoter_ids.to(device, non_blocking=True)
        enhancer_features = enhancer_features.to(device, non_blocking=True)
        promoter_features = promoter_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs, emd = epimodel(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
        
        labels = labels.unsqueeze(1).float()
        
        train_epoch_preds = torch.cat((train_epoch_preds, outputs.view(-1)))
        train_epoch_target = torch.cat((train_epoch_target, labels.view(-1)))
        
        if labels.shape == torch.Size([1, 1]):
            labels = torch.reshape(labels, (1,))

        loss = epimodel.criterion(outputs, labels)
        train_epoch_loss += loss.item()
        train_epoch_correct += get_num_correct(outputs, labels)

        loss.backward()
        
        # 对抗训练 - PGD
        if ENABLE_ADVERSARIAL_TRAINING:
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动
                if t != K-1:
                    epimodel.zero_grad()
                else:
                    pgd.restore_grad()
                
                outputs_adv, _ = epimodel(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
                loss_adv = epimodel.criterion(outputs_adv, labels)
                loss_adv.backward()
            
            pgd.restore() # 恢复embedding参数

        epimodel.optimizer.step()
        epimodel.optimizer.zero_grad()
        
        # 更新进度条显示当前损失
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    scheduler.step()
    logger.info("learning_rate: {}".format(scheduler.get_last_lr()))
    train_epoch_aupr = average_precision_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
    train_epoch_auc = roc_auc_score(train_epoch_target.cpu().detach().numpy(), train_epoch_preds.cpu().detach().numpy())
      
    # 验证阶段 - 只在指定epoch进行验证以节省时间
    if i % VALIDATION_INTERVAL == 0 or i == EPOCH - 1:
        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_loader_GM12878, "GM12878")
        torch.save(epimodel, os.path.join(SAVE_MODEL_DIR, f'DNABERT1_pgd_genes_{TRAIN_CELL_LINE}_train_model_lr{LEARNING_RATE}_epoch{i}.pt'))
        logger.info("fine_tuning GM12878 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))
        
        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_loader_IMR90, "IMR90")
        logger.info("fine_tuning IMR90 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_loader_HeLa, "HeLa-S3")
        logger.info("fine_tuning HeLa epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_loader_HUVEC, "HUVEC")
        logger.info("fine_tuning HUVEC epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_loader_K562, "K562")
        logger.info("fine_tuning K562 epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))

        test_epoch_loss, test_epoch_aupr, test_epoch_auc = val_forwrd(epimodel, val_loader_NHEK, "NHEK")
        logger.info("fine_tuning NHEK epoch: {}, test_loss：{}, test_aupr：{}, test_auc:{}".format(i + 1, test_epoch_loss, test_epoch_aupr, test_epoch_auc))