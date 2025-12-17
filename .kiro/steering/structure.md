# 项目结构

## 根目录文件
- **PRISM.py**: 主训练脚本，包含能量耗散框架
- **predict.py**: 模型评估和推理脚本
- **config.py**: 集中配置管理(所有参数都在这里)
- **data_loader.py**: 数据加载工具和自定义采样器
- **run.sh**: 快速执行脚本(训练+关机)

## 核心目录

### `/models/`
- **PRISMModel.py**: 主模型架构，包含能量耗散系统
- **layers/**: 自定义神经网络组件
  - **attn.py**: 注意力机制(CBAT, RoPE)
  - **FourierKAN.py**: 基于傅里叶的Kolmogorov-Arnold网络
- **pleat/**: DNA序列处理模块
  - **embedding.py**: K-mer分词和DNA嵌入
  - **RoPE.py**: 旋转位置编码实现
- **AuxiliaryModel.py**: 辅助模型组件

### `/dataset/` 和 `/dataset2/`
- 标准数据集结构: `train/`, `test/`, `val/`
- 细胞系子目录: `GM12878/`, `HUVEC/`等
- 每个细胞系的文件:
  - **pairs_hg19.csv**: 增强子-启动子配对及标签
  - **e_seq.csv**: 增强子序列
  - **p_seq.csv**: 启动子序列

### `/domain-kl/`
- PRISM专用训练数据(域知识学习)
- 与dataset相同结构但针对PRISM训练优化

### `/_cache/`
- PyTorch张量缓存系统
- 细胞系特定的缓存目录
- **cache_index.pkl**: 缓存元数据
- **data_*.pt**: 缓存的张量文件

### `/save_model/` 和 `/CBAT/`
- 模型检查点和权重
- **prism_epoch_*.pth**: 仅模型检查点
- **prism_full_epoch_*.pt**: 完整训练状态

### `/log/`
- 训练和评估日志
- 带时间戳的日志文件，包含全面指标

### `/vocab/`
- **tokenizer.pkl**: K-mer分词器状态
- **cell_type.xml**: 细胞类型配置
- **hash.key**: 词汇表哈希键

### `/compete/`
- 竞赛/基准测试结果
- **decouple/**: 解耦分析结果
- **footprint/**: 足迹分析输出

### `/docx/` 和 `/workflow/`
- 文档和工作流指南
- 历史记录和实验笔记

## 数据流架构

1. **输入**: 原始DNA序列(增强子/启动子配对)
2. **分词**: 通过`models/pleat/embedding.py`进行K-mer编码
3. **处理**: CNN特征提取 + transformer注意力
4. **能量模型**: U_I(内势) - R_E(环境阻抗)
5. **输出**: EP互作概率

## 配置层次结构

1. **config.py**: 主配置(仅在此修改)
2. **无CLI参数**: 所有参数基于文件
3. **设备检测**: 自动CUDA/CPU选择
4. **路径管理**: 所有路径相对于PROJECT_ROOT

## 命名约定

- **文件**: Python文件使用snake_case
- **类**: PascalCase (如PRISMBackbone, CBATTransformerEncoderLayer)
- **函数**: snake_case，描述性命名
- **常量**: config.py中使用UPPER_CASE
- **目录**: 数据目录使用小写加连字符

## 开发工作流

1. 在`config.py`中修改参数
2. 运行训练: `python PRISM.py`
3. 评估: `python predict.py`
4. 检查`/log/`目录中的训练日志和`compete/tss`中的未知细胞系(OOD Cell-Type)预测结果
5. 模型权重自动保存到配置目录

## 关键设计原则

- **解耦架构**: 主干网络分离共性表征(z_I)和细胞系特异模块(z_F)
- **域对抗**: 使用GRL确保z_I不包含细胞系信息
- **能量物理**: U_I - R_E框架模拟生物物理过程
- **正交约束**: z_I ⊥ z_F 确保特征独立性
- **目标导向**: 所有设计围绕AUPR ≥ 0.75的验收标准