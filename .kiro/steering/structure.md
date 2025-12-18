# 项目背景

PRISM是一个用于预测基因组序列中增强子-启动子(EP)互作的深度学习系统。该项目实现了一个复杂的神经网络架构，结合了：

## 核心任务
- **任务目标**: 仅使用DNA序列信息挖掘增强子-启动子互作关系
- **核心挑战**: 不同细胞系间特异性差异巨大，单一模型在未见细胞系上容易失效，目前AUPR=70，处于瓶颈

## 验收标准
- **目标指标**: domain-kl/test及各细胞系AUPR ≥ 0.75
- **评估范围**: 跨细胞系泛化性能，特别是未见细胞系的鲁棒性

---

# 项目结构

## 根目录文件
- **PRISM.py**: 主训练脚本，包含能量耗散框架
- **predict.py**: 模型评估和推理脚本
- **config.py**: 集中配置管理(所有参数都在这里)
- **data_loader.py**: 数据加载工具和自定义采样器

## 核心目录

### `/models/`
- **PRISMModel.py**: 主模型架构，包含能量耗散系统
- **layers/**: 自定义神经网络组件，运行 `ls models/layers`以获取最新组件情况，以下内容为早期总结
  - **attn.py**: 注意力机制(CBAT, RoPE)
  - **FourierKAN.py**: 基于傅里叶的Kolmogorov-Arnold网络
  - **footprint.py**: 早期测试遗留的小波变换备用组件
  - **graph_context.py**: 早期测试遗留的基于图的上下文备用组件
  - **ISAB.py**: 早期测试遗留的SET Transformers备用组件
- **pleat/**: 自定义工具包，运行 `ls models/pleat`以获取最新组件情况，以下内容为早期总结
  - **embedding.py**: K-mer分词和DNA嵌入
  - **RoPE.py**: 旋转位置编码实现

### `/domain-kl/`
- 标准数据集结构: `train/`, `test/`, `val/`
- 每个细胞系的文件:
  - **pairs_hg19.csv**: 增强子-启动子配对及细胞系标签、EP互作0/1标签
  - **e_seq.csv**: 增强子序列
  - **p_seq.csv**: 启动子序列

### `/_cache/`
- 缓存文件

### `/save_model/` 
- **prism_epoch_*.pth**: 仅模型检查点
- **prism_full_epoch_*.pt**: 完整训练状态

### `/log/`
- 训练和评估日志
- 带时间戳的日志文件，包含全面指标

### `/vocab/`
- 固定词表

### `/compete/`
- 位置细胞系预测测试结果

### `/docx/` 
- 文档和工作流指南
- 历史记录和实验笔记

## 配置层次结构
1. **config.py**: 主配置(仅在此修改)
2. **无shell参数**: 所有参数基于脚本文件本身而不基于额外的传参
3. **设备检测**: 自动CUDA选择
4. **路径管理**: 所有路径相对于PROJECT_ROOT避免绝对路径导致的路径不匹配问题

## 命名约定
- **文件**: Python文件使用snake_case
- **类**: PascalCase (如PRISMBackbone, CBATTransformerEncoderLayer)
- **函数**: snake_case，描述性命名
- **常量**: config.py中使用UPPER_CASE
- **目录**: 数据目录使用小写加连字符

## 关键文件概览
1. 在`config.py`中修改参数
2. 运行训练: `python PRISM.py`
3. 评估: `python predict.py`
4. 检查`/log/`目录中的训练日志和`compete/tss`中的未知细胞系(OOD Cell-Type)预测结果
5. 模型权重自动保存到配置目录