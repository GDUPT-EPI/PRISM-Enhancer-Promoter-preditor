# PRISM: Predictive RNA-seq-independent Interaction System for genomic Modifiers

PRISM 是一个用于预测基因组序列中增强子-启动子 (Enhancer-Promoter, EP) 互作的深度学习系统。该项目旨在仅利用 DNA 序列信息，通过复杂的神经网络架构挖掘 EP 互作关系，并特别关注跨细胞系的泛化性能和鲁棒性。

## **核心任务**
- **目标**: 仅使用 DNA 序列信息预测增强子与启动子之间的相互作用。
- **挑战**: 不同细胞系间的特异性差异巨大。PRISM 通过引入能量耗散框架和域对抗训练，致力于解决单一模型在未见细胞系上失效的问题。

---

## **核心架构**

PRISM 采用了多项先进技术构建其模型架构 `PRISMBackbone`：

![全流程](png/全流程.png)

1.  **DNA 序列编码**:
    - **K-mer Tokenization**: 使用 6-mer 对 DNA 序列进行分词。
    - **CNN 编码器**: 提取序列的局部特征。
    - **RoPE (Rotary Positional Embedding)**: 引入旋转位置编码，增强模型对序列位置信息的感知。

2.  **注意力机制**:
    - **跨序列注意力**: 结合增强子和启动子的特征，进行深度融合。
    ![注意力机制](png/注意力机制.png)
    - **CBAT (Cross-Boundary Attention Transformer)**: 自定义的交叉边界注意力机制，专门用于建模增强子与启动子之间的长距离交互。
    ![CBAT模块](png/CBAT模块.png)


3.  **能量耗散框架 (Energy Dissipation)**:
    - **内势 $U_I$ (Internal Potential)**: 代表序列本身的固有属性，由 `FourierEnergyKAN` 生成。
    - **环境阻抗 $R_E$ (Environmental Resistance)**: 代表细胞系特有的环境影响。
    - 模型通过建模能量在环境中的耗散过程，提高预测的物理可解释性和泛化能力。

    ![耗散系统](png/耗散系统.png)

4.  **域对抗训练 (Domain Adaptation)**:
    - **GRL (Gradient Reversal Layer)**: 引入梯度反转层，配合域判别器，强迫模型学习与细胞系无关的通用特征（$z_I$），从而提升跨细胞系的迁移能力。

5.  **KAN (Kolmogorov-Arnold Networks)**:
    - 使用 `FourierKAN` 作为分类头和能量生成器，相比传统 MLP 具有更强的非线性表达能力。

---

## **项目结构**

```text
PRISM/
├── PRISM.py              # 主训练脚本，包含训练循环和损失函数计算
├── predict.py            # 模型评估和推理脚本，支持 AUPR/AUC 计算及可视化
├── config.py             # 集中参数配置管理
├── data_loader.py        # 数据加载工具，包含自定义采样器
├── models/               # 模型定义目录
│   ├── PRISMModel.py     # 主模型架构 PRISMBackbone
│   ├── layers/           # 自定义网络层 (CBAT, RoPE, FourierKAN 等)
│   └── pleat/            # 工具包 (Embedding, Loss 函数等)
├── domain-kl/            # 数据集目录 (train, test, val)
├── save_model/           # 模型检查点保存路径
├── log/                  # 训练和评估日志
└── vocab/                # 词表和细胞类型配置文件
```

---

## **快速开始**

### **环境要求**
- Python 3.10+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, tqdm

### **1. 配置参数**
所有超参数均在 `config.py` 中集中管理。您可以根据需要调整 `PRISM_BATCH_SIZE`、`LEARNING_RATE`、`EPOCH` 等参数。

### **2. 模型训练**
运行以下命令开始训练：
```bash
python PRISM.py
```
训练过程会自动加载 `domain-kl/train` 数据，并将模型权重保存至 `save_model/`。

### **3. 模型评估**
运行以下命令在测试集上进行评估：
```bash
python predict.py
```
评估脚本将输出总体及各细胞系的 AUPR、AUC、F1 等指标，并在 `compete/` 目录下生成 PR 曲线和 ROC 曲线图。

---

## **主要指标**
PRISM 专注于跨细胞系的鲁棒性。目前系统正在优化中，目标是实现所有测试细胞系的 AUPR 稳定在 0.75 以上。

## **联系与反馈**
如有任何问题或改进建议，请参考项目文档或联系开发团队。
