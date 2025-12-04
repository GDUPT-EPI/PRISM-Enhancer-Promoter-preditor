# 基于特征矩阵的细胞系条件化调节机制

## 一、核心思想：软性条件调节而非硬性解耦

放弃正交分解，采用**条件化特征调节矩阵**，让模型自动学习细胞系特异性的调节方式，同时保留跨细胞系的共性信息。

## 二、具体机制设计

### 2.1 特征矩阵的构建与解释

设主干网络提取的E/P对特征为：

$$
F \in \mathbb{R}^{d \times m}
$$

其中：
- $d$: 特征维度（如512）
- $m$: 特征维度数（对应不同语义信息）

**特征矩阵的语义分解**：

$$
F = [F_1, F_2, \dots, F_m] \quad \text{其中} \quad F_i \in \mathbb{R}^d
$$

每个 $F_i$ 编码不同的序列语义信息：
1. $F_1$: 保守motif响应（高跨细胞系一致性）
2. $F_2$: 细胞系偏好模式（高细胞系间方差）
3. $F_3$: E/P协同激活模式
4. $F_4$: 染色质可及性预测
5. ...（其他自学习语义）

### 2.2 细胞系专家网络的作用

专家网络基于细胞系上下文 $C \in \mathbb{R}^{B \times d}$（B=64个E/P对的聚合信息）生成**条件调节矩阵**：

$$
\Theta_C = \text{ExpertNet}(C) \in \mathbb{R}^{k \times m}
$$

其中 $k$ 个调节维度对应：
1. **特异性增强系数** $\theta_{\text{specificity}}$：放大细胞系特异性信号
2. **共性保留系数** $\theta_{\text{conservation}}$：维持保守信号强度
3. **交互调节系数** $\theta_{\text{interaction}}$：调整E/P协同模式
4. **噪声抑制系数** $\theta_{\text{denoise}}$：降低细胞系无关噪声
5. **动态权重系数** $\theta_{\text{weight}}$：特征维度重加权

### 2.3 条件化特征变换

采用**门控特征融合机制**：

$$
\tilde{F} = \underbrace{\sigma(\theta_{\text{weight}} \cdot F)}_{\text{特征重加权}} \odot 
\Big[ \underbrace{(1 + \theta_{\text{specificity}}) \cdot F_{\text{specific}}}_{\text{特异性增强}} + 
\underbrace{(1 + \theta_{\text{conservation}}) \cdot F_{\text{conserved}}}_{\text{共性保持}} \Big]
$$

其中：
- $F_{\text{specific}} = \text{HighVarianceComponents}(F)$
- $F_{\text{conserved}} = \text{LowVarianceComponents}(F)$
- $\sigma$: sigmoid函数
- $\odot$: 逐元素乘法

### 2.4 简化的损失函数设计

**总损失函数**（仅三项）：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{consistency}} + \lambda_2 \mathcal{L}_{\text{adapt}}
$$

#### (1) 分类损失（标准BCE）：

$$
\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^N [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]
$$

#### (2) 跨细胞系一致性损失：

鼓励相同E/P对在不同细胞系中保持相似预测：

$$
\mathcal{L}_{\text{consistency}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \|\hat{y}_{i,c_1} - \hat{y}_{i,c_2}\|_2^2
$$

其中 $\mathcal{P}$ 是跨细胞系共享的E/P对集合。

#### (3) 自适应正则化损失：

防止专家网络过度调节：

$$
\mathcal{L}_{\text{adapt}} = \|\Theta_C - I\|_F^2
$$

其中 $I$ 是单位矩阵，鼓励调节接近恒等变换。

## 三、前沿技术栈选择

### 3.1 骨干编码器

- **DNABERT-2**：预训练Transformer，捕获k-mer模式
- **Performer** 或 **Linformer**：线性注意力，降低计算复杂度
- **ResNet+Attention**：CNN提取局部模式，Attention捕获长程依赖

### 3.2 专家网络架构

- **FiLM层**（Feature-wise Linear Modulation）：

  $$
  \text{FiLM}(F, \gamma, \beta) = \gamma \odot F + \beta
  $$

  其中 $\gamma, \beta \in \mathbb{R}^m$ 由专家网络从 $C$ 生成

- **自适应门控机制**：

  $$
  g = \sigma(W_g C + b_g) \in [0,1]^m
  $$

  $$
  \tilde{F} = g \odot F_{\text{transformed}} + (1-g) \odot F
  $$

### 3.3 细胞系上下文编码

- **双向Transformer编码器**：

  $$
  C = \text{Transformer}(\{F_i\}_{i=1}^B)
  $$

- **层次化池化**：

  $$
  C = \text{MeanPool}(F) \oplus \text{MaxPool}(F) \oplus \text{StdPool}(F)
  $$

### 3.4 测试时适应机制

- **原型网络快速适应**：

  $$
  p_c = \frac{1}{B} \sum_{i=1}^B f(x_i)
  $$

  $$
  \gamma_c, \beta_c = \text{ExpertNet}(p_c)
  $$

- **一步梯度调整**（仅调节专家网络）：

  $$
  \theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{consistency}}
  $$

## 四、完整数学表述

### 4.1 前向传播流程

1. **特征提取**：

   $$
   F_i = \text{Encoder}(x_i^E, x_i^P) \in \mathbb{R}^{d \times m}
   $$

2. **细胞系上下文聚合**（批量B=64）：

   $$
   C = \text{MultiHeadAttention}\left(\frac{1}{B}\sum_{i=1}^B F_i, \{F_i\}_{i=1}^B, \{F_i\}_{i=1}^B\right)
   $$

3. **专家参数生成**：

   $$
   \Gamma = \text{MLP}_\gamma(C) \in \mathbb{R}^{m}, \quad B = \text{MLP}_\beta(C) \in \mathbb{R}^{m}
   $$

4. **条件化特征变换**：

   $$
   \tilde{F}_i = \Gamma \odot F_i + B
   $$

5. **动态特征选择**（通过自注意力）：

   $$
   A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right), \quad Q = \tilde{F}_i W_Q, \quad K = \tilde{F}_i W_K
   $$

   $$
   F_i^{\text{final}} = A \tilde{F}_i
   $$

6. **预测输出**：

   $$
   \hat{y}_i = \sigma\left(W_p \cdot \text{Flatten}(F_i^{\text{final}}) + b_p\right)
   $$

### 4.2 训练策略

**两阶段训练**：

1. **阶段一：预训练主干**

   $$
   \min_{\theta_{\text{backbone}}} \mathcal{L}_{\text{cls}}
   $$

2. **阶段二：联合训练专家网络**

   $$
   \min_{\theta_{\text{expert}}} \mathcal{L}_{\text{cls}} + 0.1 \cdot \mathcal{L}_{\text{consistency}} + 0.01 \cdot \mathcal{L}_{\text{adapt}}
   $$

## 五、关键创新点

1. **条件化特征调节**：不强行解耦，而是学习调节矩阵
2. **软性一致性约束**：跨细胞系相似E/P对应有相似预测
3. **轻量级专家网络**：仅生成少量调节参数，避免过拟合
4. **自适应性正则**：鼓励调节接近恒等变换，保证稳定性

## 六、数学验证

1. **调节强度度量**：

   $$
   \text{Adjustment Magnitude} = \frac{1}{m} \|\Gamma - 1\|_1
   $$

2. **跨细胞系一致性**：

   $$
   \text{Consistency Score} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \exp\left(-\|\hat{y}_i - \hat{y}_j\|_1\right)
   $$

3. **专家网络贡献度**：

   $$
   \text{Expert Contribution} = \frac{\text{AUC}_{\text{with expert}} - \text{AUC}_{\text{without expert}}}{\text{AUC}_{\text{without expert}}}
   $$

这个方案避免了复杂的正交分解，通过条件化特征调节实现细胞系特异性适应，同时保持模型简洁和训练稳定性。