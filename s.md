下面这份是专门给“预训练阶段”的高维投影任务设计说明，假定：

- footprint 基本已经由你的小波（LCWnet + CWT/CWC）模块算出来；
- 预训练阶段会同时跑两个任务：
  - BERT 掩蔽预测任务（token 级）；
  - 高维投影任务（cell 级 / footprint 级），用来“分解共同特征 + 特别特征”。

重点是第二个任务的数学形式：  
如何在“一个 batch = 一个细胞系”的设定下，让模型学到：

- 一个“**共同子空间**”：承载 BERT 所需的大部分信息，尽量 cell‑不变；
- 一个“**特异子空间**”：容量小、稀疏，只在区分细胞系时被使用；
- 共同 vs 特异 的“占比”不是手调，而是由 BERT 任务的梯度自动决定。

---

## 1. 输入与小波 footprint 表征

你的小波部分已经是这样的流程（略化为函数符号）：

- 对某条“信号” \(f\)（可以是 enhancer 的某个通道，或对 E/P 组合后的序列表示）：

  \[
  \text{CWC}(f) \in \mathbb{C}^{S\times L}
  \]

- 这一步的具体计算是：

  \[
  \begin{aligned}
  F_f &= \text{DFT}(f) \\
  \Psi_{a_s} &= \big[ \mathcal{F}[\psi_\theta]^*(a_s q_l) \big]_{l=0}^{L-1} \\
  Y_{a_s} &= F_f \odot \Psi_{a_s} \\
  T_s &= \text{IDFT}(Y_{a_s}) \\
  \text{CWC} &= \{T_s\}_{s=0}^{S-1} \in \mathbb{C}^{S\times L}
  \end{aligned}
  \]

在预训练的高维投影任务中，我们不直接用整个 CWC 矩阵，而是把它变成一个实值向量特征。

### 1.1 从 CWC 到样本级 footprint 向量

对单个 EP 样本 \(i\)（属于细胞 \(c\)）：

1. 对 enhancer（或增强子+启动子组合）的时间序列 \(f_i\) 做 CWC：

   \[
   \text{CWC}_i \in \mathbb{C}^{S\times L}
   \]

2. 将其转为实值特征（幅度 + 多尺度汇聚），例如：

   \[
   M_i = |\text{CWC}_i| \in \mathbb{R}^{S\times L}
   \]

3. 用一个小的 2D 卷积 + 池化网络（或直接多尺度 pooling）得到样本级 footprint 向量：

   \[
   v_i = \Phi_{\text{CWC}}(M_i) \in \mathbb{R}^{D_v}
   \]

   - 这里 \(\Phi_{\text{CWC}}\) 是可学习的（Conv / MLP），但规模要远小于主干；
   - 你可以把它看作“把 CWC 这个 \(S\times L\) 的频–时图压缩成一个 footprint embedding”。

对同一个细胞 \(c\) 的 K 个样本，得到：

\[
\{v_i\}_{i=1}^K,\quad v_i\in\mathbb{R}^{D_v}
\]

---

## 2. 公共子空间 + 特异子空间：线性投影与分解

我们希望在同一个 \(v_i\) 上，拆出：

- 一部分是“共同特征” \(z^{\text{com}}_{c,i}\)，在不同 cell 之间尽量一致；
- 一部分是“特异特征” \(z^{\text{spec}}_{c,i}\)，用于表示 cell‑specific footprint。

### 2.1 线性投影定义

设公共子空间维度 \(d_{\text{com}}\)，特异子空间维度 \(d_{\text{spec}}\)（一般 \(d_{\text{com}} \gg d_{\text{spec}}\)）：

\[
\begin{aligned}
z^{\text{com}}_{c,i} &= W_{\text{com}} v_i \in \mathbb{R}^{d_{\text{com}}} \\
z^{\text{spec}}_{c,i} &= W_{\text{spec}} v_i \in \mathbb{R}^{d_{\text{spec}}}
\end{aligned}
\]

其中：

- \(W_{\text{com}} \in \mathbb{R}^{d_{\text{com}}\times D_v}\)  
- \(W_{\text{spec}} \in \mathbb{R}^{d_{\text{spec}}\times D_v}\)

为了让两个子空间不互相污染，我们对 \(W_{\text{com}}, W_{\text{spec}}\) 加一个近似正交约束：

\[
\mathcal{L}_{\text{ortho}} = 
\big\| W_{\text{com}} W_{\text{spec}}^\top \big\|_F^2
\]

再配合行向量归一化（或额外约束 \(W_{\text{com}} W_{\text{com}}^\top \approx I\)）。

### 2.2 cell 级公共/特异 footprint

为了做 cell 级约束，我们对一个 cell \(c\) 的样本取均值：

\[
\begin{aligned}
g^{\text{com}}_c &= \frac{1}{K}\sum_{i=1}^K z^{\text{com}}_{c,i} \in \mathbb{R}^{d_{\text{com}}} \\
g^{\text{spec}}_c &= \frac{1}{K}\sum_{i=1}^K z^{\text{spec}}_{c,i} \in \mathbb{R}^{d_{\text{spec}}}
\end{aligned}
\]

- \(g^{\text{com}}_c\) = 该细胞在公共子空间中的 footprint 平均；  
- \(g^{\text{spec}}_c\) = 该细胞在特异子空间中的特征。

---

## 3. BERT 任务如何“决定”共同 vs 特异的占比（可学习机制）

你已经有一个 BERT / masked‑LM 任务，这里我们只假设：

- BERT encoder 输出某个样本 \(i\) 的 token‑level 表示 \(\{h_{i,t}\}\)；
- 你在 token 维/时间维上还会做 mask 预测。

我们要做的是：让 BERT 任务主要靠“共同特征”来工作，但**在必要的时候，允许它通过一个可学习门控少量使用“特异特征”**，并对这个门控加正则。这就是你说的“占比由 BERT 反馈决定”。

### 3.1 BERT 侧的融合：z_com + α·z_spec

为了方便对接，你可以在 BERT 的某一层（比如 encoder 输出后、送入预测头前）引入一个 **cell‑级上下文向量**：

- 对属于 cell \(c\) 的所有样本，BERT 可以访问 \(g^{\text{com}}_c, g^{\text{spec}}_c\)；
- 对每个样本的每个 token，我们构造一个上下文 embedding：

  \[
  u_{c} = g^{\text{com}}_c + \alpha\, g^{\text{spec}}_c \in \mathbb{R}^{d_{\text{com}}}
  \]

  这里为了对齐维度，你可以先把 \(g^{\text{spec}}_c\) 通过一个投影映射到 \(d_{\text{com}}\) 维：

  \[
  \tilde{g}^{\text{spec}}_c = P_{\text{spec}\to\text{com}}\, g^{\text{spec}}_c,\quad P_{\text{spec}\to\text{com}}\in\mathbb{R}^{d_{\text{com}}\times d_{\text{spec}}}
  \]

  再写成：

  \[
  u_c = g^{\text{com}}_c + \alpha\, \tilde{g}^{\text{spec}}_c
  \]

- BERT 的 masked‑LM 头可以把这个 \(u_c\) 整合进 token 表示，例如：
  - 拼接到每个 token 的 hidden 上再线性变换；
  - 或只作为 bias/context 输入最后一层预测头。

**关键点：\(\alpha\) 是可学习参数**（可以是标量或低维向量），并且我们在损失里对它加惩罚：

\[
\mathcal{L}_{\alpha} = \lambda_\alpha \|\alpha\|_2^2
\]

于是：

- 如果 BERT 完全可以用公共特征 \(g^{\text{com}}_c\) 完成任务，梯度会让 \(\alpha\) 趋近于 0；
- 若某些 BERT 任务确实需要 cell‑specific 信息（例如 motif 分布确实影响预测），BERT 会推动 \(\alpha\) 略大，让一部分特异信息通过；
- 这就实现了“**占比由 BERT 任务反馈自动决定**”。

### 3.2 BERT 任务损失

记 BERT 的 masked‑LM 损失为：

\[
\mathcal{L}_{\text{BERT}} = 
\mathbb{E}_{(c,i)}\big[
\text{CrossEntropy}(\hat{x}_{c,i}^{\text{masked}}(u_c),\ x_{c,i}^{\text{masked}})
\big]
\]

这里 \(\hat{x}_{c,i}^{\text{masked}}(u_c)\) 表示在加入上下文 \(u_c\) 后的预测分布。

---

## 4. “共同特征尽量多、特异特征尽量特别”的约束设计

这是高维投影任务的核心：  
我们要用一些几何/统计约束，让空间自然分裂成：

- **公共空间**：容量大、承载 BERT 所需的信息；  
- **特异空间**：容量小、足以区分 cell，但不会被噪声充斥。

### 4.1 公共特征：跨细胞一致 + 内部高信息量

这两点可以通过两个损失来实现：

#### 4.1.1 跨细胞一致（invariance）

我们希望不同细胞的公共 footprint \(g^{\text{com}}_c\) 尽量接近一个“全局均值”，  
但是不能塌到完全常数，因为 BERT 任务会约束它必须保留充分信息。

在“一个 batch = 一个 cell” 的设定下，我们可以维护一个 **全局 EMA 均值** \(\mu_{\text{com}}\)：

- 初始化：\(\mu_{\text{com}} = 0\)；
- 每处理完一个 cell \(c\) 的 batch，更新：
  \[
  \mu_{\text{com}} \leftarrow (1-\beta)\mu_{\text{com}} + \beta g^{\text{com}}_c
  \]
- 加一个 invariance 损失：
  \[
  \mathcal{L}_{\text{invar}} = 
  \|g^{\text{com}}_c - \mu_{\text{com}}\|_2^2
  \]

这样，**公共部分的 cell‑间差异被压缩**，但仍然要足以完成 BERT 任务，所以不会塌缩。

#### 4.1.2 公共特征内部要“多”：维度尽量被用满

为了防止 \(g^{\text{com}}_c\) 在少数维度上振荡，其余维完全废掉，我们可以在每个 batch 上计算样本级的公共向量（不聚合到 cell），例如：

\[
\tilde{z}^{\text{com}}_{c,i} = z^{\text{com}}_{c,i}
\]

在一个训练窗口内，估计公共空间的协方差：

\[
C_{\text{com}} = 
\mathbb{E}_i\big[
(\tilde{z}^{\text{com}}_{c,i} - \bar{z}^{\text{com}})
(\tilde{z}^{\text{com}}_{c,i} - \bar{z}^{\text{com}})^\top
\big]
\]

我们希望：

- 每个维度的方差不太小（>某个阈值）；
- 不同维度间协方差尽量小（去相关）。

可以用类似 VICReg / Barlow Twins 的形式：

\[
\begin{aligned}
\mathcal{L}_{\text{var,com}} &= 
\sum_{d=1}^{d_{\text{com}}}
\max(0, \gamma - \sqrt{\text{Var}(\tilde{z}^{\text{com}}_{:,d})})^2 \\
\mathcal{L}_{\text{cov,com}} &= 
\sum_{d\neq d'} \big(C_{\text{com}}[d,d']\big)^2
\end{aligned}
\]

其中 \(\gamma>0\) 是期望的最小标准差。  
这两项鼓励 **公共空间的每个维度都有非平凡信息**，也互不冗余，符合你说的“共同特征尽可能多”。

### 4.2 特异特征：区分细胞 + 稀疏/小容量

#### 4.2.1 区分细胞（discriminative）

最直接的方式是用 cell ID 做监督（只在预训练集 \(\mathcal{C}_A\) 上）：

\[
\hat{y}_c = \text{MLP}_{\text{cell}}(g^{\text{spec}}_c),\quad
\mathcal{L}_{\text{cell}} = \text{CrossEntropy}(\hat{y}_c,\ \text{cell\_id}(c))
\]

或者用对比损失（InfoNCE），把同一 cell 的不同视图作为正对，其他 cell 的 \(g^{\text{spec}}\) 作为负对。

这保证特异空间里确实刻画“哪个 cell”。

#### 4.2.2 特异空间小容量 + 稀疏

为了让“特别特征尽可能少、尽量集中在少数维度”，可以结合两种约束：

1. 维度本身设置得小：\(d_{\text{spec}} \ll d_{\text{com}}\)；
2. 对 \(g^{\text{spec}}_c\) 加上 l1 / l2 正则：

   \[
   \mathcal{L}_{\text{spec\_reg}} = 
   \lambda_{\text{spec},1} \|g^{\text{spec}}_c\|_1 
   + \lambda_{\text{spec},2} \|g^{\text{spec}}_c\|_2^2
   \]

这样，模型倾向于用**尽可能少的几维**就把细胞系区分开，从而让“特异特征尽可能特别，而不是到处乱抖动”。

#### 4.2.3 抗噪声：within‑cell 稳定

还可以对一个 cell 内的样本级特异向量加一致性约束：

\[
\mathcal{L}_{\text{within}} = 
\frac{1}{K}\sum_{i=1}^K 
\big\| z^{\text{spec}}_{c,i} - g^{\text{spec}}_c \big\|_2^2
\]

这保证特异特征在同一个 cell 内相对稳定，不会被单个样本的随机噪声主导。

---

## 5. 共同 vs 特异的整体博弈：总损失

综合起来，**高维投影任务 + BERT 任务** 的总损失可以写成：

\[
\begin{aligned}
\mathcal{L} =\;&
\underbrace{\mathcal{L}_{\text{BERT}}}_{\text{掩蔽预测}}
+ \lambda_\alpha \underbrace{\|\alpha\|_2^2}_{\text{共同 > 特异 的门控惩罚}} \\
&+ \lambda_{\text{invar}} \underbrace{\|g^{\text{com}}_c - \mu_{\text{com}}\|_2^2}_{\text{跨细胞公共一致}} \\
&+ \lambda_{\text{var}} \underbrace{\mathcal{L}_{\text{var,com}} + \mathcal{L}_{\text{cov,com}}}_{\text{公共空间“多而不冗余”}} \\
&+ \lambda_{\text{cell}} \underbrace{\mathcal{L}_{\text{cell}}}_{\text{特异空间区分细胞}} \\
&+ \lambda_{\text{spec}} \underbrace{\big(
\mathcal{L}_{\text{spec\_reg}} + \mathcal{L}_{\text{within}}
\big)}_{\text{特异空间稀疏且稳健}} \\
&+ \lambda_{\text{ortho}} \underbrace{\mathcal{L}_{\text{ortho}}}_{\text{公共/特异子空间互正交}}
\end{aligned}
\]

这些项共同实现了你想要的几个性质：

1. **共同特征“大”**：  
   - BERT 必须通过公共空间/门控组合 \((g^{\text{com}}_c, \alpha \tilde{g}^{\text{spec}}_c)\) 来完成任务；  
   - \(\lambda_\alpha\) 惩罚过多使用特异；  
   - 公共空间的方差/协方差正则让每个公共维都被充分利用。

2. **特异特征“小但有力”**：  
   - 低维 + 稀疏正则；  
   - 只在“区分 cell”的任务中发挥作用；  
   - within‑cell 一致性 + cell 分类损失让它稳定表征“这是谁”，而不是噪声。

3. **共同 vs 特异 占比是可学习的**：  
   - \(\alpha\) 作为门控权重，直接由 BERT 的梯度决定“到底需要多少特异信息”；  
   - 当公共空间足够表达 BERT 任务时，\(\alpha\to 0\)，特异只用于高维投影任务（cell 区分）；  
   - 当确实存在某些需要 cell‑context 才能解释的模式，\(\alpha\) 会保留一部分通路。

---

## 6. 训练流程（预训练阶段）总结

在“一个 batch = 一个细胞系”的设定下，预训练脚本每一步大致是：

1. 采样一个 cell \(c\)，取 K 条 EP 样本 \((e_i,p_i)\)；
2. 对每个样本：
   - 跑 backbone（CNN/self‑attn/cross‑attn/BERT encoder）；
   - 从序列/隐状态中构造时间序列 \(f_i\)（你已决定的方式），做 CWC → \(M_i\) → \(\Phi_{\text{CWC}}\) → \(v_i\)；
   - 得到 BERT 的 token 隐状态用于 MLM；
3. 对 \(\{v_i\}\) 做公共/特异投影：
   - \(z^{\text{com}}_{c,i} = W_{\text{com}} v_i\)；
   - \(z^{\text{spec}}_{c,i} = W_{\text{spec}} v_i\)；
   - \(g^{\text{com}}_c = \frac{1}{K}\sum z^{\text{com}}_{c,i}\)， \(g^{\text{spec}}_c = \frac{1}{K}\sum z^{\text{spec}}_{c,i}\)；
4. 构造 BERT 上下文向量：
   - \(u_c = g^{\text{com}}_c + \alpha \tilde{g}^{\text{spec}}_c\)；
   - 把 \(u_c\) 注入到 BERT 预测头中；
5. 计算所有损失项：\(\mathcal{L}_{\text{BERT}}, \mathcal{L}_{\alpha}, \mathcal{L}_{\text{invar}}, \dots\)；
6. 反向传播，更新：
   - backbone（适当）、\(\Phi_{\text{CWC}}, W_{\text{com}}, W_{\text{spec}}, \alpha, \text{MLP}_{\text{cell}}\) 等；
   - 同时更新 EMA 均值 \(\mu_{\text{com}}\)。

经过这样预训练后：

- 你得到的 \(W_{\text{com}},W_{\text{spec}}\) 就是“把小波 footprint 分解为共同/特异子空间的变换”；
- \(g^{\text{com}}_c, g^{\text{spec}}_c\) 在后续正式任务中可以直接作为 cell‑level footprint 来源，经由你前面讨论的“gate 控制 CBAT 注意力”机制参与函数化使用。

这就是你要的“**区分细胞 + 找到共同 footprint + 控制共同/特异占比**”的数学方案，高维投影部分已包括：

- 显式的子空间分解；
- BERT‑耦合的可学习门控；
- 公共/特异的方差、协方差、稀疏、稳定性约束。