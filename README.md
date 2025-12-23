# PRISM 研究报告：细胞系感知的增强子-启动子互作预测（供科研周报刊登）

## 摘要
- 问题：在不同细胞系分布下，直接从 DNA 序列预测增强子–启动子互作概率，易出现假阳性与阈值不稳定。
- 方法：提出能量耗散建模，将"结构驱动"与"环境抑制"显式分解为内势 $U_I$ 与阻抗 $R_E$，并以温度 $T>0$ 控制敏感度；结合域对抗以提升 OOD（分布外）泛化。
- 结论（方法层面）：概率由能量差驱动 
$$
P(y=1) = \sigma\left( \frac{U_I - R_E}{T} \right)
$$
其中 $R_E$ 仅在确有抑制证据时激活（L1 稀疏），从而降低假阳性并稳定跨细胞系阈值。

## 1. 问题定义与目标
- 输入：增强子序列 $x_E$、启动子序列 $x_P$，细胞系标识（可选）
- 输出：二分类概率 $P(y=1\mid x_E,x_P,\mathrm{cell})$
- 目标：在已知与未知细胞系上保持稳定的 AUPR/AUC 与 F1，降低 OOD 假阳性

## 2. 方法与架构
- 组件位置：主干与能量系统在 `models/PRISMModel.py:193`；训练循环在 `PRISM.py:208`；评估与阈值优化在 `predict.py:252`

### 2.1 序列编码与双向交互
- K-mer 编码与嵌入、1D CNN、RoPE 预注意力（`models/PRISMModel.py:233`、`models/PRISMModel.py:370`、`models/PRISMModel.py:391`）
- 双向跨序列注意（CBAT）：E→P 与 P→E 两个分支（`models/PRISMModel.py:411`、`models/PRISMModel.py:437`）
- 锚点与方向门控：利用对侧初始池化生成锚点并门控融合（`models/PRISMModel.py:394`、`models/PRISMModel.py:421`、`models/PRISMModel.py:455`）

形式化表述：
$$
h_E=\phi(x_E),\quad h_P=\phi(x_P);\quad
(h'_E,h'_P)=A(h_E,h_P;\ \mathrm{anchors},\ \mathrm{gates})
$$
其中 $\phi$ 为"嵌入+CNN+预注意力"表征，$A$ 为 CBAT 双向交互。

### 2.2 温度池化与主特征
令 $\text{norm}(\cdot)$ 为层归一化，$\mathrm{proj}(\cdot)$ 为线性投影，$\mathrm{softmax}$ 对长度维：
$$
w=\mathrm{softmax}\Big(\frac{\mathrm{proj}(\text{norm}(h'_E))}{T}\Big),\quad
y=\sum_t w_t\cdot h'_E(t),\quad z_I:=y
$$
其中温度 $T>0$ 由可学习参数经 $\mathrm{softplus}$ 保证正值（`models/PRISMModel.py:148`、`models/PRISMModel.py:491`）。

### 2.3 能量耗散概率模型
- 内势：$\ U_I=g_I(z_I)$（FourierEnergyKAN，`models/PRISMModel.py:259`、`models/PRISMModel.py:476`）
- 阻抗：$\ R_E=g_R(z_F)$（细胞系嵌入后经非负约束的 FourierEnergyKAN，`models/PRISMModel.py:275`、`models/PRISMModel.py:279`、`models/PRISMModel.py:480`）
- 概率：
$$
P(y=1\mid x_E,x_P,\mathrm{cell})=\sigma\Big(\frac{U_I-R_E}{T}\Big)
$$
实现于 `models/PRISMModel.py:490`–`models/PRISMModel.py:494`。

### 2.4 域对抗与解耦
- 对抗：对主特征 $z_I$ 施加 GRL（梯度反转），以去除细胞系可识别信息（`models/PRISMModel.py:469`–`models/PRISMModel.py:474`）
- 正交约束：鼓励 $z_I$ 与环境特征 $z_F$ 方向独立，增强分解的可解释性（`models/PRISMModel.py:665`–`models/PRISMModel.py:674`）

## 3. 训练目标与损失函数
- 组合目标：
$$
\begin{aligned}
L_{\text{total}}
&=L_{\text{base}}+L_{\text{attn}}+L_{\text{spec}}+L_{\text{sparse}}+L_{\text{orth}}+\alpha\,L_{\text{domain}}\
L_{\text{base}}&=\text{AdaptiveIMMAX}(\text{outputs},\ \text{labels})\
L_{\text{attn}}&=\text{CBAT 自适应项（双分支汇总）}\
L_{\text{spec}}&=\text{SpeculationPenalty}(\text{outputs},\ \text{labels})\
L_{\text{sparse}}&=\lambda_s\,\Vert R_E\Vert_1\quad(\text{鼓励阻抗在非必要时趋零})\
L_{\text{orth}}&=\lambda_o\,\big|\langle \text{norm}(z_I),\ \text{norm}(z_F)\rangle\big|\
L_{\text{domain}}&=\text{CE}(D(z_I^{\text{GRL}}),\ \text{cell\_id})
\end{aligned}
$$
- 实现位置：损失组合在 `models/PRISMModel.py:612`，域对抗加权在训练循环 `PRISM.py:339`–`PRISM.py:351`

## 4. 评估方案与指标定义
- 数据划分：`load_prism_data(split)`，评估入口 `predict.py:252`
- 阈值优化：
$$
\tau^*=\arg\max_{\tau\in[\tau_{\min},\tau_{\max}]}M(\tau),\quad
M\in\{\text{F1},\ \text{Precision},\ \text{Recall},\ \text{Accuracy}\}
$$
实现与可视化见 `predict.py:309`–`predict.py:345`。
- 指标定义（对二值化阈值 $\tau$）：
$$
\text{Precision}(\tau)=\frac{TP(\tau)}{TP(\tau)+FP(\tau)},\quad
\text{Recall}(\tau)=\frac{TP(\tau)}{TP(\tau)+FN(\tau)}
$$
$$
F1(\tau)=\frac{2\cdot \text{Precision}(\tau)\cdot \text{Recall}(\tau)}{\text{Precision}(\tau)+\text{Recall}(\tau)}
$$
- AUPR/ROC-AUC：分别为 PR 与 ROC 曲线下的面积；分细胞系统计与阈值可在 `predict.py:374`–`predict.py:411` 获取。

## 5. 实验设置与实现索引
- 训练数据流程与批处理：`PRISM.py:67`–`PRISM.py:133`，细胞系标签映射 `PRISM.py:243`–`PRISM.py:265`
- 主干构建与参数统计：`PRISM.py:265`–`PRISM.py:277`
- 检查点保存与恢复：`PRISM.py:398`–`PRISM.py:408`，恢复加载 `PRISM.py:153`–`PRISM.py:199`
- 推理可视化输出目录：`compete/{SAVEMODEL_NAME}`（PR、ROC、阈值–指标曲线）

## 6. 预期表现与消融建议
- 阻抗稀疏（$L_{\text{sparse}}$）减少 OOD 假阳性，稳定跨细胞系阈值
- 域对抗（$L_{\text{domain}}$）提升 $z_I$ 的域不可分性，缓解分布偏移
- 正交约束（$L_{\text{orth}}$）增强结构–环境解耦的可解释性
- 建议消融：分别关闭 $R_E$、GRL、锚点门控，观察 AUPR、F1 的变化

## 7. 结论
- 能量式概率 \(P(y)=\sigma((U_I-R_E)/T)\) 将结构驱动与环境抑制统一于可解释框架
- 稀疏阻抗与域对抗在 OOD 条件下抑制假阳性，温度池化调控阈值敏感度
- 双向交互与门控机制提升 E/P 对齐与稳定性，为跨细胞系泛化提供支撑

本项目围绕 PRISM（Promoter–Enhancer Interaction with Structured Modulation）研究型系统展开。目标是从 DNA 序列直接预测增强子与启动子的互作概率，并在不同细胞系环境下保持稳健的泛化能力。代码仓库以科研为核心导向，强调清晰的数据流、可解释的能量式建模、以及面向 OOD（分布外）细胞系的鲁棒性设计。

- 研究对象：增强子-启动子互作概率预测（EP prediction）
- 关键挑战：跨细胞系分布差异、假阳性控制、阈值稳定性、稳健泛化
- 核心指标：AUPR、AUC、F1、Precision/Recall（支持全局与分细胞系评估）

## 架构总览（旧版节略，详见上文“方法与架构”）

PRISM 由三个高度内聚的脚本/模块协同构成：

- `models/PRISMModel.py`：主干网络与能量耗散系统的核心实现（`PRISMBackbone`）。
- `PRISM.py`：训练流程（数据装载、损失计算、域对抗、检查点保存）。
- `predict.py`：推理评估流程（加载检查点、阈值优化、分细胞系统计与可视化）。

整体数据流：
1. 原始 DNA 序列经 K-mer 分词为 token ID。
2. CNN 特征提取 + RoPE 预注意力，得到 E/P 两路表征。
3. 采用 CBAT 跨序列注意力实现 E→P 与 P→E 的双向交互，注入"锚点"与方向门控以稳定对齐。
4. 序列池化后得到主特征 `z_I`（内在势能特征），通过能量式头部生成内势 $U_I$。
5. 细胞系嵌入生成环境阻抗 $R_E$，联合可学习温度系数 $T$，以能量耗散形式计算概率：
   - $P(y=1) = \text{sigmoid}((U_I - R_E) / T)$
6. 训练中通过 GRL（梯度反转层）对 `z_I` 执行域对抗，剔除环境信息，提升泛化与可解释性。

## 数学定义与建模公式

- 符号说明：
  - $x_E, x_P$：增强子与启动子 DNA 序列
  - $\phi(\cdot)$：K-mer 编码与嵌入 + CNN + RoPE 预注意力的表征函数
  - $A(\cdot)$：CBAT 跨序列注意力（含锚点与门控）
  - $\text{Pool}_T(\cdot)$：带温度的序列池化（温度 $T>0$ 控制 softmax 平滑度）
  - $z_I$：内在势能特征；$z_F$：环境特征（细胞系嵌入）
  - $U_I$：内在势能；$R_E$：环境阻抗；$T$：温度系数

- 特征生成：
  - $h_E = \phi(x_E)$，$h_P = \phi(x_P)$
  - 双向交互：$h'_E, h'_P = A(h_E, h_P; \text{anchors}, \text{gates})$
  - 温度池化（以增强子为例）：$y = \text{Pool}_T(h'_E; T_{\text{anchor}}(h_P))$
  - 主特征：$z_I = y$

- 能量式概率：
  - 内势：$U_I = g_I(z_I)$（$g_I$ 为 FourierEnergyKAN，非线性可解释）
  - 阻抗：$R_E = g_R(z_F)$（细胞系嵌入后经非负约束的 FourierEnergyKAN）
  - 温度：$T = \text{softplus}(\theta_T)$ 保证正值
  - 概率：
    - $P(y=1 | x_E, x_P, \text{cell}) = \sigma((U_I - R_E) / T)$

- 约束与损失：
  - 基础任务损失（自适应 IMMAX）：$L_{\text{base}} = \text{AdaptiveIMMAX}(\text{outputs}, \text{labels})$
  - 注意力自适应项（CBAT 返回）：$L_{\text{attn}}$
  - 投机惩罚：$L_{\text{spec}} = \text{SpecPenalty}(\text{outputs}, \text{labels})$（抑制不确定区间的过度自信）
  - 稀疏阻抗：$L_{\text{sparse}} = \lambda_s \cdot ||R_E||_1$（鼓励在非必要时环境项趋零）
  - 正交约束：$L_{\text{orth}} = \lambda_o \cdot |\langle \text{norm}(z_I), \text{norm}(z_F) \rangle|$
  - 域对抗（训练循环中加入）：$L_{\text{domain}} = \text{CE}(D(z_I^{\text{GRL}}), \text{cell}\_\text{id})$
  - 总损失（训练循环加权汇总）：
    - $L_{\text{total}} = L_{\text{base}} + L_{\text{attn}} + L_{\text{spec}} + L_{\text{sparse}} + L_{\text{orth}} + \alpha \cdot L_{\text{domain}}$

解释：能量式将预测分解为"结构驱动（$U_I$）—环境抑制（$R_E$）—温度敏感度（$T$）"，其中 $R_E$ 通过 L1 稀疏与域对抗共同约束，仅在确有环境抑制证据时显著发挥作用，从而稳定阈值并抑制假阳性。