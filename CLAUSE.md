# PRISM项目 - Claude Code配置

## 项目概述

PRISM（预测基因组序列中增强子-启动子互作）项目是一个机器学习研究项目，目标是在仅使用DNA序列信息的前提下，预测增强子-启动子(EP)互作，达到AUPR ≥ 0.75的目标。

## 配置结构

本项目使用Claude Code规范进行配置，目录结构如下：

```
.claude/                    # Claude Code配置目录
├── config.json            # 主配置文件
├── agents/                # Agents配置目录
│   ├── agent-analyst.json      # 算法分析师配置
│   ├── agent-executor.json     # 方案执行者配置
│   └── agent-inspector.json    # 质检者配置
├── hooks/                 # Hooks配置目录
│   └── prism-workflow-hooks.json  # 工作流钩子配置
├── workflows/             # Workflows配置目录
│   └── ep-workflow.json          # EP互作预测工作流
└── context/              # 上下文配置目录
    ├── project-context.json      # 项目背景配置
    └── mathematical-context.json # 数学上下文配置
```

## 三角色协作系统

### 1. 算法分析师 (Algorithm Analyst)
- **角色**：拥有深邃数学直觉的算法分析师
- **职责**：深度问题诊断、创新方案设计、反思总结、结果整理
- **配置文件**：`.claude/agents/agent-analyst.json`

### 2. 方案执行者 (Solution Executor)
- **角色**：精通PyTorch实现和代码工程的方案执行者
- **职责**：深度理解方案、高质量代码实现、版本管理、实验执行
- **配置文件**：`.claude/agents/agent-executor.json`

### 3. 质检者 (Quality Inspector)
- **角色**：具备深厚机器学习理论功底和敏锐工程直觉的质检者
- **职责**：方案质量评估、训练结果监控、流程决策
- **配置文件**：`.claude/agents/agent-inspector.json`

## 自动化工作流

### 工作流状态机
```
[项目启动/预测完成] → 算法分析师 → 创建方案文档 → (Hook触发) → 质检者评估方案
    ↓ (通过) → 方案执行者实现+训练 → 训练完成 → (Hook触发) → 质检者评估训练
    ↓ (正常) → 方案执行者预测 → 预测完成 → (Hook触发) → 算法分析师分析结果
    ↓ (达标) → 项目完成
    ↓ (不达标) → 算法分析师重新设计 (循环)
```

### 关键Hook触发点
1. **方案文档监控器**：监控`docx/记录点*/记录点*方案*`文件创建，触发质检者方案评估
2. **训练完成监控器**：监控`./hook/train.txt`文件创建，触发质检者训练评估
3. **预测完成监控器**：监控`./hook/predict.txt`文件创建，触发算法分析师结果分析
4. **方案通过触发器**：监控`./hook/solution_pass.txt`文件创建，触发方案执行者
5. **方案拒绝触发器**：监控`./hook/solution_reject.txt`文件创建，触发算法分析师重新设计
6. **训练通过触发器**：监控`./hook/train_pass.txt`文件创建，触发方案执行者预测
7. **训练修复触发器**：监控`./hook/train_fix.txt`文件创建，触发方案执行者修复重训
8. **训练重新设计触发器**：监控`./hook/train_redesign.txt`文件创建，触发算法分析师重新设计
9. **项目启动触发器**：手动命令`start_prism_iteration`触发算法分析师开始新一轮分析

## 项目当前状态

- **当前记录点**：16
- **目标性能**：AUPR ≥ 0.75
- **当前性能**：AUPR ~0.72
- **所需提升**：~4% 相对提升
- **核心挑战**：
  1. 细胞系泛化：不同细胞系之间的特异性差异
  2. S1-S2竞争失效：学生与对手的竞争机制问题
  3. 上下文感知缺失：模型缺乏序列上下文感知能力

## 技术架构

### 数据处理
- **输入**：DNA序列
- **编码**：6-mer K-mer分词
- **输出**：EP互作概率

### 模型架构
- **主干网络**：CNN特征提取 + Transformer交叉注意力
- **能量耗散框架**：内势(U_I) - 环境阻抗(R_E)的物理启发模型
- **细胞系处理**：域对抗训练处理不同细胞系
- **关键技术**：CBAT(卷积块注意力模块)、FourierKAN、RoPE(旋转位置编码)

### 评估指标
- 主要指标：AUPR (Area Under Precision-Recall Curve)
- 辅助指标：AUC、Precision、Recall、F1 Score

## 使用方法

### 启动工作流
```bash
# 启动Claude Code工作流监控
claude workflow start prism
```

### 手动触发
```bash
# 手动启动新一轮迭代
claude hook trigger start_prism_iteration
```

### 状态查询
```bash
# 查看工作流状态
claude workflow status

# 查看hook状态
claude hook status
```

### 配置文件验证
```bash
# 验证配置文件
claude config validate --config .claude/config.json
```

## 配置迁移说明

本配置是从原始的`.kiro/`目录结构迁移而来：
- **原始agent配置**：`.kiro/steering/` → `.claude/agents/`
- **原始hook配置**：`.kiro/hooks/` → `.claude/hooks/prism-workflow-hooks.json`
- **原始工作流**：`workflow/EP_workflow.md` → `.claude/workflows/ep-workflow.json`
- **项目背景**：整合到`.claude/context/`目录

## 注意事项

1. **文件路径**：所有文件路径基于项目根目录`/home/z/CBAT/`
2. **自动化脚本**：训练脚本`PRISM.py`和预测脚本`predict.py`会自动管理hook文件
3. **版本控制**：代码修改后需要执行`git add -A`、`git commit`、`git push`流程
4. **文档规范**：所有方案、反思、结果文档需按照指定模板创建
5. **质量控制**：质检者在两个关键节点（方案评估、训练评估）进行严格质量控制

## 故障排除

### Hook未触发
1. 检查hook文件是否存在且可读
2. 检查监控的文件路径是否正确
3. 检查Claude Code服务是否正常运行

### 工作流中断
1. 检查是否创建了正确的hook决策文件
2. 检查相关agent配置是否正确
3. 查看Claude Code日志获取详细信息

### 配置错误
1. 使用`claude config validate`验证配置文件
2. 检查JSON格式是否正确
3. 确认文件路径和权限

---

*本配置使用Claude Code规范，旨在实现PRISM项目的自动化、高质量的迭代开发流程。*