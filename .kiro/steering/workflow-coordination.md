# PRISM三角色协作工作流

## 系统概述
本工作流实现算法分析师、方案执行者、质检者三角色协作，通过Hook系统自动化触发，确保PRISM项目高效推进至AUPR≥0.75目标。

## 角色职责分工
- **算法分析师**：深度问题诊断 + 创新方案设计
- **方案执行者**：代码实现 + 实验执行
- **质检者**：方案评估 + 训练监控 + 流程决策

## 工作流状态机

```
[项目启动/预测完成] 
    ↓
[算法分析师：问题诊断 + 方案设计] → 创建 docx/记录点N/记录点N方案.md
    ↓ (Hook: docx-solution-monitor 触发)
[质检者：方案评估] → 新建聊天会话
    ↓
[通过] → [方案执行者：代码实现 + 训练] → 写入 hook/train.txt "done"
    ↓ (Hook: train-completion-hook 触发)
[质检者：训练结果评估] → 新建聊天会话
    ↓
[正常] → [方案执行者：预测评估] → 写入 hook/predict.txt "done"
    ↓ (Hook: predict-done-hook 触发)
[算法分析师：结果分析] → 新建聊天会话 → [达标检查]
    ↓
[达标] → [项目完成]
[不达标] → [算法分析师：问题诊断] (循环)

[不通过] → [算法分析师：重新设计] (新建聊天会话)
[异常/过拟合] → [算法分析师：重新设计] (新建聊天会话)
```

## Hook触发机制详解

### 核心Hook（自动触发）

| Hook名称 | 监控路径 | 触发事件 | 执行动作 |
|----------|----------|----------|----------|
| docx-solution-monitor | `./docx/记录点*/记录点*方案*` | fileCreated | 质检者评估方案 |
| train-completion-hook | `./hook/train.txt` | fileCreated | 质检者评估训练 |
| predict-done-hook | `./hook/predict.txt` | fileCreated | 算法分析师分析结果 |

### 决策分支Hook（质检者写入触发）

| Hook名称 | 监控路径 | 触发事件 | 执行动作 |
|----------|----------|----------|----------|
| solution-review-pass-hook | `./hook/solution_pass.txt` | fileCreated | 方案执行者实现+训练 |
| solution-review-reject-hook | `./hook/solution_reject.txt` | fileCreated | 算法分析师重新设计 |
| train-review-pass-hook | `./hook/train_pass.txt` | fileCreated | 方案执行者执行预测 |
| train-review-fix-hook | `./hook/train_fix.txt` | fileCreated | 方案执行者修复重训 |
| train-review-redesign-hook | `./hook/train_redesign.txt` | fileCreated | 算法分析师重新设计 |

### 重要说明
- **所有Hook使用fileCreated事件**：脚本在开始时删除hook文件，完成时创建新文件
- **PRISM.py**：训练开始时删除`hook/train.txt`，训练完成后创建（内容为"done"）
- **predict.py**：预测开始时删除`hook/predict.txt`，预测完成后创建（内容为"done"）
- **质检者决策**：通过创建对应的hook文件（如`hook/train_pass.txt`）触发下一步

## 路径配置（基于config.py）

### 训练相关路径
- **训练日志**：`{PRISM_SAVE_MODEL_DIR}/log/` (save_model/baseline/log/)
- **模型检查点**：`{PRISM_SAVE_MODEL_DIR}/` (save_model/baseline/)
- **训练完成标志**：`./hook/train.txt`

### 预测相关路径
- **预测结果**：`compete/{SAVEMODEL_NAME}/` (compete/baseline/)
- **预测完成标志**：`./hook/predict.txt`

### 方案管理路径
- **方案文档**：`./docx/记录点N/记录点N方案.md`
- **历史索引**：`./docx/历史索引.md`
- **基线结果**：`./docx/基线结果.log`

## 决策节点详解

### 节点1：方案质量评估
**触发条件**：算法分析师提交方案 (docx-solution-monitor Hook)
**质检者判断**：
- 通过 → 在新聊天会话中激活方案执行者
- 不通过 → 在新聊天会话中要求算法分析师重新设计

**评估标准**：
- 创新性≥6分 且 可行性≥7分 且 有效性预测≥7分
- 与历史中无效的方案有本质差异，不会重蹈覆辙
- 有明确的理论支撑和实现路径

### 节点2：训练结果评估
**触发条件**：训练完成 (train-completion-hook Hook)
**质检者判断**：
- 正常 → 在新聊天会话中指示方案执行者继续预测流程
- 过拟合 → 在新聊天会话中要求算法分析师重新设计
- 异常 → 在新聊天会话中要求方案执行者修复重训

**评估标准**：
- 收敛正常，无数值异常
- 学习效率≥ 0.9 AUPR/epoch
- 无明显过拟合迹象

### 节点3：最终验收
**触发条件**：预测完成 (predict-done-hook Hook)
**算法分析师判断**：
- domain-kl/test AUPR ≥ 0.75 → 项目成功
- AUPR < 0.75 → 在新聊天会话中开始下一轮迭代

## 角色激活机制

### 算法分析师激活条件
1. 项目启动时 (手动)
2. 预测完成后进行结果分析 (predict-done-hook)
3. 质检者判断方案不合格时 (新聊天会话)
4. 质检者判断训练过拟合时 (新聊天会话)

### 方案执行者激活条件
1. 质检者通过方案评估时 (新聊天会话)
2. 质检者要求修复重训时 (新聊天会话)
3. 质检者指示进行预测评估时 (新聊天会话)

### 质检者激活条件
1. 算法分析师提交方案时 (docx-solution-monitor Hook，新聊天会话)
2. 方案执行者完成训练时 (train-completion-hook Hook，新聊天会话)

## 防止协同造假机制

### 独立会话
- 每个角色在独立的聊天会话中工作
- 质检者的评估不受其他角色主观影响
- 所有评估基于具体数据和日志证据

### 历史对比
- 强制检查与历史方案的差异性
- 防止重复失败方案的提交
- 维护失败方案数据库

### 严格标准
- 方案通过标准：创新性≥6分 且 可行性≥7分
- 训练正常标准：收敛+效率+稳定性
- 最终验收标准：AUPR≥0.75

## 效率优化机制

### 快速失败
- 方案评估阶段快速识别低质量方案
- 训练异常早期识别和中断
- 避免无意义的长时间实验

### 自动化触发
- Hook系统自动监控关键文件变化
- 无需人工干预的工作流推进
- 实时响应实验状态变化

### 知识积累
- 每轮实验的经验教训记录
- 成功/失败模式的系统化总结
- 历史数据的充分利用

## 当前项目状态
- **记录点**：16
- **当前方案**：自适应温度场与多尺度能量耦合
- **性能现状**：最佳AUPR约0.72，距离目标0.75还有0.03差距
- **主要挑战**：固定温度T导致的数学病态性，跨细胞系泛化不足

## Hook文件状态监控

### 自动生成的Hook文件
| 文件 | 生成时机 | 触发的Hook |
|------|----------|------------|
| `./hook/train.txt` | PRISM.py训练完成后创建 | train-completion-hook |
| `./hook/predict.txt` | predict.py预测完成后创建 | predict-done-hook |

### 质检者决策写入的Hook文件
| 文件 | 写入时机 | 触发的Hook |
|------|----------|------------|
| `./hook/solution_pass.txt` | 方案评审通过 | solution-review-pass-hook |
| `./hook/solution_reject.txt` | 方案评审不通过 | solution-review-reject-hook |
| `./hook/train_pass.txt` | 训练评审正常 | train-review-pass-hook |
| `./hook/train_fix.txt` | 训练评审需修复 | train-review-fix-hook |
| `./hook/train_redesign.txt` | 训练评审需重设计 | train-review-redesign-hook |

### 注意事项
- 脚本在开始时会删除对应的hook文件，防止误触发
- 只有任务完成后才会创建hook文件
- 质检者需要通过创建对应文件来触发下一步流程

## 成功标准
- **技术指标**：所有测试集AUPR≥0.75
- **稳定性**：连续3次实验达标
- **泛化性**：未见细胞系性能稳定
- **效率**：总迭代轮数≤20轮

## 工作流启动
1. **手动启动**：算法分析师分析当前状态，设计新方案
2. **自动推进**：Hook系统监控文件变化，自动触发相应角色
3. **独立会话**：每次角色切换都在新的聊天会话中进行，确保权责清晰