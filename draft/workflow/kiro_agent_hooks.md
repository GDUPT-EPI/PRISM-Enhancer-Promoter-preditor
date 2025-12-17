# PRISM项目Agent Hooks配置

## Hook系统概述
本文件定义自动化触发机制，实现三角色协作的无缝衔接。Hook在关键节点自动拉起相应的Agent，确保工作流高效运转。

## Hook配置列表

### Hook 1: 方案提交后质检触发
**触发条件**：算法分析师创建方案文件
**监控路径**：`docx/记录点*/记录点*方案.md`
**触发事件**：文件创建或修改
**执行动作**：激活质检者进行方案评估

```json
{
  "name": "方案质检触发器",
  "trigger": {
    "type": "file_change",
    "pattern": "docx/记录点*/记录点*方案.md",
    "events": ["created", "modified"]
  },
  "action": {
    "type": "agent_message",
    "agent": "质检者",
    "message": "算法分析师已提交新方案，请进行方案质量评估。方案路径：{file_path}"
  }
}
```

### Hook 2: 训练完成后质检触发
**触发条件**：训练日志文件生成
**监控路径**：`log/记录点*训练日志.log`
**触发事件**：文件创建
**执行动作**：激活质检者进行训练结果评估

```json
{
  "name": "训练质检触发器",
  "trigger": {
    "type": "file_change",
    "pattern": "log/记录点*训练日志.log",
    "events": ["created"]
  },
  "action": {
    "type": "agent_message",
    "agent": "质检者",
    "message": "训练已完成，请进行训练结果评估。日志路径：{file_path}"
  }
}
```

### Hook 3: 预测完成后验收触发
**触发条件**：预测结果文件生成
**监控路径**：`log/记录点*预测结果.txt`
**触发事件**：文件创建
**执行动作**：激活质检者进行最终验收

```json
{
  "name": "预测验收触发器",
  "trigger": {
    "type": "file_change",
    "pattern": "log/记录点*预测结果.txt",
    "events": ["created"]
  },
  "action": {
    "type": "agent_message",
    "agent": "质检者",
    "message": "预测已完成，请进行最终验收评估。结果路径：{file_path}"
  }
}
```

### Hook 4: 质检不通过后算法分析师触发
**触发条件**：质检者输出不通过报告
**监控关键词**：质检报告中的"要求算法分析师重新设计"
**执行动作**：激活算法分析师重新分析问题

```json
{
  "name": "重新设计触发器",
  "trigger": {
    "type": "message_content",
    "pattern": "要求算法分析师重新设计",
    "source": "质检者"
  },
  "action": {
    "type": "agent_message",
    "agent": "算法分析师",
    "message": "质检者要求重新设计方案，请分析问题并提出新的解决方案。"
  }
}
```

### Hook 5: 质检通过后方案执行者触发
**触发条件**：质检者输出通过报告
**监控关键词**：质检报告中的"激活方案执行者"
**执行动作**：激活方案执行者开始代码实现

```json
{
  "name": "方案执行触发器",
  "trigger": {
    "type": "message_content",
    "pattern": "激活方案执行者",
    "source": "质检者"
  },
  "action": {
    "type": "agent_message",
    "agent": "方案执行者",
    "message": "方案已通过质检，请开始代码实现和实验执行。"
  }
}
```

### Hook 6: 项目启动触发器
**触发条件**：手动启动或达标失败
**执行动作**：激活算法分析师开始新一轮分析

```json
{
  "name": "项目启动触发器",
  "trigger": {
    "type": "manual",
    "command": "start_prism_iteration"
  },
  "action": {
    "type": "agent_message",
    "agent": "算法分析师",
    "message": "开始新一轮PRISM优化迭代，请分析当前问题并设计解决方案。目标：AUPR≥0.75"
  }
}
```

## Hook执行逻辑

### 自动化流程
1. **文件监控**：持续监控关键文件的变化
2. **条件匹配**：检查触发条件是否满足
3. **Agent激活**：向指定Agent发送激活消息
4. **状态跟踪**：记录Hook执行历史和状态

### 错误处理
- **Hook失效**：如果Hook未能正确触发，提供手动激活机制
- **重复触发**：防止同一事件重复触发Hook
- **异常恢复**：Hook执行异常时的恢复机制

### 调试模式
- **日志记录**：记录所有Hook触发和执行情况
- **手动测试**：提供手动触发Hook的测试接口
- **状态查询**：查询当前Hook状态和历史记录

## 特殊场景处理

### 场景1：训练异常中断
**处理方式**：
1. 检测训练日志中的异常关键词（NaN, OOM, Error）
2. 立即触发质检者评估
3. 质检者判断是否需要方案执行者修复或算法分析师重新设计

### 场景2：长时间无响应
**处理方式**：
1. 设置超时机制（如24小时无新文件）
2. 自动发送提醒消息给相应Agent
3. 提供手动干预接口

### 场景3：达标后稳定性验证
**处理方式**：
1. 连续3次达标后进入稳定性验证模式
2. 自动触发多次重复实验
3. 确认稳定性后正式完成项目

## 当前Hook状态
- **活跃Hook**：6个
- **监控文件**：docx/记录点*/, log/记录点*
- **下一个预期触发**：根据记录点15状态确定

## Hook维护
- **定期检查**：每周检查Hook配置的有效性
- **性能优化**：优化文件监控的性能开销
- **规则更新**：根据项目进展更新Hook规则

## 使用说明
1. **启动监控**：`kiro hook start prism-workflow`
2. **查看状态**：`kiro hook status`
3. **手动触发**：`kiro hook trigger <hook_name>`
4. **停止监控**：`kiro hook stop prism-workflow`