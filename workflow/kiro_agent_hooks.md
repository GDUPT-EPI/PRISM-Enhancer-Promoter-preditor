Kiro Agent Hooks（EP互作预测工作流）
============================

本文件将 `workflow/EP_workflow.md` 拆解为可执行的“Hook 列表”。
Hook 的定位：当 Agent 进入某个阶段或检测到某个条件时，按固定动作产出固定工件，确保迭代闭环、避免遗漏与“只写代码不写反思/方案”。

Hook 约定
--------
- 触发点（Trigger）：进入某阶段 / 发现某信号 / 完成某动作后。
- 输入（Inputs）：Agent 需要读取的文件、日志或上下文。
- 动作（Actions）：必须执行的检查、对齐、汇总或写入。
- 产出（Artifacts）：必须落盘的文件（相对项目根目录）。
- 退出条件（Exit Criteria）：满足后才允许进入下一阶段。


Hook 01：启动对齐（Start-Of-Iteration）
------------------------------------
- Trigger：开始一次新的迭代轮次（准备从记录点 n 推进到 n+1）。
- Inputs：
  - `docx/历史索引.md`
  - `docx/基线结果.log`
  - `docx/记录点n/记录点n反思.md`（若存在）
  - `docx/记录点n/记录点n方案.md`（若存在）
- Actions：
  - 确认当前记录点编号 n（以 `docx/历史索引.md` 的最新条目为准）。
  - 抽取本轮必须遵守的“硬约束”：纯序列输入、未见细胞系鲁棒性目标、AUPR≥0.75。
  - 列出本轮“不可回归点”：历史索引里已明确不可行的路径（如模式坍缩成因、推理断裂等）。
- Artifacts：无（本 Hook 只做对齐，不写入）。
- Exit Criteria：明确 n，并能用 3–5 句话复述本轮约束与主要风险。


Hook 02：差距诊断（Gap Diagnosis）
-------------------------------
- Trigger：完成 Hook 01 后。
- Inputs：
  - `docx/基线结果.log`
  - `log/记录点n训练日志.log`（若存在）
  - `log/记录点n预测结果.txt`（若存在）
  - `docx/记录点n/记录点n结果.md`（若存在）
- Actions：
  - 用“目标阈值 AUPR≥0.75”作为唯一验收锚点，提取：
    - domain-kl/test 的 AUPR
    - 各细胞系 AUPR（若有）
  - 归因失败模式（必须二选一或多选，但不能空泛）：
    - 排序能力不足（PR 曲线整体不佳）
    - 置信度上移/下移导致 Precision/Recall 失衡
    - 模式坍缩（输出接近常数或极端 0/1）
    - 训练-推理不一致（训练用到的模块推理没用到）
    - 细胞系模块未生效（旁路不影响最终预测或仅在 loss 中出现）
- Artifacts：无（诊断结论写入 Hook 03）。
- Exit Criteria：能给出“差距数值 + 失败模式 + 最可能根因”的三段式结论。


Hook 03：记录点 n 反思写入（Write Reflection）
-------------------------------------------
- Trigger：完成 Hook 02 后。
- Inputs：
  - Hook 02 的差距诊断结论
  - `docx/历史索引.md` 中记录点 n 的“结论/反思”条目
- Actions：
  - 在 `docx/记录点n/记录点n反思.md` 末尾追加“本轮反思”段落（保留历史内容）。
  - 反思必须包含四块：
    - 现象：AUPR 与 Precision/Recall 的表现（含未见细胞系）
    - 根因：用数据流/损失/推理路径描述，不用形容词
    - 对照：与 `历史索引.md` 的既有结论一致/冲突点
    - 约束：下一轮必须保持什么不变（防止引入新变量）
- Artifacts：
  - `docx/记录点n/记录点n反思.md`
- Exit Criteria：反思中出现明确的“下一轮可检验假设”与“必须避免的坑”。


Hook 04：记录点 n+1 方案写入（Write Plan）
----------------------------------------
- Trigger：完成 Hook 03 后。
- Inputs：
  - `docx/记录点n/记录点n反思.md`（含新增段落）
  - `docx/历史索引.md`（用于风格与因果链对齐）
- Actions：
  - 在 `docx/记录点(n+1)/记录点(n+1)方案.md` 写入方案（若无则创建目录与文件）。
  - 方案必须包含五块：
    - 目标：本轮要提升的指标与范围（至少包含未见细胞系 AUPR）
    - 最小改动：只允许改动 1–2 个核心机制（避免变量爆炸）
    - 数据流：训练与推理都必须使用同一形式的细胞系模块输入
    - 风险：本改动可能触发的失败模式以及如何快速判定
    - 成功判据：量化门槛（例如 AUPR 提升 ≥0.03 或达到 ≥0.75）
- Artifacts：
  - `docx/记录点(n+1)/记录点(n+1)方案.md`
- Exit Criteria：能用“假设-改动-预期观测”三元组描述方案。


Hook 05：历史索引更新（Update History Index）
-------------------------------------------
- Trigger：完成 Hook 04 后。
- Inputs：
  - `docx/记录点(n+1)/记录点(n+1)方案.md`
  - `docx/历史索引.md`
- Actions：
  - 在 `docx/历史索引.md` 追加“记录点 n+1”条目，内容仅保留：
    - 结构改动（1–3 行）
    - 要解决的问题（1–3 行）
    - 成功/失败标准（1–2 行）
  - 保持与记录点 1–13 的写法一致（可读、可追溯）。
- Artifacts：
  - `docx/历史索引.md`
- Exit Criteria：索引新增条目可独立让读者理解“为什么改、改了什么、怎么判定”。


Hook 06：代码前置核对（Pre-Code Gate）
------------------------------------
- Trigger：准备进入代码修改前。
- Inputs：
  - `docx/记录点(n+1)/记录点(n+1)方案.md`
  - `models/PRISMModel.py`
  - `models/layers/distill.py`
  - `PRISM.py`
  - `predict.py`
- Actions：
  - 在脑内做一次“训练-推理一致性审计”：
    - 细胞系模块在训练与推理是否都参与最终预测
    - 推理是否需要 Support Set/Context Batch，是否在 `predict.py` 体现
  - 明确本轮要修改的文件范围与入口函数（不扩散到无关模块）。
- Artifacts：无（进入代码阶段后再产出）。
- Exit Criteria：能把本轮改动映射到“哪一个文件/哪一条数据流/哪一个损失或推理公式”。


Hook 07：训练执行与日志落盘（Run Training）
----------------------------------------
- Trigger：完成代码修改并准备训练。
- Inputs：
  - `PRISM.py`
  - 本轮方案中的训练设置要点
- Actions：
  - 运行训练并确保日志可追溯（记录点编号、关键开关、时间）。
  - 确保训练输出能支持 Hook 08 的指标提取。
- Artifacts：
  - `log/记录点(n+1)训练日志.log`
- Exit Criteria：日志中包含关键指标（至少 loss、AUPR/AUC 或可计算它们的输出）。


Hook 08：预测执行与结果落盘（Run Predict）
----------------------------------------
- Trigger：训练完成后。
- Inputs：
  - `predict.py`
  - 训练得到的模型权重位置（由项目既有逻辑决定）
- Actions：
  - 运行预测，确保按细胞系汇总指标，特别是未见细胞系。
- Artifacts：
  - `log/记录点(n+1)预测结果.txt`
- Exit Criteria：输出包含各细胞系（含 domain-kl/test）的 AUPR（至少可定位到数值）。


Hook 09：结果总结写入（Write Results）
-----------------------------------
- Trigger：完成 Hook 08 后。
- Inputs：
  - `log/记录点(n+1)训练日志.log`
  - `log/记录点(n+1)预测结果.txt`
  - `docx/基线结果.log`
- Actions：
  - 在 `docx/记录点(n+1)/记录点(n+1)结果.md` 写入：
    - 本轮与基线/上一记录点的 AUPR 对比
    - 未见细胞系是否达标（≥0.75）
    - 若未达标：最明显的失败模式与证据（为下一轮 Hook 02 提供输入）
- Artifacts：
  - `docx/记录点(n+1)/记录点(n+1)结果.md`
- Exit Criteria：结果文件可直接支持下一轮的差距诊断。


Hook 10：循环控制（Iteration Controller）
-------------------------------------
- Trigger：完成 Hook 09 后。
- Inputs：
  - `docx/记录点(n+1)/记录点(n+1)结果.md`
- Actions：
  - 若 AUPR≥0.75：进入“稳定性验证”而非继续大改。
  - 若 AUPR<0.75：将 n 设为 n+1，回到 Hook 01，开始下一轮闭环。
- Artifacts：无。
- Exit Criteria：明确下一步是“收敛验证”还是“继续迭代”。

