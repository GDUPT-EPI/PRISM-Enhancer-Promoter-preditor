Kiro Agent Hooks（最小循环）
========================

Hook 的职责：在完成一次 `predict.py` 之后，做一次验收判断，并据此重新拉起 `workflow/kiro_agent_steering.md` 的循环。


Hook 01：Predict 结果验收并重启循环
--------------------------------
- Trigger：`predict.py` 运行完成，产出本轮预测结果。
- Inputs：
  - `log/记录点(n+1)预测结果.txt`
  - `docx/基线结果.log`
  - `docx/历史索引.md`
- Decision：
  - 若未见细胞系（domain-kl/test）与各细胞系 AUPR ≥ 0.75：进入“达标稳定性验证”分支。
  - 否则：进入“继续迭代”分支。

- Actions（达标稳定性验证）：
  - 进入 `workflow/kiro_agent_steering.md` 的“达标后流程”，仅做复现与小修复，不做结构性大改。

- Actions（继续迭代）：
  - 将本轮差距与失败模式写入当前记录点反思：`docx/记录点(n+1)/记录点(n+1)反思.md`。
  - 在 `docx/历史索引.md` 中确保本轮记录点条目完整。
  - 重新从头执行 `workflow/kiro_agent_steering.md` 的“迭代循环”。

