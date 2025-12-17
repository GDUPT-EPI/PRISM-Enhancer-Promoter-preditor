#!/usr/bin/env python3
"""
PRISM 三角色协作工作流自动化脚本
监控 hook 文件变化，自动拉起对应 Agent

工作流：手动拉起分析 → 质检 → 改代码 → 训练 → 质检 → 预测 → 自动拉起分析
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# 使用相对路径
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

HOOK_DIR = Path("./hook")
STEERING_DIR = Path("./.kiro/steering")

# 确保 hook 目录存在
HOOK_DIR.mkdir(exist_ok=True)

# ============================================================
# Hook 文件 → Agent 映射
# ============================================================
HOOK_TRIGGERS = {
    # 方案相关
    "solution_pass.txt": {
        "agent": "executor",
        "steering": "agent-executor.md",
        "prompt": "方案评审已通过。请以方案执行者身份，按照agent-executor.md中的工作流程：1) 深度理解方案文档 2) 实现代码修改 3) 使用getDiagnostics验证代码 4) Git提交并推送 5) 执行训练 python PRISM.py（训练完成后脚本会自动创建hook/train.txt触发质检者）"
    },
    "solution_reject.txt": {
        "agent": "analyst",
        "steering": "agent-analyst.md",
        "prompt": "方案评审未通过。请以算法分析师身份，阅读质检者的评审报告，理解方案被拒绝的原因，然后重新设计一个更优的方案。请确保新方案解决了质检者指出的问题，并输出到 docx/记录点(n+1)/记录点(n+1)方案.md"
    },
    # 训练相关
    "train.txt": {
        "agent": "inspector",
        "steering": "agent-inspector.md",
        "prompt": "训练已完成（./hook/train.txt 已创建）。请以质检者身份，按照agent-inspector.md中的检查点B流程评估训练结果：1) 读取 save_model/baseline/log/ 最新日志 2) 分析收敛性、过拟合、数值稳定性 3) 做出决策并写入对应hook文件触发下一步：正常→写入hook/train_pass.txt，过拟合/退化→写入hook/train_redesign.txt，技术异常→写入hook/train_fix.txt"
    },
    "train_pass.txt": {
        "agent": "executor",
        "steering": "agent-executor.md",
        "prompt": "训练评审已通过。请以方案执行者身份，执行预测流程：运行 python predict.py（预测完成后脚本会自动创建hook/predict.txt触发算法分析师）"
    },
    "train_fix.txt": {
        "agent": "executor",
        "steering": "agent-executor.md",
        "prompt": "训练评审显示存在技术异常（如NaN、收敛失败等）。请以方案执行者身份，阅读质检者的训练评审报告，定位并修复技术问题，然后重新执行训练：1) 修复代码问题 2) 使用getDiagnostics验证 3) Git提交 4) 重新运行 python PRISM.py（训练完成后脚本会自动创建hook/train.txt触发质检者）"
    },
    "train_redesign.txt": {
        "agent": "analyst",
        "steering": "agent-analyst.md",
        "prompt": "训练评审显示存在过拟合或性能退化问题。请以算法分析师身份，阅读质检者的训练评审报告和训练日志，深度分析失败原因，撰写当前记录点的反思文档，然后设计新的方案。输出：1) docx/记录点(n)/记录点(n)反思.md 2) docx/记录点(n+1)/记录点(n+1)方案.md 3) 更新 docx/历史索引.md"
    },
    # 预测相关
    "predict.txt": {
        "agent": "analyst",
        "steering": "agent-analyst.md",
        "prompt": "预测已完成（./hook/predict.txt 已创建）。请以算法分析师身份，按照agent-analyst.md中的工作流程：1) 读取 compete/baseline/ 目录下的预测结果 2) 誊抄结果到 docx/记录点(n)/记录点(n)结果.md 3) 判断是否达标(AUPR≥0.75)：达标→项目成功并结束，不达标→撰写反思文档 docx/记录点(n)/记录点(n)反思.md 并设计新方案 docx/记录点(n+1)/记录点(n+1)方案.md 4) 更新 docx/历史索引.md。注意：新方案文件创建后会自动触发质检者评估。"
    },
}

# 方案文档监控（特殊处理）
SOLUTION_PATTERN = "./docx/记录点*/记录点*方案*"


def read_steering(steering_file: str) -> str:
    """读取 steering 文件内容"""
    path = STEERING_DIR / steering_file
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def invoke_claude(prompt: str, steering_file: str = None):
    """调用 Claude Agent"""
    # 构建完整提示词
    full_prompt = prompt
    
    # 如果有 steering 文件，先让 Claude 阅读
    if steering_file:
        steering_content = read_steering(steering_file)
        if steering_content:
            full_prompt = f"请先阅读以下 Agent 指南：\n\n{steering_content}\n\n---\n\n{prompt}"
    
    # 调用 claude
    cmd = ["claude", "-p", full_prompt, "--dangerously-skip-permissions"]
    print(f"\n{'='*60}")
    print(f"[WORKFLOW] 调用 Claude Agent")
    print(f"{'='*60}\n")
    
    subprocess.run(cmd)


def check_new_solution_files(known_files: set) -> list:
    """检查是否有新的方案文件创建"""
    import glob
    current_files = set(glob.glob(SOLUTION_PATTERN))
    new_files = current_files - known_files
    return list(new_files), current_files


def clear_hook(hook_file: str):
    """清除已处理的 hook 文件"""
    path = HOOK_DIR / hook_file
    if path.exists():
        path.unlink()
        print(f"[HOOK] 已清除: {hook_file}")


def check_hooks() -> tuple:
    """检查是否有 hook 文件被创建，返回 (hook_file, config)"""
    for hook_file, config in HOOK_TRIGGERS.items():
        path = HOOK_DIR / hook_file
        if path.exists():
            return hook_file, config
    return None, None


def run_initial_analyst():
    """运行初始的算法分析师（手动拉起）"""
    steering_content = read_steering("agent-analyst.md")
    
    initial_prompt = f"""请先阅读以下 Agent 指南：

{steering_content}

---

你是一位拥有深邃数学直觉的算法分析师。你不修补表象——你揭示本质。

1. 自从记录点4引入解耦组件后，EP互作AUPR从cross attn baseline的65提升到70(测试集)后，我们认识到新的瓶颈。AUPR70只是一个平凡解，我们期望模型面对OOD细胞系能得到更具鲁棒性的结果，于是我们做了记录点7-记录点13的一系列实验。不知道是代码落实存在问题抑或方案本身存在缺陷，模型的效果不增反降

2. 当前代码已回退至记录点6

3. 现在请你按照算法分析师的要求进行工作，揭示问题本质，提出更更更高价值的方案。think harder and harder

完成方案设计后，请将方案输出到 docx/记录点(n+1)/记录点(n+1)方案.md"""

    invoke_claude(initial_prompt)


def run_solution_inspector():
    """运行方案质检"""
    config = {
        "steering": "agent-inspector.md",
        "prompt": "检测到新方案文档已创建。请以质检者身份，按照agent-inspector.md中的检查点A流程评估方案：1) 读取新创建的方案文档 2) 进行理论深度、创新性、可行性、风险可控性、预期收益的多维评分 3) 做出决策并写入对应hook文件触发下一步：通过→写入hook/solution_pass.txt，不通过→写入hook/solution_reject.txt"
    }
    invoke_claude(config["prompt"], config["steering"])


def main_loop():
    """主循环：监控 hook 文件变化"""
    import glob
    
    print("\n" + "="*60)
    print("PRISM 三角色协作工作流 - 自动化监控启动")
    print("="*60)
    print("\n监控中... (Ctrl+C 退出)\n")
    
    # 记录已知的方案文件
    known_solution_files = set(glob.glob(SOLUTION_PATTERN))
    
    while True:
        try:
            # 1. 检查是否有新的方案文件（触发质检者）
            new_solutions, known_solution_files = check_new_solution_files(known_solution_files)
            if new_solutions:
                print(f"[HOOK] 检测到新方案文件: {new_solutions}")
                run_solution_inspector()
                continue
            
            # 2. 检查其他 hook 文件
            hook_file, config = check_hooks()
            if hook_file:
                print(f"[HOOK] 检测到: {hook_file}")
                clear_hook(hook_file)
                invoke_claude(config["prompt"], config["steering"])
                continue
            
            # 等待一段时间再检查
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\n[WORKFLOW] 监控已停止")
            break


def main():
    """主入口
    
    工作流：
    1. 手动拉起算法分析师（仅首次，使用特定的初始提示词）
    2. 算法分析师设计方案 → 创建方案文件
    3. 方案文件创建 → 自动触发质检者评估
    4. 质检通过 → 触发方案执行者实现+训练
    5. 训练完成 → 触发质检者评估训练
    6. 训练通过 → 触发方案执行者预测
    7. 预测完成 → 触发算法分析师分析结果
    8. 如果不达标 → 算法分析师设计新方案 → 回到步骤3
    9. 如果达标 → 项目成功结束
    """
    print("\n" + "="*60)
    print("PRISM 三角色协作工作流")
    print("="*60)
    print("\n[STEP 1] 启动算法分析师进行问题诊断和方案设计...\n")
    
    # 第一步：手动拉起算法分析师（仅首次使用特定初始提示词）
    run_initial_analyst()
    
    # 第二步：进入自动监控循环
    # 后续的算法分析师调用（如 predict.txt 触发）会使用标准流程提示词
    print("\n[STEP 2] 进入自动监控模式...\n")
    main_loop()


if __name__ == "__main__":
    main()
