#!/usr/bin/env python3
"""
PRISM 三角色协作工作流自动化脚本

工作流：
  分析师设计方案 → 质检评审 → 编码Agent改代码 → 脚本运行训练 → 质检评审 → 脚本运行预测 → 分析师分析结果

核心设计：
1. 编码Agent只负责写代码，不运行 PRISM.py/predict.py
2. 训练/预测由脚本自动执行，出错直接返修给编码Agent
3. Agent超时检测（15分钟），超时则报告异常并重新拉起
4. 每次调用 claude 都是独立新会话（--print 模式）
"""

import os
import sys
import time
import glob
import subprocess
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple

# 使用相对路径
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

# ============================================================
# 常量配置
# ============================================================
MAX_ATTEMPTS = 50         # 最大尝试次数
MAX_ITERATIONS = 20       # 最大迭代轮数
ERROR_LINE_LIMIT = 20     # 错误信息显示的行数限制
AGENT_TIMEOUT_SECONDS = 60 * 60  # Agent 超时时间：15分钟
HOOK_DIR = Path("./hook")
STEERING_DIR = Path("./.kiro/steering")
SOLUTION_PATTERN = "./docx/记录点*/记录点*方案*"
ANA_INIT_BOOL = True  # 分析师是否进行初始化

# 确保 hook 目录存在
HOOK_DIR.mkdir(exist_ok=True)

# ============================================================
# 环境变量初始化（DeepSeek API 配置）
# ============================================================
def init_environment():
    """初始化 DeepSeek API 环境变量"""
    env_vars = {
        "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
        "DEEPSEEK_API_KEY": "sk-1fb9049001b14e7cb42a92c18c5cb329",
        "ANTHROPIC_AUTH_TOKEN": "sk-1fb9049001b14e7cb42a92c18c5cb329",
        "API_TIMEOUT_MS": "6000000",
        "ANTHROPIC_MODEL": "deepseek-reasoner",
        "ANTHROPIC_SMALL_FAST_MODEL": "deepseek-chat",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "IS_SANDBOX": "1",
    }
    for key, value in env_vars.items():
        os.environ[key] = value
    print("[ENV] DeepSeek API 环境变量已初始化")

init_environment()

# ============================================================
# 模式切换
# ============================================================
class ModelMode(Enum):
    CHAT = "chat"           # 质检用
    REASONER = "reasoner"   # 分析和编码用

AGENT_MODE_MAP = {
    "inspector": ModelMode.CHAT,
    "analyst": ModelMode.REASONER,
    "executor": ModelMode.CHAT,
}

def set_model_mode(mode: ModelMode):
    if mode == ModelMode.CHAT:
        os.environ["ANTHROPIC_MODEL"] = "deepseek-chat"
        os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = "deepseek-chat"
        print(f"[MODE] CHAT 模式 (deepseek-chat)")
    else:
        os.environ["ANTHROPIC_MODEL"] = "deepseek-reasoner"
        os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = "deepseek-reasoner"
        print(f"[MODE] REASONER 模式 (deepseek-reasoner)")

# ============================================================
# Agent 调用结果
# ============================================================
class AgentResult(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"

# ============================================================
# 工具函数
# ============================================================
def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def read_steering(steering_file: str) -> str:
    path = STEERING_DIR / steering_file
    return path.read_text(encoding="utf-8") if path.exists() else ""

def clear_hook(hook_file: str):
    path = HOOK_DIR / hook_file
    if path.exists():
        path.unlink()
        print(f"[HOOK] 已清除: {hook_file}")

def check_hook_exists(hook_file: str) -> bool:
    return (HOOK_DIR / hook_file).exists()

def create_hook(hook_file: str, content: str = "done"):
    (HOOK_DIR / hook_file).write_text(content)
    print(f"[HOOK] 已创建: {hook_file}")

def check_new_solution_files(known_files: set) -> Tuple[list, set]:
    current_files = set(glob.glob(SOLUTION_PATTERN))
    new_files = current_files - known_files
    return list(new_files), current_files


# ============================================================
# Agent 调用（带超时检测）
# ============================================================
def invoke_claude(prompt: str, steering_file: str = None, agent_type: str = None, 
                  timeout: int = AGENT_TIMEOUT_SECONDS) -> Tuple[AgentResult, str]:
    """
    调用 Claude Agent，带超时检测
    
    Returns:
        (AgentResult, output): 结果状态和输出内容
    """
    if agent_type and agent_type in AGENT_MODE_MAP:
        set_model_mode(AGENT_MODE_MAP[agent_type])
    
    full_prompt = prompt
    if steering_file:
        steering_content = read_steering(steering_file)
        if steering_content:
            full_prompt = f"请先阅读以下 Agent 指南：\n\n{steering_content}\n\n---\n\n{prompt}"
    
    cmd = ["claude", "--print", "--dangerously-skip-permissions", full_prompt]
    
    current_model = os.environ.get("ANTHROPIC_MODEL", "unknown")
    print(f"\n{'='*60}")
    print(f"[{ts()}] 调用 Agent: {agent_type or 'initial'} | 模型: {current_model}")
    print(f"[{ts()}] 超时设置: {timeout // 60} 分钟")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    last_status_time = start_time
    output_lines = []
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        
        while True:
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                process.terminate()
                process.wait(timeout=5)
                print(f"\n[{ts()}] ⚠️ Agent 超时 ({timeout // 60} 分钟)，已终止")
                return AgentResult.TIMEOUT, "\n".join(output_lines)
            
            # 读取输出（非阻塞检查）
            line = process.stdout.readline()
            if line:
                print(line, end='', flush=True)
                output_lines.append(line.rstrip())
            elif process.poll() is not None:
                break
            
            # 每 60 秒输出状态
            current_time = time.time()
            if current_time - last_status_time >= 60:
                minutes, seconds = divmod(int(elapsed), 60)
                print(f"\n[{ts()}] ⏱️ Agent 运行中 ({minutes}m{seconds}s) | 输出 {len(output_lines)} 行\n", flush=True)
                last_status_time = current_time
        
        # 读取剩余输出
        remaining = process.stdout.read()
        if remaining:
            print(remaining, end='', flush=True)
            output_lines.extend(remaining.rstrip().split('\n'))
        
        return_code = process.wait()
        elapsed = int(time.time() - start_time)
        minutes, seconds = divmod(elapsed, 60)
        
        print(f"\n[{ts()}] Agent 完成 (返回码: {return_code}, 耗时: {minutes}m{seconds}s)")
        print(f"{'='*60}\n")
        
        if return_code != 0:
            return AgentResult.ERROR, "\n".join(output_lines)
        return AgentResult.SUCCESS, "\n".join(output_lines)
        
    except FileNotFoundError:
        print("[ERROR] claude 命令未找到")
        return AgentResult.ERROR, "claude command not found"
    except Exception as e:
        print(f"[ERROR] Agent 调用失败: {e}")
        return AgentResult.ERROR, str(e)

# ============================================================
# 脚本执行训练/预测（无超时限制）
# ============================================================
def run_python_script(script_name: str) -> Tuple[bool, str]:
    """
    运行 Python 脚本（PRISM.py 或 predict.py）
    不设超时，因为训练/预测本身耗时较长
    
    Returns:
        (success, output): 是否成功和输出内容
    """
    print(f"\n{'='*60}")
    print(f"[{ts()}] 🚀 开始执行: python {script_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    last_status_time = start_time
    output_lines = []
    
    try:
        process = subprocess.Popen(
            ["python", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end='', flush=True)
                output_lines.append(line.rstrip())
            elif process.poll() is not None:
                break
            
            # 每 5 分钟输出状态
            current_time = time.time()
            if current_time - last_status_time >= 300:
                elapsed = int(current_time - start_time)
                minutes, seconds = divmod(elapsed, 60)
                print(f"\n[{ts()}] ⏱️ {script_name} 运行中 ({minutes}m{seconds}s)\n", flush=True)
                last_status_time = current_time
        
        remaining = process.stdout.read()
        if remaining:
            print(remaining, end='', flush=True)
            output_lines.extend(remaining.rstrip().split('\n'))
        
        return_code = process.wait()
        elapsed = int(time.time() - start_time)
        minutes, seconds = divmod(elapsed, 60)
        
        output = "\n".join(output_lines)
        
        if return_code != 0:
            print(f"\n[{ts()}] ❌ {script_name} 执行失败 (返回码: {return_code}, 耗时: {minutes}m{seconds}s)")
            return False, output
        
        print(f"\n[{ts()}] ✅ {script_name} 执行成功 (耗时: {minutes}m{seconds}s)")
        return True, output
        
    except Exception as e:
        print(f"[ERROR] {script_name} 执行异常: {e}")
        return False, str(e)


# ============================================================
# 各角色 Agent 调用
# ============================================================

def run_analyst(prompt: str) -> AgentResult:
    """运行算法分析师"""
    result, _ = invoke_claude(prompt, "agent-analyst.md", "analyst")
    return result

def run_inspector(prompt: str) -> AgentResult:
    """运行质检者"""
    result, _ = invoke_claude(prompt, "agent-inspector.md", "inspector")
    return result

def run_executor_code_only(error_info: str = None) -> AgentResult:
    """
    运行编码Agent - 只负责写代码，不运行训练/预测
    """
    if error_info:
        prompt = f"""方案评审已通过，但上次代码执行出错。请以方案执行者身份修复代码：

【错误信息】
{error_info}

请：
1) 分析错误原因
2) 修复代码问题
3) 使用 getDiagnostics 验证代码无语法错误
4) Git 提交修改

⚠️ 注意：你只需要修改代码，训练会由脚本自动执行。不要运行 python PRISM.py"""
    else:
        prompt = """方案评审已通过。请以方案执行者身份实现代码：
p.s. 记得在`config.py`中设置对应的`SAVEMODEL_NAME`避免覆盖之前的模型
1) 深度理解方案文档 (docx/记录点*/记录点*方案.md)
2) 实现代码修改
3) 使用 getDiagnostics 验证代码无语法错误
4) Git 提交并推送

⚠️ 注意：你只需要修改代码，训练会由脚本自动执行。不要运行 python PRISM.py"""
    
    result, _ = invoke_claude(prompt, "agent-executor.md", "executor")
    return result

def run_executor_fix_predict(error_info: str) -> AgentResult:
    """编码Agent修复预测错误"""
    prompt = f"""预测执行出错，请修复代码：

【错误信息】
{error_info}

请：
1) 分析错误原因
2) 修复 predict.py 或相关代码
3) 使用 getDiagnostics 验证
4) Git 提交

⚠️ 注意：你只需要修改代码，预测会由脚本自动执行。"""
    
    result, _ = invoke_claude(prompt, "agent-executor.md", "executor")
    return result

def run_initial_analyst() -> AgentResult:
    # 初始化算法分析师
    if ANA_INIT_BOOL==True:
        prompt = """你是一位拥有深邃数学直觉的算法分析师。你不修补表象——你揭示本质。

    1. 自从记录点4引入解耦组件后，EP互作AUPR从cross attn baseline的65提升到70(测试集)后，我们认识到新的瓶颈。AUPR70只是一个平凡解，我们期望模型面对OOD细胞系能得到更具鲁棒性的结果，于是我们做了记录点7-记录点13的一系列实验。不知道是代码落实存在问题抑或方案本身存在缺陷，模型的效果不增反降

    2. 当前代码已回退至记录点6

    3. 现在请你按照算法分析师的要求进行工作，揭示问题本质，提出更更更高价值的方案。think harder and harder

    完成方案设计后，请将方案输出到 docx/记录点(n+1)/记录点(n+1)方案.md"""
    else:
        prompt = """请以算法分析师身份：阅读`.kiro/steering/agent-analyst.md`,阅读`.kiro/steering/structure.md`理解项目背景和难点"""
    
    return run_analyst(prompt)

def run_solution_inspector() -> Tuple[AgentResult, bool]:
    """
    运行方案质检
    Returns:
        (AgentResult, passed): Agent结果 和 方案是否通过
    """
    prompt = """检测到新方案文档已创建。请以质检者身份评估方案：

1) 读取新创建的方案文档
2) 进行理论深度、创新性、可行性、风险可控性、预期收益的多维评分
3) 做出决策并写入对应hook文件：
   - 通过 → echo "pass" > ./hook/solution_pass.txt
   - 不通过 → echo "reject" > ./hook/solution_reject.txt

⚠️ 必须创建hook文件，否则工作流无法继续！"""
    
    result, _ = invoke_claude(prompt, "agent-inspector.md", "inspector")
    
    # 检查质检结果
    if result == AgentResult.SUCCESS:
        if check_hook_exists("solution_pass.txt"):
            clear_hook("solution_pass.txt")
            return result, True
        elif check_hook_exists("solution_reject.txt"):
            clear_hook("solution_reject.txt")
            return result, False
        else:
            print(f"[{ts()}] ⚠️ 质检者未创建hook文件，视为超时")
            return AgentResult.TIMEOUT, False
    
    return result, False

def run_train_inspector() -> Tuple[AgentResult, str]:
    """
    运行训练质检
    Returns:
        (AgentResult, decision): Agent结果 和 决策(pass/fix/redesign)
    """
    prompt = """训练已完成。请以质检者身份评估训练结果：

1) 读取 save_model/baseline/log/ 最新日志
2) 分析收敛性、过拟合、数值稳定性
3) 做出决策并写入对应hook文件：
   - 正常 → echo "pass" > ./hook/train_pass.txt
   - 技术异常(NaN等) → echo "fix" > ./hook/train_fix.txt
   - 过拟合/退化 → echo "redesign" > ./hook/train_redesign.txt

⚠️ 必须创建hook文件，否则工作流无法继续！"""
    
    result, _ = invoke_claude(prompt, "agent-inspector.md", "inspector")
    
    if result == AgentResult.SUCCESS:
        if check_hook_exists("train_pass.txt"):
            clear_hook("train_pass.txt")
            return result, "pass"
        elif check_hook_exists("train_fix.txt"):
            clear_hook("train_fix.txt")
            return result, "fix"
        elif check_hook_exists("train_redesign.txt"):
            clear_hook("train_redesign.txt")
            return result, "redesign"
        else:
            print(f"[{ts()}] ⚠️ 质检者未创建hook文件，视为超时")
            return AgentResult.TIMEOUT, ""
    
    return result, ""


# ============================================================
# 主工作流
# ============================================================

def workflow_design_phase(known_solution_files: set) -> Tuple[bool, set]:
    """
    设计阶段：分析师设计方案 → 质检评审
    
    Returns:
        (passed, updated_known_files): 方案是否通过，更新后的已知方案文件集合
    """
    max_retries = MAX_ATTEMPTS
    
    for attempt in range(max_retries):
        # 检查是否有新方案
        new_solutions, known_solution_files = check_new_solution_files(known_solution_files)
        
        if not new_solutions:
            print(f"[{ts()}] 等待分析师创建方案...")
            time.sleep(5)
            continue
        
        print(f"[{ts()}] 检测到新方案: {new_solutions}")
        
        # 质检评审
        result, passed = run_solution_inspector()
        
        if result == AgentResult.TIMEOUT:
            print(f"[{ts()}] ⚠️ 质检超时，重新拉起质检者 (尝试 {attempt + 1}/{max_retries})")
            continue
        
        if result == AgentResult.ERROR:
            print(f"[{ts()}] ❌ 质检出错，重新拉起质检者 (尝试 {attempt + 1}/{max_retries})")
            continue
        
        if passed:
            print(f"[{ts()}] ✅ 方案评审通过")
            return True, known_solution_files
        else:
            print(f"[{ts()}] ❌ 方案评审未通过，需要分析师重新设计")
            # 拉起分析师重新设计
            analyst_result = run_analyst("方案评审未通过。请阅读质检报告，重新设计方案。")
            if analyst_result == AgentResult.TIMEOUT:
                print(f"[{ts()}] ⚠️ 分析师超时")
            # 继续循环检查新方案
    
    print(f"[{ts()}] ❌ 设计阶段失败，超过最大重试次数")
    return False, known_solution_files


def workflow_coding_and_training() -> Tuple[bool, str]:
    """
    编码+训练阶段：编码Agent改代码 → 脚本运行训练
    
    Returns:
        (success, error_info): 是否成功，错误信息
    """
    max_code_retries = MAX_ATTEMPTS
    error_info = None
    
    for attempt in range(max_code_retries):
        # 编码Agent写代码
        print(f"\n[{ts()}] 📝 编码阶段 (尝试 {attempt + 1}/{max_code_retries})")
        
        executor_result = run_executor_code_only(error_info)
        
        if executor_result == AgentResult.TIMEOUT:
            print(f"[{ts()}] ⚠️ 编码Agent超时，重新拉起")
            continue
        
        if executor_result == AgentResult.ERROR:
            print(f"[{ts()}] ❌ 编码Agent出错，重新拉起")
            continue
        
        # 脚本自动运行训练
        print(f"\n[{ts()}] 🏋️ 开始训练...")
        success, output = run_python_script("PRISM.py")
        
        if success:
            print(f"[{ts()}] ✅ 训练完成")
            return True, ""
        else:
            print(f"[{ts()}] ❌ 训练失败，返修给编码Agent")
            error_lines = output.split('\n')[-ERROR_LINE_LIMIT:]
            error_info = "\n".join(error_lines)
    
    print(f"[{ts()}] ❌ 编码+训练阶段失败，超过最大重试次数")
    return False, error_info


def workflow_train_review() -> str:
    """
    训练质检阶段
    
    Returns:
        decision: "pass" / "fix" / "redesign" / "timeout"
    """
    max_retries = MAX_ATTEMPTS
    
    for attempt in range(max_retries):
        result, decision = run_train_inspector()
        
        if result == AgentResult.TIMEOUT:
            print(f"[{ts()}] ⚠️ 训练质检超时，重新拉起 (尝试 {attempt + 1}/{max_retries})")
            continue
        
        if result == AgentResult.ERROR:
            print(f"[{ts()}] ❌ 训练质检出错，重新拉起 (尝试 {attempt + 1}/{max_retries})")
            continue
        
        return decision
    
    return "timeout"


def workflow_prediction() -> Tuple[bool, str]:
    """
    预测阶段：脚本运行预测
    
    Returns:
        (success, error_info): 是否成功，错误信息
    """
    max_retries = MAX_ATTEMPTS
    error_info = None
    
    for attempt in range(max_retries):
        if error_info:
            # 有错误，先让编码Agent修复
            print(f"\n[{ts()}] 🔧 修复预测代码 (尝试 {attempt + 1}/{max_retries})")
            executor_result = run_executor_fix_predict(error_info)
            
            if executor_result in (AgentResult.TIMEOUT, AgentResult.ERROR):
                print(f"[{ts()}] ⚠️ 编码Agent异常，重试")
                continue
        
        # 运行预测
        print(f"\n[{ts()}] 🔮 开始预测...")
        success, output = run_python_script("predict.py")
        
        if success:
            print(f"[{ts()}] ✅ 预测完成")
            return True, ""
        else:
            print(f"[{ts()}] ❌ 预测失败，返修给编码Agent")
            error_lines = output.split('\n')[-ERROR_LINE_LIMIT:]
            error_info = "\n".join(error_lines)
    
    print(f"[{ts()}] ❌ 预测阶段失败，超过最大重试次数")
    return False, error_info


def workflow_result_analysis(known_solution_files: set) -> Tuple[bool, set]:
    """
    结果分析阶段：分析师分析预测结果
    
    Returns:
        (target_reached, updated_known_files): 是否达标，更新后的已知方案文件
    """
    prompt = """预测已完成。请以算法分析师身份分析结果：

1) 读取 compete/baseline/ 目录下的预测结果
2) 誊抄结果到 docx/记录点(n)/记录点(n)结果.md
3) 判断是否达标 (AUPR ≥ 0.75)：
   - 达标 → 项目成功，更新历史索引标记完成
   - 不达标 → 撰写反思文档 + 设计新方案
4) 更新 docx/历史索引.md

如果不达标，新方案会自动触发下一轮迭代。"""
    
    result = run_analyst(prompt)
    
    if result == AgentResult.TIMEOUT:
        print(f"[{ts()}] ⚠️ 分析师超时")
        return False, known_solution_files
    
    # 检查是否有新方案（说明不达标，需要继续迭代）
    new_solutions, current_files = check_new_solution_files(known_solution_files)
    
    if new_solutions:
        print(f"[{ts()}] 📄 检测到新方案，AUPR未达标，继续迭代")
        # 注意：不更新 known_solution_files，让下一轮 workflow_design_phase 能检测到这个新方案
        return False, known_solution_files
    else:
        print(f"[{ts()}] 🎉 未检测到新方案，可能已达标或分析师未完成")
        # 这里可以进一步检查历史索引确认是否达标
        return True, known_solution_files


# ============================================================
# 主循环
# ============================================================

def main():
    """主入口 - 完整工作流"""
    print("\n" + "="*60)
    print("PRISM 三角色协作工作流")
    print("="*60)
    print(f"""
工作流设计：
  1. 分析师设计方案 → 2. 质检评审 → 3. 编码Agent改代码
  → 4. 脚本运行训练 → 5. 质检评审训练 → 6. 脚本运行预测
  → 7. 分析师分析结果 → (不达标则循环)

关键改进：
  - 编码Agent只写代码，不运行训练/预测
  - 训练/预测由脚本自动执行
  - Agent超时检测（15分钟）
  - 出错自动返修给编码Agent
""")
    
    known_solution_files = set(glob.glob(SOLUTION_PATTERN))
    max_iterations = MAX_ITERATIONS  # 最大迭代轮数
    
    # Step 1: 初始分析师设计方案
    print(f"\n[{ts()}] 🚀 Step 1: 启动算法分析师...")
    analyst_result = run_initial_analyst()
    
    if analyst_result == AgentResult.TIMEOUT:
        print(f"[{ts()}] ⚠️ 初始分析师超时，请检查问题后重新运行")
        return
    
    # 主迭代循环
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"[{ts()}] 🔄 迭代轮次: {iteration}/{max_iterations}")
        print(f"{'='*60}")
        
        # Step 2: 设计阶段（方案质检）
        print(f"\n[{ts()}] 📋 Step 2: 方案质检...")
        passed, known_solution_files = workflow_design_phase(known_solution_files)
        
        if not passed:
            print(f"[{ts()}] ❌ 设计阶段失败，终止")
            break
        
        # Step 3: 编码+训练阶段
        print(f"\n[{ts()}] 💻 Step 3: 编码+训练...")
        success, error_info = workflow_coding_and_training()
        
        if not success:
            print(f"[{ts()}] ❌ 编码+训练阶段失败，终止")
            break
        
        # Step 4: 训练质检
        print(f"\n[{ts()}] 🔍 Step 4: 训练质检...")
        decision = workflow_train_review()
        
        if decision == "pass":
            print(f"[{ts()}] ✅ 训练质检通过，进入预测阶段")
        elif decision == "fix":
            print(f"[{ts()}] 🔧 需要修复技术问题，重新训练")
            # 回到编码+训练阶段
            success, _ = workflow_coding_and_training()
            if not success:
                print(f"[{ts()}] ❌ 修复后训练仍失败，终止")
                break
            # 重新质检
            decision = workflow_train_review()
            if decision != "pass":
                print(f"[{ts()}] ❌ 修复后质检仍未通过，终止")
                break
        elif decision == "redesign":
            print(f"[{ts()}] 🔄 需要重新设计方案")
            # 拉起分析师重新设计
            run_analyst("训练评审显示过拟合或性能退化。请分析原因并重新设计方案。")
            continue  # 回到设计阶段
        else:
            print(f"[{ts()}] ❌ 训练质检异常，终止")
            break
        
        # Step 5: 预测阶段
        print(f"\n[{ts()}] 🔮 Step 5: 预测...")
        success, error_info = workflow_prediction()
        
        if not success:
            print(f"[{ts()}] ❌ 预测阶段失败，终止")
            break
        
        # Step 6: 结果分析
        print(f"\n[{ts()}] 📊 Step 6: 结果分析...")
        target_reached, known_solution_files = workflow_result_analysis(known_solution_files)
        
        if target_reached:
            print(f"\n{'='*60}")
            print(f"[{ts()}] 🎉 项目成功！AUPR ≥ 0.75 目标达成！")
            print(f"{'='*60}")
            break
        else:
            print(f"[{ts()}] 📈 AUPR未达标，继续下一轮迭代...")
    
    else:
        print(f"\n[{ts()}] ⚠️ 达到最大迭代次数 ({max_iterations})，工作流结束")
    
    print(f"\n[{ts()}] 工作流结束")


if __name__ == "__main__":
    main()
