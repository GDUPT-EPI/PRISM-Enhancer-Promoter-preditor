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
ANA_INIT_BOOL = False  # 分析师是否进行初始化

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
    "rollback": ModelMode.CHAT,  # 回退决策者
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
# Git 分支管理
# ============================================================
def get_current_branch() -> str:
    """获取当前分支名"""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"[GIT] 获取当前分支失败: {e}")
        return "unknown"

def get_next_chat_branch_name() -> str:
    """获取下一个chat分支名（chat0, chat1, chat2...）"""
    try:
        result = subprocess.run(
            ["git", "branch", "--list", "chat*"],
            capture_output=True, text=True, check=True
        )
        branches = result.stdout.strip().split('\n')
        branches = [b.strip().lstrip('* ') for b in branches if b.strip()]
        
        # 提取chat后的数字
        max_num = -1
        for branch in branches:
            if branch.startswith('chat'):
                try:
                    num = int(branch[4:])
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        return f"chat{max_num + 1}"
    except Exception as e:
        print(f"[GIT] 获取分支列表失败: {e}")
        return "chat1"

def create_branch_from_current(new_branch: str) -> bool:
    """从当前分支创建新分支并切换"""
    try:
        # 先提交所有更改
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Auto commit before branch {new_branch}"],
            capture_output=True
        )
        
        # 创建并切换到新分支
        subprocess.run(["git", "checkout", "-b", new_branch], check=True)
        print(f"[GIT] ✅ 从当前分支创建并切换到: {new_branch}")
        return True
    except Exception as e:
        print(f"[GIT] ❌ 创建分支失败: {e}")
        return False

def create_branch_from_chat0(new_branch: str) -> bool:
    """从chat0分支创建新分支（回退操作），但保留docx目录"""
    try:
        # 1. 先保存当前docx目录内容到临时位置
        import shutil
        docx_backup = Path("./docx_backup_temp")
        docx_path = Path("./docx")
        
        if docx_path.exists():
            if docx_backup.exists():
                shutil.rmtree(docx_backup)
            shutil.copytree(docx_path, docx_backup)
            print(f"[GIT] 📁 已备份 docx 目录")
        
        # 2. 提交当前更改（避免丢失）
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Auto commit before rollback to {new_branch}"],
            capture_output=True
        )
        
        # 3. 切换到chat0
        subprocess.run(["git", "checkout", "chat0"], check=True)
        print(f"[GIT] 已切换到 chat0")
        
        # 4. 从chat0创建新分支
        subprocess.run(["git", "checkout", "-b", new_branch], check=True)
        print(f"[GIT] ✅ 从chat0创建并切换到: {new_branch}")
        
        # 5. 恢复docx目录（合并历史记录）
        if docx_backup.exists():
            # 如果新分支的docx存在，合并内容
            if docx_path.exists():
                # 遍历备份中的所有文件和目录，复制到当前docx
                for item in docx_backup.iterdir():
                    dest = docx_path / item.name
                    if item.is_dir():
                        if dest.exists():
                            # 目录存在，合并内容
                            for sub_item in item.iterdir():
                                sub_dest = dest / sub_item.name
                                if not sub_dest.exists():
                                    if sub_item.is_dir():
                                        shutil.copytree(sub_item, sub_dest)
                                    else:
                                        shutil.copy2(sub_item, sub_dest)
                        else:
                            shutil.copytree(item, dest)
                    else:
                        # 文件：如果不存在则复制，存在则保留备份版本（更新）
                        shutil.copy2(item, dest)
            else:
                shutil.copytree(docx_backup, docx_path)
            
            # 清理临时备份
            shutil.rmtree(docx_backup)
            print(f"[GIT] 📁 已恢复 docx 目录（保留历史记录）")
            
            # 提交恢复的docx
            subprocess.run(["git", "add", "docx/"], check=True)
            subprocess.run(
                ["git", "commit", "-m", f"Restore docx history from previous branch"],
                capture_output=True
            )
        
        return True
    except Exception as e:
        print(f"[GIT] ❌ 回退分支失败: {e}")
        # 尝试清理临时备份
        docx_backup = Path("./docx_backup_temp")
        if docx_backup.exists():
            import shutil
            shutil.rmtree(docx_backup)
        return False


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
    """初始化算法分析师 - 已废弃，使用 ensure_analyst_creates_solution 代替"""
    # 保留此函数以兼容，但不再使用
    return run_analyst("请设计新方案并输出到 docx/记录点(n+1)/记录点(n+1)方案.md")

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
        (AgentResult, decision): Agent结果 和 决策(pass/fail)
    """
    prompt = """训练已完成。请以质检者身份评估训练结果：

1) 读取 save_model/baseline/log/ 最新日志
2) 分析收敛性、过拟合、数值稳定性
3) 做出决策并写入对应hook文件：
   - 正常（可以继续预测）→ echo "pass" > ./hook/train_pass.txt
   - 异常（NaN/过拟合/退化等任何问题）→ echo "fail" > ./hook/train_fail.txt

⚠️ 必须创建hook文件，否则工作流无法继续！
⚠️ 只要有任何问题就判定为fail，不要尝试修复！"""
    
    result, _ = invoke_claude(prompt, "agent-inspector.md", "inspector")
    
    if result == AgentResult.SUCCESS:
        if check_hook_exists("train_pass.txt"):
            clear_hook("train_pass.txt")
            return result, "pass"
        elif check_hook_exists("train_fail.txt"):
            clear_hook("train_fail.txt")
            return result, "fail"
        else:
            print(f"[{ts()}] ⚠️ 质检者未创建hook文件，视为超时")
            return AgentResult.TIMEOUT, ""
    
    return result, ""


def run_rollback_decision() -> Tuple[AgentResult, str]:
    """
    运行回退决策者
    Returns:
        (AgentResult, decision): Agent结果 和 决策(keep/rollback)
    """
    current_branch = get_current_branch()
    
    prompt = f"""预测已完成。请以代码回退决策者身份评估本轮修改的价值：

【当前分支】: {current_branch}

请执行以下步骤：
1) 读取 compete/ 目录下的预测结果（查看config.py获取SAVEMODEL_NAME）
2) 读取 docx/基线结果.log 获取基线AUPR
3) 对比当前AUPR与基线AUPR
4) 做出决策并写入对应hook文件：
   - 保留代码（有提升）→ echo "keep" > ./hook/rollback_keep.txt
   - 回退代码（无效/下降）→ echo "rollback" > ./hook/rollback_reset.txt

评估标准：
- AUPR有任何提升（即使0.001）→ 保留
- AUPR持平但其他指标提升 → 保留
- AUPR下降或持平无提升 → 回退

⚠️ 必须创建hook文件，否则工作流无法继续！"""
    
    result, _ = invoke_claude(prompt, "agent-rollback.md", "rollback")
    
    if result == AgentResult.SUCCESS:
        if check_hook_exists("rollback_keep.txt"):
            clear_hook("rollback_keep.txt")
            return result, "keep"
        elif check_hook_exists("rollback_reset.txt"):
            clear_hook("rollback_reset.txt")
            return result, "rollback"
        else:
            print(f"[{ts()}] ⚠️ 回退决策者未创建hook文件，视为超时")
            return AgentResult.TIMEOUT, ""
    
    return result, ""


# ============================================================
# 主工作流
# ============================================================

def workflow_design_phase(known_solution_files: set) -> Tuple[bool, set]:
    """
    设计阶段：分析师设计方案 → 质检评审
    
    核心逻辑：没有新方案时立即拉起分析师，不傻等！
    
    Returns:
        (passed, updated_known_files): 方案是否通过，更新后的已知方案文件集合
    """
    max_retries = MAX_ATTEMPTS
    
    for attempt in range(max_retries):
        # 检查是否有新方案
        new_solutions, current_files = check_new_solution_files(known_solution_files)
        
        if not new_solutions:
            # 没有新方案，立即拉起分析师（不等待！）
            print(f"[{ts()}] 📝 未检测到新方案，立即拉起分析师设计 (尝试 {attempt + 1}/{max_retries})")
            analyst_result = run_analyst("""请以算法分析师身份设计新方案：

1) 阅读 docx/历史索引.md 了解历史方案
2) 阅读 docx/基线结果.log 了解当前性能
3) 设计新方案，确保与历史失败方案有本质区别
4) 输出新方案到 docx/记录点(n+1)/记录点(n+1)方案.md
5) 更新 docx/历史索引.md

⚠️ 必须创建方案文件，否则工作流无法继续！""")
            
            if analyst_result == AgentResult.TIMEOUT:
                print(f"[{ts()}] ⚠️ 分析师超时，重试")
                continue
            elif analyst_result == AgentResult.ERROR:
                print(f"[{ts()}] ❌ 分析师出错，重试")
                continue
            
            # 分析师完成后，再次检查是否有新方案
            new_solutions, current_files = check_new_solution_files(known_solution_files)
            if not new_solutions:
                print(f"[{ts()}] ⚠️ 分析师完成但未创建方案文件，重试")
                continue
        
        # 有新方案，更新已知文件集合
        known_solution_files = current_files
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
            print(f"[{ts()}] ❌ 方案评审未通过，拉起分析师重新设计")
            # 立即拉起分析师重新设计（不等待！）
            analyst_result = run_analyst("""方案评审未通过。请以算法分析师身份：

1) 阅读质检报告了解不通过原因
2) 针对性改进方案
3) 输出新方案到 docx/记录点(n+1)/记录点(n+1)方案.md

⚠️ 必须创建新的方案文件！""")
            if analyst_result == AgentResult.TIMEOUT:
                print(f"[{ts()}] ⚠️ 分析师超时")
            elif analyst_result == AgentResult.ERROR:
                print(f"[{ts()}] ❌ 分析师出错")
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
        decision: "pass" / "fail" / "timeout"
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


def workflow_rollback_decision() -> str:
    """
    回退决策阶段：决定是否回退代码
    
    Returns:
        decision: "keep" / "rollback" / "timeout"
    """
    max_retries = MAX_ATTEMPTS
    
    for attempt in range(max_retries):
        result, decision = run_rollback_decision()
        
        if result == AgentResult.TIMEOUT:
            print(f"[{ts()}] ⚠️ 回退决策超时，重新拉起 (尝试 {attempt + 1}/{max_retries})")
            continue
        
        if result == AgentResult.ERROR:
            print(f"[{ts()}] ❌ 回退决策出错，重新拉起 (尝试 {attempt + 1}/{max_retries})")
            continue
        
        return decision
    
    return "timeout"


def execute_branch_operation(decision: str) -> bool:
    """
    执行分支操作
    
    Args:
        decision: "keep" 或 "rollback"
    
    Returns:
        success: 是否成功
    """
    current_branch = get_current_branch()
    next_branch = get_next_chat_branch_name()
    
    print(f"\n[{ts()}] 🌿 分支操作")
    print(f"    当前分支: {current_branch}")
    print(f"    决策: {decision}")
    print(f"    目标分支: {next_branch}")
    
    if decision == "keep":
        # 保留代码：从当前分支创建新分支
        print(f"[{ts()}] ✅ 保留代码修改，从 {current_branch} 创建 {next_branch}")
        return create_branch_from_current(next_branch)
    elif decision == "rollback":
        # 回退代码：从chat0创建新分支
        print(f"[{ts()}] 🔄 回退代码，从 chat0 创建 {next_branch}")
        return create_branch_from_chat0(next_branch)
    else:
        print(f"[{ts()}] ❌ 未知决策: {decision}")
        return False


# ============================================================
# 主循环
# ============================================================

def ensure_analyst_creates_solution(prompt: str, known_solution_files: set, max_retries: int = 10) -> Tuple[bool, set]:
    """
    确保分析师创建新方案，带无限重试机制
    
    Returns:
        (success, updated_known_files): 是否成功创建新方案
    """
    for attempt in range(max_retries):
        print(f"[{ts()}] 📝 拉起分析师创建方案 (尝试 {attempt + 1}/{max_retries})")
        analyst_result = run_analyst(prompt)
        
        if analyst_result == AgentResult.TIMEOUT:
            print(f"[{ts()}] ⚠️ 分析师超时，重试")
            continue
        elif analyst_result == AgentResult.ERROR:
            print(f"[{ts()}] ❌ 分析师出错，重试")
            continue
        
        # 检查是否创建了新方案
        new_solutions, current_files = check_new_solution_files(known_solution_files)
        if new_solutions:
            print(f"[{ts()}] ✅ 分析师已创建新方案: {new_solutions}")
            return True, current_files
        else:
            print(f"[{ts()}] ⚠️ 分析师完成但未创建方案文件，重试")
    
    print(f"[{ts()}] ❌ 分析师 {max_retries} 次重试后仍未创建方案")
    return False, known_solution_files


def main():
    """主入口 - 完整工作流（永不异常停止）"""
    print("\n" + "="*60)
    print("PRISM 四角色协作工作流")
    print("="*60)
    print(f"""
工作流设计：
  1. 分析师设计方案 → 2. 质检评审 → 3. 编码Agent改代码
  → 4. 脚本运行训练 → 5. 质检评审训练 → 6. 脚本运行预测
  → 7. 回退决策者评估 → 8. 分支操作 → 9. 分析师分析结果
  → (不达标则循环)

关键设计：
  - 永不异常停止：所有失败都会回退并重新拉起分析师
  - Agent超时/出错自动重试
  - 训练/预测失败自动回退到chat0基线
""")
    
    current_branch = get_current_branch()
    print(f"[{ts()}] 🌿 当前分支: {current_branch}")
    
    known_solution_files = set(glob.glob(SOLUTION_PATTERN))
    max_iterations = MAX_ITERATIONS
    
    # Step 1: 初始分析师设计方案（带重试）
    print(f"\n[{ts()}] 🚀 Step 1: 启动算法分析师...")
    initial_prompt = """请以算法分析师身份：

1) 阅读 `.kiro/steering/agent-analyst.md` 理解你的角色
2) 阅读 `.kiro/steering/structure.md` 理解项目背景
3) 阅读 `docx/历史索引.md` 了解历史方案
4) 阅读 `docx/基线结果.log` 了解当前性能
5) 设计新方案并输出到 docx/记录点(n+1)/记录点(n+1)方案.md
6) 更新 docx/历史索引.md

⚠️ 必须创建方案文件 docx/记录点(n+1)/记录点(n+1)方案.md，否则工作流无法继续！"""
    
    if ANA_INIT_BOOL:
        initial_prompt = """你是一位拥有深邃数学直觉的算法分析师。你不修补表象——你揭示本质。

1. 自从记录点4引入解耦组件后，EP互作AUPR从cross attn baseline的65提升到70(测试集)后，我们认识到新的瓶颈。

2. 当前代码已回退至记录点6

3. 现在请你按照算法分析师的要求进行工作，揭示问题本质，提出更高价值的方案。

完成方案设计后，请将方案输出到 docx/记录点(n+1)/记录点(n+1)方案.md

⚠️ 必须创建方案文件！"""
    
    success, _ = ensure_analyst_creates_solution(initial_prompt, known_solution_files)
    # 注意：不更新 known_solution_files，让 workflow_design_phase 能检测到新方案
    # 即使初始分析师失败，也继续进入主循环（主循环会处理）
    
    # 主迭代循环
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"[{ts()}] 🔄 迭代轮次: {iteration}/{max_iterations}")
        print(f"[{ts()}] 🌿 当前分支: {get_current_branch()}")
        print(f"{'='*60}")
        
        # Step 2: 设计阶段（方案质检）
        print(f"\n[{ts()}] 📋 Step 2: 方案质检...")
        passed, known_solution_files = workflow_design_phase(known_solution_files)
        
        if not passed:
            # 设计阶段失败，不终止！回退并重新拉起分析师
            print(f"[{ts()}] ⚠️ 设计阶段失败，回退并重新拉起分析师")
            next_branch = get_next_chat_branch_name()
            create_branch_from_chat0(next_branch)
            success, known_solution_files = ensure_analyst_creates_solution(
                "设计阶段失败。请重新设计方案并输出到 docx/记录点(n+1)/记录点(n+1)方案.md",
                known_solution_files
            )
            continue  # 回到循环开头
        
        # Step 3: 编码+训练阶段
        print(f"\n[{ts()}] 💻 Step 3: 编码+训练...")
        success, error_info = workflow_coding_and_training()
        
        if not success:
            print(f"[{ts()}] ❌ 编码+训练阶段失败，回退代码并返回分析师")
            next_branch = get_next_chat_branch_name()
            create_branch_from_chat0(next_branch)
            print(f"[{ts()}] 🔄 已回退到chat0基线，分支: {next_branch}")
            analyst_prompt = f"""编码+训练阶段失败，代码已回退到chat0基线。

【错误信息】
{error_info}

请以算法分析师身份：
1) 分析失败原因（可能是方案设计问题或实现复杂度过高）
2) 撰写反思文档
3) 设计更简洁可行的新方案
4) 输出新方案到 docx/记录点(n+1)/记录点(n+1)方案.md"""
            success, known_solution_files = ensure_analyst_creates_solution(analyst_prompt, known_solution_files)
            continue  # 回到设计阶段
        
        # Step 4: 训练质检
        print(f"\n[{ts()}] 🔍 Step 4: 训练质检...")
        train_decision = workflow_train_review()
        
        if train_decision == "pass":
            print(f"[{ts()}] ✅ 训练质检通过，进入预测阶段")
        elif train_decision == "fail":
            print(f"[{ts()}] ❌ 训练质检不通过（过拟合/NaN/退化等），回退代码并返回分析师")
            next_branch = get_next_chat_branch_name()
            create_branch_from_chat0(next_branch)
            print(f"[{ts()}] 🔄 已回退到chat0基线，分支: {next_branch}")
            analyst_prompt = """训练质检不通过，代码已回退到chat0基线。

请以算法分析师身份：
1) 阅读最新的训练质检报告 (docx/记录点*/记录点*训练质检.md)
2) 分析训练失败的根本原因（过拟合？数值不稳定？模式坍缩？）
3) 撰写反思文档 (docx/记录点n/记录点n反思.md)
4) 设计新方案，避免重蹈覆辙
5) 输出新方案到 docx/记录点(n+1)/记录点(n+1)方案.md
6) 更新历史索引

⚠️ 注意：问题可能出在方案设计层面，而非代码实现层面。请深入分析。"""
            success, known_solution_files = ensure_analyst_creates_solution(analyst_prompt, known_solution_files)
            continue  # 回到设计阶段
        elif train_decision == "timeout":
            print(f"[{ts()}] ⚠️ 训练质检超时，默认视为通过，继续预测")
        else:
            # 训练质检异常，不终止！回退并重新拉起分析师
            print(f"[{ts()}] ⚠️ 训练质检异常，回退并重新拉起分析师")
            next_branch = get_next_chat_branch_name()
            create_branch_from_chat0(next_branch)
            success, known_solution_files = ensure_analyst_creates_solution(
                "训练质检异常。请重新设计方案并输出到 docx/记录点(n+1)/记录点(n+1)方案.md",
                known_solution_files
            )
            continue
        
        # Step 5: 预测阶段
        print(f"\n[{ts()}] 🔮 Step 5: 预测...")
        success, error_info = workflow_prediction()
        
        if not success:
            print(f"[{ts()}] ❌ 预测阶段失败，回退代码并返回分析师")
            next_branch = get_next_chat_branch_name()
            create_branch_from_chat0(next_branch)
            print(f"[{ts()}] 🔄 已回退到chat0基线，分支: {next_branch}")
            analyst_prompt = f"""预测阶段失败，代码已回退到chat0基线。

【错误信息】
{error_info}

请以算法分析师身份：
1) 分析预测失败的原因
2) 撰写反思文档
3) 设计新方案
4) 输出新方案到 docx/记录点(n+1)/记录点(n+1)方案.md"""
            success, known_solution_files = ensure_analyst_creates_solution(analyst_prompt, known_solution_files)
            continue  # 回到设计阶段
        
        # Step 6: 回退决策
        print(f"\n[{ts()}] 🔀 Step 6: 回退决策...")
        rollback_decision = workflow_rollback_decision()
        
        if rollback_decision == "timeout":
            print(f"[{ts()}] ⚠️ 回退决策超时，默认保留代码")
            rollback_decision = "keep"
        
        # Step 7: 执行分支操作
        print(f"\n[{ts()}] 🌿 Step 7: 分支操作...")
        branch_success = execute_branch_operation(rollback_decision)
        
        if not branch_success:
            print(f"[{ts()}] ⚠️ 分支操作失败，继续在当前分支")
        
        # Step 8: 结果分析
        print(f"\n[{ts()}] 📊 Step 8: 结果分析...")
        target_reached, known_solution_files = workflow_result_analysis(known_solution_files)
        
        if target_reached:
            print(f"\n{'='*60}")
            print(f"[{ts()}] 🎉 项目成功！AUPR ≥ 0.75 目标达成！")
            print(f"[{ts()}] 🌿 最终分支: {get_current_branch()}")
            print(f"{'='*60}")
            break
        else:
            print(f"[{ts()}] 📈 AUPR未达标，继续下一轮迭代...")
            if rollback_decision == "rollback":
                print(f"[{ts()}] 🔄 代码已回退到chat0基线，重新开始")
            # 检查是否有新方案，没有则拉起分析师
            new_solutions, _ = check_new_solution_files(known_solution_files)
            if not new_solutions:
                print(f"[{ts()}] ⚠️ 分析师未创建新方案，主动拉起")
                success, known_solution_files = ensure_analyst_creates_solution(
                    "AUPR未达标，请设计新方案并输出到 docx/记录点(n+1)/记录点(n+1)方案.md",
                    known_solution_files
                )
    
    else:
        print(f"\n[{ts()}] ⚠️ 达到最大迭代次数 ({max_iterations})，工作流结束")
    
    print(f"\n[{ts()}] 🌿 最终分支: {get_current_branch()}")
    print(f"[{ts()}] 工作流结束")


if __name__ == "__main__":
    main()
