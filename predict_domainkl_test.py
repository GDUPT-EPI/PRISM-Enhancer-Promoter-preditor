import os  # 文件与目录操作
import re  # 正则表达式
import torch  # 深度学习框架
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 可视化
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve, roc_curve  # 评估指标
from torch.utils.data import DataLoader  # 数据加载器
from tqdm import tqdm  # 进度条
from typing import List, Tuple, Dict, Optional  # 类型注解

from config import (  # 集中配置导入
    DEVICE,
    PRISM_SAVE_MODEL_DIR,
    NUM_WORKERS,
    BATCH_SIZE,
    PRISM_BATCH_SIZE,
    CNN_KERNEL_SIZE,
    POOL_KERNEL_SIZE,
    DNA_EMBEDDING_PADDING_IDX,
    MAX_ENHANCER_LENGTH,
    MAX_PROMOTER_LENGTH,
)
from data_loader import load_prism_data, PRISMDataset, CellBatchSampler  # 数据加载与采样
from models.PRISMModel import PRISMBackbone  # 模型主干
from models.pleat.embedding import KMerTokenizer  # K-mer分词器
from torch.nn.utils.rnn import pad_sequence  # 序列填充


def find_optimal_threshold(labels: np.ndarray, predictions: np.ndarray, 
                          metric: str = "f1", 
                          threshold_range: Tuple[float, float] = (0.01, 0.99),
                          num_steps: int = 99) -> Tuple[float, Dict[str, float]]:
    """寻找最优分类阈值
    
    基于指定指标在给定范围内搜索最佳分类阈值。
    
    Args:
        labels: 真实标签 (0或1)
        predictions: 预测概率值 (0-1之间)
        metric: 优化指标，可选'f1', 'precision', 'recall', 'accuracy'
        threshold_range: 阈值搜索范围 (最小值, 最大值)
        num_steps: 搜索步数
        
    Returns:
        (最佳阈值, 各指标在最佳阈值下的值字典)
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
    best_score = -1.0
    best_threshold = threshold_range[0]
    
    # 存储每个阈值对应的指标值
    metrics_history = {
        "thresholds": thresholds,
        "f1": [],
        "precision": [],
        "recall": [],
        "accuracy": []
    }
    
    for threshold in thresholds:
        binary_preds = (predictions >= threshold).astype(int)
        
        # 计算各项指标
        f1 = f1_score(labels, binary_preds, zero_division=0)
        precision = precision_score(labels, binary_preds, zero_division=0)
        recall = recall_score(labels, binary_preds, zero_division=0)
        accuracy = np.mean(binary_preds == labels)
        
        # 存储指标历史
        metrics_history["f1"].append(f1)
        metrics_history["precision"].append(precision)
        metrics_history["recall"].append(recall)
        metrics_history["accuracy"].append(accuracy)
        
        # 根据指定指标选择最佳阈值
        current_score = locals()[metric]
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold
    
    # 计算最佳阈值下的各项指标
    binary_preds = (predictions >= best_threshold).astype(int)
    best_metrics = {
        "threshold": best_threshold,
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "accuracy": np.mean(binary_preds == labels)
    }
    
    return best_threshold, best_metrics, metrics_history


class EvalConfig:
    """评估配置类

    将所有评估相关的可调参数集中到一个类中，避免在代码中硬编码散落。

    说明：本文件不接受命令行参数，遵循项目规则的“配置集中管理”。
    """

    DEVICE = torch.device(DEVICE if torch.cuda.is_available() else "cpu")  # 设备选择
    SAVE_DIR = PRISM_SAVE_MODEL_DIR  # 权重保存目录
    BATCH_SIZE = PRISM_BATCH_SIZE or BATCH_SIZE  # 评估批量大小
    NUM_WORKERS = NUM_WORKERS  # DataLoader工作线程数
    PIN_MEMORY = True  # 加速CPU→GPU拷贝
    SHUFFLE = False  # 评估不打乱

    THRESHOLD = 0.92  # 二值化阈值
    OUTPUT_DIR_NAME = "compete/fff"  # 输出目录名称
    PLOT_PR = True  # 是否绘制PR曲线
    PLOT_ROC = True  # 是否绘制ROC曲线
    
    # 阈值寻找相关配置
    FIND_OPTIMAL_THRESHOLD = False  # 是否启用阈值寻找
    OPTIMIZE_METRIC = "f1"  # 优化指标，可选'f1', 'precision', 'recall', 'accuracy'
    THRESHOLD_RANGE = (0.01, 0.99)  # 阈值搜索范围
    THRESHOLD_STEPS = 99  # 搜索步数
    PLOT_THRESHOLD_METRICS = True  # 是否绘制阈值-指标曲线


def collate_fn(batch: List[Tuple[str, str, str, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """批处理拼接函数

    将原始DNA序列转换为固定长度的token ID张量，并返回标签与细胞系名称列表。

    Args:
        batch: 列表项为 (enhancer_seq, promoter_seq, cell_line, label)

    Returns:
        (enh_ids, pr_ids, labels_t, cells)
    """
    tokenizer = getattr(collate_fn, "tokenizer", None)  # 复用静态tokenizer避免重复构造
    if tokenizer is None:  # 首次调用时构建
        tokenizer = KMerTokenizer()  # 6-mer分词器
        setattr(collate_fn, "tokenizer", tokenizer)  # 挂载到函数属性
    enh_seqs = [b[0] for b in batch]  # 增强子序列列表
    pr_seqs = [b[1] for b in batch]  # 启动子序列列表
    cells = [b[2] for b in batch]  # 细胞系名称列表
    labels = [int(b[3]) for b in batch]  # 标签列表
    enh_ids_list = [tokenizer.encode(s) for s in enh_seqs]  # 增强子编码
    pr_ids_list = [tokenizer.encode(s) for s in pr_seqs]  # 启动子编码
    K = CNN_KERNEL_SIZE  # 卷积核大小
    P = POOL_KERNEL_SIZE  # 池化核大小
    pad_id = DNA_EMBEDDING_PADDING_IDX  # PAD索引
    min_req = K + P - 1  # 最小有效长度（保证卷积+池化后至少一个片段）
    max_len_en = max(int(x.size(0)) for x in enh_ids_list) if enh_ids_list else min_req  # 批内最大增强子长度
    max_len_pr = max(int(x.size(0)) for x in pr_ids_list) if pr_ids_list else min_req  # 批内最大启动子长度
    adj_base_en = max(1, max_len_en - (K - 1))  # 去除卷积边界后可池化基长
    adj_base_pr = max(1, max_len_pr - (K - 1))  # 去除卷积边界后可池化基长
    target_len_en = (K - 1) + ((adj_base_en + P - 1) // P) * P  # 按池化步长对齐的目标增强子长度
    target_len_pr = (K - 1) + ((adj_base_pr + P - 1) // P) * P  # 按池化步长对齐的目标启动子长度
    target_len_en = max(min_req, min(target_len_en, MAX_ENHANCER_LENGTH))  # 长度裁剪（上限）
    target_len_pr = max(min_req, min(target_len_pr, MAX_PROMOTER_LENGTH))  # 长度裁剪（上限）
    proc_en: List[torch.Tensor] = []  # 处理后的增强子列表
    for ids in enh_ids_list:  # 遍历增强子
        L = int(ids.size(0))  # 原始长度
        if L > target_len_en:  # 超长居中裁剪
            s = (L - target_len_en) // 2  # 起始位置
            ids = ids[s:s + target_len_en]  # 居中截取
        proc_en.append(ids)  # 收集
    proc_pr: List[torch.Tensor] = []  # 处理后的启动子列表
    for ids in pr_ids_list:  # 遍历启动子
        L = int(ids.size(0))  # 原始长度
        if L > target_len_pr:  # 超长居中裁剪
            s = (L - target_len_pr) // 2  # 起始位置
            ids = ids[s:s + target_len_pr]  # 居中截取
        proc_pr.append(ids)  # 收集
    enh_ids = pad_sequence(proc_en, batch_first=True, padding_value=pad_id)  # 增强子PAD填充
    pr_ids = pad_sequence(proc_pr, batch_first=True, padding_value=pad_id)  # 启动子PAD填充
    if enh_ids.size(1) < target_len_en:  # 右侧补齐到目标长度
        enh_ids = torch.nn.functional.pad(enh_ids, (0, target_len_en - enh_ids.size(1)), value=pad_id)  # 右侧PAD
    if pr_ids.size(1) < target_len_pr:  # 右侧补齐到目标长度
        pr_ids = torch.nn.functional.pad(pr_ids, (0, target_len_pr - pr_ids.size(1)), value=pad_id)  # 右侧PAD
    labels_t = torch.tensor(labels, dtype=torch.long)  # 标签张量
    return enh_ids, pr_ids, labels_t, cells  # 返回拼接结果


def _find_latest_checkpoint(save_dir: str) -> Tuple[int, Optional[str], Optional[str]]:
    """查找保存目录中最近的PRISM检查点

    返回最近 epoch 号，以及完整状态文件与仅模型文件的路径。
    """
    if not os.path.exists(save_dir):  # 目录不存在
        return 0, None, None  # 无检查点
    latest_full_epoch = 0  # 最近完整状态epoch
    latest_full_path = None  # 最近完整状态路径
    latest_model_epoch = 0  # 最近模型权重epoch
    latest_model_path = None  # 最近模型权重路径
    for name in os.listdir(save_dir):  # 遍历文件名
        m_full = re.match(r"prism_full_epoch_(\d+)\.pt$", name)  # 完整状态文件匹配
        if m_full:
            e = int(m_full.group(1))  # 提取epoch
            if e > latest_full_epoch:  # 更新最近完整状态
                latest_full_epoch = e
                latest_full_path = os.path.join(save_dir, name)
        m_model = re.match(r"prism_epoch_(\d+)\.pth$", name)  # 仅模型权重匹配
        if m_model:
            e = int(m_model.group(1))  # 提取epoch
            if e > latest_model_epoch:  # 更新最近模型权重
                latest_model_epoch = e
                latest_model_path = os.path.join(save_dir, name)
    print(f"Latest full epoch: {latest_full_epoch}, latest model epoch: {latest_model_epoch}")  # 打印最近信息
    epoch = max(latest_full_epoch, latest_model_epoch)  # 选择更大epoch
    chosen_full = latest_full_path if latest_full_epoch == epoch and latest_full_path else None  # 选择完整状态
    chosen_model = latest_model_path if latest_model_epoch == epoch and latest_model_path else None  # 选择模型权重
    return epoch, chosen_full, chosen_model  # 返回

def load_prism_checkpoint(backbone: PRISMBackbone, save_dir: str, device: torch.device) -> bool:
    """加载最近PRISM检查点到模型

    优先加载完整状态，其次加载仅模型权重；对键不匹配采取非严格加载方式。
    """
    epoch, full_path, model_path = _find_latest_checkpoint(save_dir)  # 查找检查点
    if epoch <= 0:  # 未找到任何有效检查点
        return False  # 加载失败
    if full_path and os.path.exists(full_path):  # 优先完整状态
        sd = torch.load(full_path, map_location=device)  # 读取文件
        if isinstance(sd, dict):  # 字典格式
            if 'model_state' in sd:  # 包含模型权重
                try:
                    backbone.load_state_dict(sd['model_state'], strict=False)  # 非严格加载
                except Exception:  # 键不匹配等异常
                    return False  # 加载失败
            return True  # 加载成功
        return False  # 格式不匹配
    if model_path and os.path.exists(model_path):  # 退化到仅模型权重
        sd = torch.load(model_path, map_location=device)  # 读取文件
        if isinstance(sd, dict):  # 字典格式
            ok = False  # 标记
            if 'backbone' in sd and isinstance(sd['backbone'], dict):  # 兼容保存为{'backbone': state}
                try:
                    backbone.load_state_dict(sd['backbone'], strict=False)  # 非严格加载
                    ok = True  # 成功
                except Exception:
                    pass  # 忽略异常
            return ok  # 返回状态
        try:
            backbone.load_state_dict(sd, strict=False)  # 直接加载
            return True  # 成功
        except Exception:
            return False  # 失败
    return False  # 未找到文件


def evaluate() -> Optional[Dict[str, object]]:
    """在 domain-kl/test 上评估 PRISM 模型

    加载最新检查点，按细胞系分组评估，输出总体与分细胞系指标。

    Returns:
        评估结果字典或 None（无预测）
    """
    device = EvalConfig.DEVICE  # 设备
    df, e_seq, p_seq = load_prism_data("test")  # 加载测试数据
    dataset = PRISMDataset(df, e_seq, p_seq)  # 构建数据集
    bs = EvalConfig.BATCH_SIZE  # 批量大小
    sampler = CellBatchSampler(dataset, batch_size=bs, shuffle=EvalConfig.SHUFFLE)  # 细胞系批采样器
    loader = DataLoader(  # 数据加载器
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=EvalConfig.NUM_WORKERS,
        pin_memory=EvalConfig.PIN_MEMORY,
        collate_fn=collate_fn,
    )
    backbone = PRISMBackbone().to(device)  # 模型加载到设备
    _ = load_prism_checkpoint(backbone, EvalConfig.SAVE_DIR, device)  # 加载最近权重
    backbone.eval()  # 评估模式
    all_preds: List[np.ndarray] = []  # 汇总预测概率
    all_labels: List[np.ndarray] = []  # 汇总标签
    per_cell: Dict[str, Dict[str, List[np.ndarray]]] = {}  # 分细胞系缓存
    with torch.no_grad():  # 关闭梯度
        for batch in tqdm(loader, desc="Predict domain-kl/test"):  # 迭代批次
            enh_ids, pr_ids, labels, cells = batch  # 取批
            enh_ids = enh_ids.to(device)  # 移动设备
            pr_ids = pr_ids.to(device)  # 移动设备
            labels_t = labels.to(device)  # 标签设备
            outputs, _, _ = backbone(enh_ids, pr_ids)  # 前向预测
            preds = outputs.squeeze(-1).detach().cpu().numpy()  # 概率
            labs = labels_t.cpu().numpy()  # 标签
            all_preds.append(preds)  # 汇总
            all_labels.append(labs)  # 汇总
            cell = cells[0] if len(cells) > 0 else "UNKNOWN"  # 细胞系名
            if cell not in per_cell:  # 初始化细胞系项
                per_cell[cell] = {"preds": [], "labels": []}
            per_cell[cell]["preds"].append(preds)  # 添加预测
            per_cell[cell]["labels"].append(labs)  # 添加标签
    if len(all_preds) == 0:  # 无预测
        return None  # 返回空
    all_preds = np.concatenate(all_preds, axis=0)  # 拼接概率
    all_labels = np.concatenate(all_labels, axis=0)  # 拼接标签
    aupr = average_precision_score(all_labels, all_preds)  # AUPR
    auc = roc_auc_score(all_labels, all_preds)  # AUC
    
    # 创建输出目录（提前定义以便在阈值可视化中使用）
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), EvalConfig.OUTPUT_DIR_NAME)  # 输出目录
    os.makedirs(out_dir, exist_ok=True)  # 创建目录
    
    # 寻找最优阈值
    if EvalConfig.FIND_OPTIMAL_THRESHOLD:
        optimal_threshold, optimal_metrics, threshold_history = find_optimal_threshold(
            all_labels, 
            all_preds, 
            metric=EvalConfig.OPTIMIZE_METRIC,
            threshold_range=EvalConfig.THRESHOLD_RANGE,
            num_steps=EvalConfig.THRESHOLD_STEPS
        )
        print(f"Optimal threshold ({EvalConfig.OPTIMIZE_METRIC}): {optimal_threshold:.4f}")
        print(f"Optimal metrics - F1: {optimal_metrics['f1']:.4f}, "
              f"Precision: {optimal_metrics['precision']:.4f}, "
              f"Recall: {optimal_metrics['recall']:.4f}, "
              f"Accuracy: {optimal_metrics['accuracy']:.4f}")
        threshold = optimal_threshold
        
        # 绘制阈值-指标曲线
        if EvalConfig.PLOT_THRESHOLD_METRICS:
            plt.figure(figsize=(10, 6))
            plt.plot(threshold_history["thresholds"], threshold_history["f1"], label='F1 Score')
            plt.plot(threshold_history["thresholds"], threshold_history["precision"], label='Precision')
            plt.plot(threshold_history["thresholds"], threshold_history["recall"], label='Recall')
            plt.plot(threshold_history["thresholds"], threshold_history["accuracy"], label='Accuracy')
            
            # 标记最优阈值点
            optimal_idx = np.argmin(np.abs(threshold_history["thresholds"] - optimal_threshold))
            plt.plot(optimal_threshold, threshold_history["f1"][optimal_idx], 'ro', 
                    label=f'Optimal Threshold ({optimal_threshold:.4f})')
            
            plt.xlabel('Threshold')
            plt.ylabel('Metric Value')
            plt.title('Metrics vs. Threshold')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "threshold_metrics.png"))
            plt.close()
    else:
        threshold = EvalConfig.THRESHOLD
    
    bin_preds = (all_preds >= threshold).astype(int)  # 二值化
    f1 = f1_score(all_labels, bin_preds, zero_division=0)  # F1
    rec = recall_score(all_labels, bin_preds, zero_division=0)  # 召回
    prec = precision_score(all_labels, bin_preds, zero_division=0)  # 精度
    pr_p, pr_r, _ = precision_recall_curve(all_labels, all_preds)  # PR曲线数据
    if EvalConfig.PLOT_PR:  # 绘制PR
        plt.figure()  # 新图
        plt.plot(pr_r, pr_p, label=f"AUPR={aupr:.4f}")  # 曲线
        plt.xlabel("Recall")  # 轴标签
        plt.ylabel("Precision")  # 轴标签
        plt.title("Precision-Recall Curve")  # 标题
        plt.legend()  # 图例
        plt.tight_layout()  # 紧凑布局
        plt.savefig(os.path.join(out_dir, "pr_curve.png"))  # 保存
        plt.close()  # 关闭
    fpr, tpr, _ = roc_curve(all_labels, all_preds)  # ROC曲线数据
    if EvalConfig.PLOT_ROC:  # 绘制ROC
        plt.figure()  # 新图
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")  # 曲线
        plt.xlabel("False Positive Rate")  # 轴标签
        plt.ylabel("True Positive Rate")  # 轴标签
        plt.title("ROC Curve")  # 标题
        plt.legend()  # 图例
        plt.tight_layout()  # 紧凑布局
        plt.savefig(os.path.join(out_dir, "roc_curve.png"))  # 保存
        plt.close()  # 关闭
    per_cell_results: Dict[str, Dict[str, float]] = {}  # 细胞系指标
    for c, d in per_cell.items():  # 遍历细胞系
        cp = np.concatenate(d["preds"], axis=0)  # 概率
        cl = np.concatenate(d["labels"], axis=0)  # 标签
        try:
            caupr = average_precision_score(cl, cp)  # AUPR
        except Exception:
            caupr = float("nan")  # 异常处理
        try:
            cauc = roc_auc_score(cl, cp)  # AUC
        except Exception:
            cauc = float("nan")  # 异常处理
        
        # 对每个细胞系也可以单独寻找最优阈值（可选）
        if EvalConfig.FIND_OPTIMAL_THRESHOLD:
            cell_optimal_threshold, cell_optimal_metrics, _ = find_optimal_threshold(
                cl, cp, metric=EvalConfig.OPTIMIZE_METRIC,
                threshold_range=EvalConfig.THRESHOLD_RANGE,
                num_steps=EvalConfig.THRESHOLD_STEPS
            )
            cb = (cp >= cell_optimal_threshold).astype(int)  # 使用细胞系特定阈值
            cell_threshold = cell_optimal_threshold
        else:
            cb = (cp >= threshold).astype(int)  # 使用全局阈值
            cell_threshold = threshold
            
        cf1 = f1_score(cl, cb, zero_division=0) if cl.size > 0 else float("nan")  # F1
        cr = recall_score(cl, cb, zero_division=0) if cl.size > 0 else float("nan")  # 召回
        cpr = precision_score(cl, cb, zero_division=0) if cl.size > 0 else float("nan")  # 精度
        per_cell_results[c] = {  # 存储结果
            "aupr": caupr,
            "auc": cauc,
            "f1": cf1,
            "recall": cr,
            "precision": cpr,
            "threshold": cell_threshold,  # 添加阈值信息
            "n": int(cl.size),
        }
    return {  # 返回总体结果
        "aupr": aupr,
        "auc": auc,
        "f1": f1,
        "recall": rec,
        "precision": prec,
        "threshold": threshold,  # 添加使用的阈值
        "per_cell": per_cell_results,
        "n": int(all_labels.size),
    }


if __name__ == "__main__":
    res = evaluate()  # 执行评估
    if res is not None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), EvalConfig.OUTPUT_DIR_NAME)  # 输出目录
        os.makedirs(out_dir, exist_ok=True)  # 确保存在

        print("domain-kl/test overall")  # 概览
        print(f"AUPR: {res['aupr']:.4f}")  # AUPR
        print(f"AUC: {res['auc']:.4f}")  # AUC
        print(f"F1: {res['f1']:.4f}")  # F1
        print(f"Recall: {res['recall']:.4f}")  # 召回
        print(f"Precision: {res['precision']:.4f}")  # 精度
        print(f"Threshold: {res['threshold']:.4f}")  # 阈值
        print(f"Samples: {res['n']}")  # 样本数
        for c, m in res["per_cell"].items():  # 逐细胞系打印
            print(
                f"{c}: AUPR={m['aupr']:.4f} AUC={m['auc']:.4f} F1={m['f1']:.4f} "
                f"Recall={m['recall']:.4f} Precision={m['precision']:.4f} "
                f"Threshold={m['threshold']:.4f} N={m['n']}"
            )

        # 保存文本结果
        with open(os.path.join(out_dir, "results.txt"), "w") as f:  # 打开文件
            f.write("domain-kl/test overall\n")  # 标题
            f.write(f"AUPR: {res['aupr']:.4f}\n")  # AUPR
            f.write(f"AUC: {res['auc']:.4f}\n")  # AUC
            f.write(f"F1: {res['f1']:.4f}\n")  # F1
            f.write(f"Recall: {res['recall']:.4f}\n")  # 召回
            f.write(f"Precision: {res['precision']:.4f}\n")  # 精度
            f.write(f"Threshold: {res['threshold']:.4f}\n")  # 阈值
            f.write(f"Samples: {res['n']}\n")  # 样本数
            f.write("\nPer-cell results:\n")  # 子标题
            for c, m in res["per_cell"].items():  # 逐细胞系
                f.write(
                    f"{c}: AUPR={m['aupr']:.4f} AUC={m['auc']:.4f} F1={m['f1']:.4f} "
                    f"Recall={m['recall']:.4f} Precision={m['precision']:.4f} "
                    f"Threshold={m['threshold']:.4f} N={m['n']}\n"
                )
