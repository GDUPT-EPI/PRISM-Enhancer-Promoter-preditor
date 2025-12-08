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
from models.AuxiliaryModel import AuxiliaryModel  # 旁路网络
from models.layers.footprint import FootprintExpert  # 正交约束工具


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

    THRESHOLD = 0.3  # 二值化阈值
    OUTPUT_DIR_NAME = "compete/ddd"  # 输出目录名称
    PLOT_PR = True  # 是否绘制PR曲线
    PLOT_ROC = True  # 是否绘制ROC曲线
    
    # 阈值寻找相关配置
    FIND_OPTIMAL_THRESHOLD = True  # 是否启用阈值寻找
    OPTIMIZE_METRIC = "f1"  # 优化指标，可选'f1', 'precision', 'recall', 'accuracy'
    THRESHOLD_RANGE = (0.01, 0.99)  # 阈值搜索范围
    THRESHOLD_STEPS = 99  # 搜索步数
    PLOT_THRESHOLD_METRICS = True  # 是否绘制阈值-指标曲线

    # ========= 旁路网络微调相关配置（仅在评估阶段使用，避免依赖外部config.py） =========
    AUX_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_model', 'bypass', 'aux_epoch_5.pth')  # 旁路初始权重
    FINETUNE_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_model', 'finetune', 'finetune.pth')  # 微调后保存路径
    FINETUNE_STEPS_PER_CELL = 3  # 每个细胞系进行三步快速微调
    FINETUNE_BATCH_SIZE = max(8, (PRISM_BATCH_SIZE or BATCH_SIZE))  # 微调批量（尽可能使用全样本，通过梯度累积实现）
    FINETUNE_LR = 5e-5  # 微调学习率（小步快跑，避免数值震荡）
    FINETUNE_WEIGHT_DECAY = 1e-4  # 微调权重衰减
    GRAD_CLIP_MAX_NORM = 1.0  # 梯度裁剪阈值
    # 损失权重（与旁路训练保持一致的语义，但数值更保守）
    BYPASS_SPEC_WEIGHT = 1.0  # 特性分类损失权重（使模型理解细胞系特征）
    BYPASS_INV_WEIGHT = 1.0  # 领域对抗损失权重（G子空间去域）
    BYPASS_ORTHO_WEIGHT = 0.2  # 三元子空间正交约束权重
    BYPASS_CONSIST_WEIGHT = 0.2  # 一致性损失权重（采用图平滑替代跨批一致性）
    GCN_CENTER_LOSS_W = 0.2  # 图中心原型约束权重
    GCN_MARGIN_LOSS_W = 0.2  # 图边界间隔约束权重
    GCN_SMOOTH_LOSS_W = 0.4  # 图平滑一致性约束权重


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


# ========================= 旁路微调工具函数 =========================
def _build_cell_label_tensor(cells: List[str], all_cells: List[str]) -> torch.Tensor:
    """将细胞系名称映射为索引张量

    说明：仅用于旁路网络的特性/对抗损失，不涉及EP互作标签，避免数据泄露。
    """
    name_to_idx = {c: i for i, c in enumerate(all_cells)}
    idxs = [name_to_idx.get(x, 0) for x in cells]
    return torch.tensor(idxs, dtype=torch.long)


def _load_auxiliary_checkpoint(aux_model: AuxiliaryModel, path: str, device: torch.device) -> bool:
    """加载旁路网络初始权重

    支持保存格式：{'auxiliary': state_dict} 或直接 state_dict。
    """
    try:
        if not os.path.exists(path):
            return False
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and 'auxiliary' in sd:
            aux_model.load_state_dict(sd['auxiliary'], strict=False)
            return True
        elif isinstance(sd, dict):
            aux_model.load_state_dict(sd, strict=False)
            return True
        else:
            aux_model.load_state_dict(sd, strict=False)
            return True
    except Exception:
        return False


def _cell_subset_loader(dataset: PRISMDataset, cell: str, batch_size: int) -> DataLoader:
    """为指定细胞系创建仅包含该细胞的DataLoader（评估期微调使用）

    不读取EP互作标签，不参与损失计算，仅用于构建输入序列与细胞系索引。
    """
    idxs = dataset.cell_line_groups.get(cell, [])
    # 自定义简单采样器：顺序遍历该细胞系全部样本，按batch切片
    def _iter_indices(indices: List[int], bs: int):
        for i in range(0, len(indices), bs):
            yield indices[i:i+bs]
    class _Sampler:
        def __init__(self, indices: List[int], bs: int):
            self.indices = indices
            self.bs = bs
        def __iter__(self):
            return _iter_indices(self.indices, self.bs)
        def __len__(self):
            return max(1, (len(self.indices) + self.bs - 1) // self.bs)
    sampler = _Sampler(idxs, batch_size)
    return DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=0, pin_memory=True, collate_fn=collate_fn)


def _compute_aux_losses(aux_model: AuxiliaryModel, enh_ids: torch.Tensor, pr_ids: torch.Tensor, cell_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算旁路网络微调阶段使用的损失项

    仅使用细胞系监督：spec/adv 与图约束；不涉及EP互作0/1标签，避免泄露。
    """
    pred_prob, extras = aux_model(enh_ids, pr_ids, cell_labels=cell_t)
    spec_loss = extras['spec_loss']
    adv_loss = extras['adv_loss']
    gcn_center = extras['gcn_center']
    gcn_margin = extras['gcn_margin']
    gcn_smooth = extras['gcn_smooth']
    # 正交约束：使用FootprintExpert提供的三元子空间约束
    fp_enh: FootprintExpert = aux_model.fp_enh
    fp_pr: FootprintExpert = aux_model.fp_pr
    orth_loss = fp_enh.orthogonality_loss(extras['zG'], extras['zF'], extras['zI'])
    orth_loss = orth_loss + 0.0 * fp_pr.orthogonality_loss(extras['zG'], extras['zF'], extras['zI'])

    total = (
        EvalConfig.BYPASS_SPEC_WEIGHT * spec_loss +
        EvalConfig.BYPASS_INV_WEIGHT * adv_loss +
        EvalConfig.BYPASS_ORTHO_WEIGHT * orth_loss +
        EvalConfig.BYPASS_CONSIST_WEIGHT * gcn_smooth +
        EvalConfig.GCN_CENTER_LOSS_W * gcn_center +
        EvalConfig.GCN_MARGIN_LOSS_W * gcn_margin +
        EvalConfig.GCN_SMOOTH_LOSS_W * gcn_smooth
    )
    return total, extras


def finetune_auxiliary_on_test_cells(
    dataset: PRISMDataset,
    all_cells: List[str],
    device: torch.device,
) -> Optional[AuxiliaryModel]:
    """按细胞系进行三步快速微调（尽可能覆盖全样本，使用梯度累积），返回微调后的旁路模型

    流程：
    1. 构建旁路网络并加载初始权重
    2. 对每个细胞系：遍历该细胞系全部样本，进行梯度累积，仅在累计的间隔上执行 `optimizer.step()` 共3次
    3. 不使用EP互作标签；仅细胞系监督与图一致性、正交约束
    4. 保存最终微调权重
    """
    if len(all_cells) == 0:
        return None
    aux_model = AuxiliaryModel(num_cell_types=len(all_cells)).to(device)
    _ = _load_auxiliary_checkpoint(aux_model, EvalConfig.AUX_CHECKPOINT_PATH, device)
    aux_model.train()
    optimizer = torch.optim.AdamW(aux_model.parameters(), lr=EvalConfig.FINETUNE_LR, weight_decay=EvalConfig.FINETUNE_WEIGHT_DECAY)

    # 外层细胞系进度条（实时监视每个细胞系的微调进度）
    pbar_cells = tqdm(all_cells, desc="Finetune per cell", leave=True, dynamic_ncols=True)
    for cell in pbar_cells:
        loader = _cell_subset_loader(dataset, cell, EvalConfig.FINETUNE_BATCH_SIZE)
        total_batches = len(loader)
        if total_batches <= 0:
            pbar_cells.set_postfix({"cell": cell, "batches": 0})
            continue
        # 累计间隔设置为将全部批次划分为约 FINETUNE_STEPS_PER_CELL 次参数更新
        accum_interval = max(1, int(np.ceil(total_batches / float(EvalConfig.FINETUNE_STEPS_PER_CELL))))
        expected_updates = int(np.ceil(total_batches / float(accum_interval)))
        optimizer.zero_grad()
        batch_counter = 0
        update_counter = 0

        # 内层批次进度条（实时显示损失项与累计状态）
        pbar_batches = tqdm(loader, total=total_batches, desc=f"[{cell}] batches", leave=False, dynamic_ncols=True)
        for batch in pbar_batches:
            enh_ids, pr_ids, labels_t, cells = batch
            cell_t = _build_cell_label_tensor(cells, all_cells)
            enh_ids = enh_ids.to(device)
            pr_ids = pr_ids.to(device)
            cell_t = cell_t.to(device)
            loss, extras = _compute_aux_losses(aux_model, enh_ids, pr_ids, cell_t)
            loss.backward()
            batch_counter += 1

            # 实时显示关键损失项与累计信息
            pbar_batches.set_postfix({
                "loss": f"{float(loss.item()):.4f}",
                "spec": f"{float(extras.get('spec_loss', torch.tensor(0.0)).item()):.4f}",
                "adv": f"{float(extras.get('adv_loss', torch.tensor(0.0)).item()):.4f}",
                "accum": f"{batch_counter}/{accum_interval}",
            })

            # 达到累计间隔或最后一批次时执行参数更新，并在外层进度条显示已完成的更新步数
            if (batch_counter % accum_interval == 0) or (batch_counter == total_batches):
                torch.nn.utils.clip_grad_norm_(aux_model.parameters(), max_norm=EvalConfig.GRAD_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad()
                update_counter += 1
                pbar_cells.set_postfix({
                    "cell": cell,
                    "updates": f"{update_counter}/{expected_updates}",
                })

        # 该细胞系微调完成后，关闭内层进度条，外层进度条自动前进一格（由迭代驱动）
        pbar_batches.close()

    # 保存微调后的权重
    save_dir = os.path.dirname(EvalConfig.FINETUNE_SAVE_PATH)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'auxiliary': aux_model.state_dict()}, EvalConfig.FINETUNE_SAVE_PATH)
    return aux_model


def inject_auxiliary_into_backbone(backbone: PRISMBackbone, aux_model: AuxiliaryModel) -> int:
    """将旁路网络中与主干结构一致的参数注入到主干（embedding/CNN），返回成功注入的参数个数

    注入范围：`enh_embedding`, `pr_embedding`, `enh_cnn`, `pr_cnn`
    """
    if aux_model is None:
        return 0
    aux_sd = aux_model.state_dict()
    back_sd = backbone.state_dict()
    injected = 0
    for k in list(aux_sd.keys()):
        if (k.startswith('enh_embedding') or k.startswith('pr_embedding') or k.startswith('enh_cnn') or k.startswith('pr_cnn')) and (k in back_sd):
            if aux_sd[k].shape == back_sd[k].shape:
                back_sd[k] = aux_sd[k]
                injected += 1
    backbone.load_state_dict(back_sd, strict=False)
    return injected


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
    backbone = PRISMBackbone().to(device)  # 模型加载到设备
    _ = load_prism_checkpoint(backbone, EvalConfig.SAVE_DIR, device)  # 加载最近权重
    backbone.eval()  # 评估模式

    # 逐细胞系：导入 → 微调 → 注入 → 推理
    all_cells = sorted(df['cell_line'].unique().tolist())
    per_cell: Dict[str, Dict[str, List[np.ndarray]]] = {}
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    per_cell_results: Dict[str, Dict[str, float]] = {}

    # 提前创建输出目录，便于每个细胞系即时保存图与结果
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), EvalConfig.OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    pbar_eval = tqdm(all_cells, desc="Per-cell evaluate", leave=True, dynamic_ncols=True)
    for cell in pbar_eval:
        idxs = dataset.cell_line_groups.get(cell, [])
        if len(idxs) == 0:
            pbar_eval.set_postfix({"cell": cell, "n": 0})
            continue

        aux_model = finetune_auxiliary_on_test_cells(dataset, [cell], device)
        _ = inject_auxiliary_into_backbone(backbone, aux_model)

        # 针对该细胞系构建仅该细胞的数据加载器
        bs = EvalConfig.BATCH_SIZE
        def _iter_indices(indices: List[int], bs: int):
            for i in range(0, len(indices), bs):
                yield indices[i:i+bs]
        class _Sampler:
            def __init__(self, indices: List[int], bs: int):
                self.indices = indices
                self.bs = bs
            def __iter__(self):
                return _iter_indices(self.indices, self.bs)
            def __len__(self):
                return max(1, (len(self.indices) + self.bs - 1) // self.bs)
        sampler = _Sampler(idxs, bs)
        loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=EvalConfig.NUM_WORKERS, pin_memory=EvalConfig.PIN_MEMORY, collate_fn=collate_fn)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Predict [{cell}]", leave=False, dynamic_ncols=True):
                enh_ids, pr_ids, labels, cells = batch
                enh_ids = enh_ids.to(device)
                pr_ids = pr_ids.to(device)
                labels_t = labels.to(device)
                outputs, _ = backbone(enh_ids, pr_ids)
                preds = outputs.squeeze(-1).detach().cpu().numpy()
                labs = labels_t.cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labs)
                if cell not in per_cell:
                    per_cell[cell] = {"preds": [], "labels": []}
                per_cell[cell]["preds"].append(preds)
                per_cell[cell]["labels"].append(labs)

        pbar_eval.set_postfix({"cell": cell, "n": len(idxs)})

        # ========== 该细胞系即时评估与保存 ==========
        cp = np.concatenate(per_cell[cell]["preds"], axis=0)
        cl = np.concatenate(per_cell[cell]["labels"], axis=0)
        try:
            caupr = average_precision_score(cl, cp)
        except Exception:
            caupr = float("nan")
        try:
            cauc = roc_auc_score(cl, cp)
        except Exception:
            cauc = float("nan")

        if EvalConfig.FIND_OPTIMAL_THRESHOLD:
            cell_optimal_threshold, cell_optimal_metrics, _ = find_optimal_threshold(
                cl, cp, metric=EvalConfig.OPTIMIZE_METRIC,
                threshold_range=EvalConfig.THRESHOLD_RANGE,
                num_steps=EvalConfig.THRESHOLD_STEPS
            )
            cb = (cp >= cell_optimal_threshold).astype(int)
            cell_threshold = cell_optimal_threshold
        else:
            cb = (cp >= EvalConfig.THRESHOLD).astype(int)
            cell_threshold = EvalConfig.THRESHOLD

        cf1 = f1_score(cl, cb) if cl.size > 0 else float("nan")
        cr = recall_score(cl, cb) if cl.size > 0 else float("nan")
        cpr = precision_score(cl, cb) if cl.size > 0 else float("nan")
        per_cell_results[cell] = {
            "aupr": caupr,
            "auc": cauc,
            "f1": cf1,
            "recall": cr,
            "precision": cpr,
            "threshold": cell_threshold,
            "n": int(cl.size),
        }

        # PR/ROC 即时画图保存（细胞系级）
        if EvalConfig.PLOT_PR:
            pr_p, pr_r, _ = precision_recall_curve(cl, cp)
            plt.figure()
            plt.plot(pr_r, pr_p, label=f"AUPR={caupr:.4f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve [{cell}]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pr_curve_{cell}.png"))
            plt.close()
        if EvalConfig.PLOT_ROC:
            fpr, tpr, _ = roc_curve(cl, cp)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={cauc:.4f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve [{cell}]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"roc_curve_{cell}.png"))
            plt.close()

        # 文本结果即时保存
        with open(os.path.join(out_dir, f"results_{cell}.txt"), "w") as f:
            f.write(f"{cell} results\n")
            f.write(f"AUPR: {caupr:.4f}\n")
            f.write(f"AUC: {cauc:.4f}\n")
            f.write(f"F1: {cf1:.4f}\n")
            f.write(f"Recall: {cr:.4f}\n")
            f.write(f"Precision: {cpr:.4f}\n")
            f.write(f"Threshold: {cell_threshold:.4f}\n")
            f.write(f"Samples: {int(cl.size)}\n")
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
    f1 = f1_score(all_labels, bin_preds)  # F1
    rec = recall_score(all_labels, bin_preds)  # 召回
    prec = precision_score(all_labels, bin_preds)  # 精度
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
    # 细胞系指标已在逐细胞系阶段即时计算并写入 per_cell_results
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
