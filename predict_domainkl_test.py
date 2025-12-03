import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DEVICE, PRISM_SAVE_MODEL_DIR, NUM_WORKERS, BATCH_SIZE, PRISM_BATCH_SIZE, CNN_KERNEL_SIZE, POOL_KERNEL_SIZE, DNA_EMBEDDING_PADDING_IDX, MAX_ENHANCER_LENGTH, MAX_PROMOTER_LENGTH
from data_loader import load_prism_data, PRISMDataset, CellBatchSampler
from models.PRISMModel import PRISMBackbone, CellClassificationExpert
from models.pleat.embedding import KMerTokenizer
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    tokenizer = getattr(collate_fn, "tokenizer", None)
    if tokenizer is None:
        tokenizer = KMerTokenizer()
        setattr(collate_fn, "tokenizer", tokenizer)
    enh_seqs = [b[0] for b in batch]
    pr_seqs = [b[1] for b in batch]
    cells = [b[2] for b in batch]
    labels = [int(b[3]) for b in batch]
    enh_ids_list = [tokenizer.encode(s) for s in enh_seqs]
    pr_ids_list = [tokenizer.encode(s) for s in pr_seqs]
    K = CNN_KERNEL_SIZE
    P = POOL_KERNEL_SIZE
    pad_id = DNA_EMBEDDING_PADDING_IDX
    min_req = K + P - 1
    max_len_en = max(int(x.size(0)) for x in enh_ids_list) if enh_ids_list else min_req
    max_len_pr = max(int(x.size(0)) for x in pr_ids_list) if pr_ids_list else min_req
    adj_base_en = max(1, max_len_en - (K - 1))
    adj_base_pr = max(1, max_len_pr - (K - 1))
    target_len_en = (K - 1) + ((adj_base_en + P - 1) // P) * P
    target_len_pr = (K - 1) + ((adj_base_pr + P - 1) // P) * P
    target_len_en = max(min_req, min(target_len_en, MAX_ENHANCER_LENGTH))
    target_len_pr = max(min_req, min(target_len_pr, MAX_PROMOTER_LENGTH))
    proc_en = []
    for ids in enh_ids_list:
        L = int(ids.size(0))
        if L > target_len_en:
            s = (L - target_len_en) // 2
            ids = ids[s:s + target_len_en]
        proc_en.append(ids)
    proc_pr = []
    for ids in pr_ids_list:
        L = int(ids.size(0))
        if L > target_len_pr:
            s = (L - target_len_pr) // 2
            ids = ids[s:s + target_len_pr]
        proc_pr.append(ids)
    enh_ids = pad_sequence(proc_en, batch_first=True, padding_value=pad_id)
    pr_ids = pad_sequence(proc_pr, batch_first=True, padding_value=pad_id)
    if enh_ids.size(1) < target_len_en:
        enh_ids = torch.nn.functional.pad(enh_ids, (0, target_len_en - enh_ids.size(1)), value=pad_id)
    if pr_ids.size(1) < target_len_pr:
        pr_ids = torch.nn.functional.pad(pr_ids, (0, target_len_pr - pr_ids.size(1)), value=pad_id)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return enh_ids, pr_ids, labels_t, cells


def _find_latest_checkpoint(save_dir: str):
    if not os.path.exists(save_dir):
        return 0, None, None
    latest_full_epoch = 0
    latest_full_path = None
    latest_model_epoch = 0
    latest_model_path = None
    for name in os.listdir(save_dir):
        m_full = re.match(r"prism_full_epoch_(\d+)\.pt$", name)
        if m_full:
            e = int(m_full.group(1))
            if e > latest_full_epoch:
                latest_full_epoch = e
                latest_full_path = os.path.join(save_dir, name)
        m_model = re.match(r"prism_epoch_(\d+)\.pth$", name)
        if m_model:
            e = int(m_model.group(1))
            if e > latest_model_epoch:
                latest_model_epoch = e
                latest_model_path = os.path.join(save_dir, name)
    epoch = max(latest_full_epoch, latest_model_epoch)
    chosen_full = latest_full_path if latest_full_epoch == epoch and latest_full_path else None
    chosen_model = latest_model_path if latest_model_epoch == epoch and latest_model_path else None
    return epoch, chosen_full, chosen_model

def load_prism_checkpoint(backbone: PRISMBackbone, cell_expert: CellClassificationExpert, save_dir: str, device: torch.device):
    epoch, full_path, model_path = _find_latest_checkpoint(save_dir)
    if epoch <= 0:
        return False
    if full_path and os.path.exists(full_path):
        sd = torch.load(full_path, map_location=device)
        if isinstance(sd, dict):
            if 'model_state' in sd:
                try:
                    backbone.load_state_dict(sd['model_state'], strict=False)
                except Exception:
                    return False
            if 'cell_expert_state' in sd:
                try:
                    cell_expert.load_state_dict(sd['cell_expert_state'], strict=False)
                except Exception:
                    pass
            return True
        return False
    if model_path and os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device)
        if isinstance(sd, dict):
            ok = False
            if 'backbone' in sd and isinstance(sd['backbone'], dict):
                try:
                    backbone.load_state_dict(sd['backbone'], strict=False)
                    ok = True
                except Exception:
                    pass
            if 'cell_expert' in sd and isinstance(sd['cell_expert'], dict):
                try:
                    cell_expert.load_state_dict(sd['cell_expert'], strict=False)
                except Exception:
                    pass
            return ok
        try:
            backbone.load_state_dict(sd, strict=False)
            return True
        except Exception:
            return False
    return False


def evaluate():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    df, e_seq, p_seq = load_prism_data("test")
    dataset = PRISMDataset(df, e_seq, p_seq)
    bs = PRISM_BATCH_SIZE if PRISM_BATCH_SIZE else BATCH_SIZE
    sampler = CellBatchSampler(dataset, batch_size=bs, shuffle=False)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    num_cells = len(dataset.cell_lines) if hasattr(dataset, 'cell_lines') else len(sorted(df['cell_line'].unique()))
    cell_expert = CellClassificationExpert(num_classes=num_cells).to(device)
    backbone = PRISMBackbone().to(device)
    _ = load_prism_checkpoint(backbone, cell_expert, PRISM_SAVE_MODEL_DIR, device)
    backbone.eval(); cell_expert.eval()
    all_preds = []
    all_labels = []
    per_cell = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict domain-kl/test"):
            enh_ids, pr_ids, labels, cells = batch
            enh_ids = enh_ids.to(device)
            pr_ids = pr_ids.to(device)
            labels_t = labels.to(device)
            cell_logits, cell_vec = cell_expert(enh_ids, pr_ids)
            outputs, _ = backbone(enh_ids, pr_ids, cell_vec)
            preds = outputs.squeeze(-1).detach().cpu().numpy()
            labs = labels_t.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labs)
            cell = cells[0] if len(cells) > 0 else "UNKNOWN"
            if cell not in per_cell:
                per_cell[cell] = {"preds": [], "labels": []}
            per_cell[cell]["preds"].append(preds)
            per_cell[cell]["labels"].append(labs)
    if len(all_preds) == 0:
        return None
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    aupr = average_precision_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    bin_preds = (all_preds >= 0.5).astype(int)
    f1 = f1_score(all_labels, bin_preds)
    rec = recall_score(all_labels, bin_preds)
    prec = precision_score(all_labels, bin_preds)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compete")
    os.makedirs(out_dir, exist_ok=True)
    pr_p, pr_r, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(pr_r, pr_p, label=f"AUPR={aupr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()
    per_cell_results = {}
    for c, d in per_cell.items():
        cp = np.concatenate(d["preds"], axis=0)
        cl = np.concatenate(d["labels"], axis=0)
        try:
            caupr = average_precision_score(cl, cp)
        except Exception:
            caupr = float("nan")
        try:
            cauc = roc_auc_score(cl, cp)
        except Exception:
            cauc = float("nan")
        cb = (cp >= 0.5).astype(int)
        cf1 = f1_score(cl, cb) if cl.size > 0 else float("nan")
        cr = recall_score(cl, cb) if cl.size > 0 else float("nan")
        cpr = precision_score(cl, cb) if cl.size > 0 else float("nan")
        per_cell_results[c] = {"aupr": caupr, "auc": cauc, "f1": cf1, "recall": cr, "precision": cpr, "n": int(cl.size)}
    return {"aupr": aupr, "auc": auc, "f1": f1, "recall": rec, "precision": prec, "per_cell": per_cell_results, "n": int(all_labels.size)}


if __name__ == "__main__":
    res = evaluate()
    if res is not None:
        # 修改输出目录为compete目录
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compete")
        os.makedirs(out_dir, exist_ok=True)
        
        # 打印结果到控制台
        print("domain-kl/test overall")
        print(f"AUPR: {res['aupr']:.4f}")
        print(f"AUC: {res['auc']:.4f}")
        print(f"F1: {res['f1']:.4f}")
        print(f"Recall: {res['recall']:.4f}")
        print(f"Precision: {res['precision']:.4f}")
        print(f"Samples: {res['n']}")
        for c, m in res["per_cell"].items():
            print(f"{c}: AUPR={m['aupr']:.4f} AUC={m['auc']:.4f} F1={m['f1']:.4f} Recall={m['recall']:.4f} Precision={m['precision']:.4f} N={m['n']}")
        
        # 保存结果到文件
        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write("domain-kl/test overall\n")
            f.write(f"AUPR: {res['aupr']:.4f}\n")
            f.write(f"AUC: {res['auc']:.4f}\n")
            f.write(f"F1: {res['f1']:.4f}\n")
            f.write(f"Recall: {res['recall']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Samples: {res['n']}\n")
            f.write("\nPer-cell results:\n")
            for c, m in res["per_cell"].items():
                f.write(f"{c}: AUPR={m['aupr']:.4f} AUC={m['auc']:.4f} F1={m['f1']:.4f} Recall={m['recall']:.4f} Precision={m['precision']:.4f} N={m['n']}\n")
