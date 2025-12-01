import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from pathlib import Path

from models.pleat.embedding import KMerTokenizer, create_dna_embedding_layer
from models.layers.footprint import LCWnetFootprint
from data_loader import load_prism_data, PRISMDataset


class TestConfig:
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kb_path = str(self.root / "save_model/prism/footprint_kb_epoch_3.pt")
        self.batch_size = 96
        self.max_en_len = 1000
        self.max_pr_len = 4000
        self.vocab_size = 4097
        self.embed_dim = 768
        self.pad_id = 0
        self.cnn_kernel = 128
        self.pool_kernel = 20
        self.out_channels = 96


def collate_fn(batch: List[Tuple[str, str, str, int]], cfg: TestConfig):
    tokenizer = collate_fn.tokenizer
    enh = [b[0] for b in batch]
    prom = [b[1] for b in batch]
    cells = [b[2] for b in batch]
    labels = [b[3] for b in batch]

    enh_ids = [tokenizer.encode(s) for s in enh]
    prom_ids = [tokenizer.encode(s) for s in prom]

    padded_enh = torch.nn.utils.rnn.pad_sequence(enh_ids, batch_first=True, padding_value=cfg.pad_id)
    padded_prom = torch.nn.utils.rnn.pad_sequence(prom_ids, batch_first=True, padding_value=cfg.pad_id)

    if padded_enh.size(1) < cfg.max_en_len:
        padded_enh = F.pad(padded_enh, (0, cfg.max_en_len - padded_enh.size(1)), value=cfg.pad_id)
    if padded_prom.size(1) < cfg.max_pr_len:
        padded_prom = F.pad(padded_prom, (0, cfg.max_pr_len - padded_prom.size(1)), value=cfg.pad_id)

    return padded_enh, padded_prom, cells, torch.tensor(labels, dtype=torch.float32)


collate_fn.tokenizer = KMerTokenizer()


class EnhancerExtractor(nn.Module):
    def __init__(self, cfg: TestConfig):
        super().__init__()
        self.embedding = create_dna_embedding_layer(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.embed_dim,
            padding_idx=cfg.pad_id,
            init_std=0.1,
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=cfg.embed_dim, out_channels=cfg.out_channels, kernel_size=cfg.cnn_kernel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.pool_kernel, stride=cfg.pool_kernel),
            nn.BatchNorm1d(cfg.out_channels),
            nn.Dropout(p=0.45),
        )

    def forward(self, enhancer_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(enhancer_ids)
        x = self.cnn(emb.permute(0, 2, 1))
        return x.permute(0, 2, 1)


def load_kb(path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    kb = torch.load(path, map_location="cpu")
    return kb


def centers_to_tensor(centers: Dict[str, torch.Tensor], device: torch.device) -> Tuple[List[str], torch.Tensor]:
    names = sorted(centers.keys())
    mats = [centers[n].to(device) for n in names]
    return names, torch.stack(mats, dim=0)


def classify(footprints: torch.Tensor, names: List[str], centers: torch.Tensor) -> List[str]:
    f = F.normalize(footprints, dim=-1)
    c = F.normalize(centers, dim=-1)
    sims = torch.matmul(f, c.T)
    idx = sims.argmax(dim=-1).tolist()
    return [names[i] for i in idx]


def compute_cell_means(split: str, cfg: TestConfig, extractor: EnhancerExtractor, footprint: LCWnetFootprint) -> Dict[str, torch.Tensor]:
    pairs_df, e_seqs, p_seqs = load_prism_data(split)
    ds = PRISMDataset(pairs_df, e_seqs, p_seqs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(b, cfg))
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    with torch.no_grad():
        for enh_ids, prom_ids, cells, labels in dl:
            enh_ids = enh_ids.to(cfg.device)
            feats = extractor(enh_ids)
            fp = footprint.forward_vector(feats)
            for i, c in enumerate(cells):
                v = fp[i].detach()
                if c not in sums:
                    sums[c] = torch.zeros_like(v)
                    counts[c] = 0
                sums[c] = sums[c] + v
                counts[c] += 1
    means = {c: (sums[c] / max(counts[c], 1)).to(cfg.device) for c in sums}
    return means


def solve_map(means: Dict[str, torch.Tensor], centers: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[str]]:
    names = sorted(set(means.keys()) & set(centers.keys()))
    if not names:
        return torch.eye(next(iter(means.values())).numel(), device=next(iter(means.values())).device), []
    M = torch.stack([means[n] for n in names], dim=0)
    K = torch.stack([centers[n].to(M.device) for n in names], dim=0)
    pinv = torch.linalg.pinv(M)
    A = pinv @ K
    return A, names


def evaluate(cfg: TestConfig) -> Dict[str, float]:
    pairs_df, e_seqs, p_seqs = load_prism_data("val")
    ds = PRISMDataset(pairs_df, e_seqs, p_seqs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(b, cfg))

    kb = load_kb(cfg.kb_path)
    com_names_all, com_centers_all = centers_to_tensor(kb["com_centers"], cfg.device)
    spec_names_all, spec_centers_all = centers_to_tensor(kb["spec_centers"], cfg.device)

    extractor = EnhancerExtractor(cfg).to(cfg.device)
    footprint = LCWnetFootprint(d_model=cfg.out_channels).to(cfg.device)
    extractor.eval()
    footprint.eval()

    train_means = compute_cell_means("train", cfg, extractor, footprint)
    A_com, common_com = solve_map(train_means, kb["com_centers"])  
    if common_com:
        idxs = [com_names_all.index(n) for n in common_com]
        com_centers = com_centers_all[idxs]
        com_names = common_com
    else:
        com_centers = com_centers_all
        com_names = com_names_all
    B_spec, common_spec = solve_map(train_means, kb["spec_centers"])  
    if common_spec:
        idxs_s = [spec_names_all.index(n) for n in common_spec]
        spec_centers = spec_centers_all[idxs_s]
        spec_names = common_spec
    else:
        spec_centers = spec_centers_all
        spec_names = spec_names_all

    total = 0
    correct = 0
    total_s = 0
    correct_s = 0
    all_fp = []
    all_cell = []

    with torch.no_grad():
        for enh_ids, prom_ids, cells, labels in dl:
            enh_ids = enh_ids.to(cfg.device)
            feats = extractor(enh_ids)
            fp = footprint.forward_vector(feats)
            fp_com = fp @ A_com
            preds = classify(fp_com, com_names, com_centers)
            fp_spec = fp @ B_spec
            preds_s = classify(fp_spec, spec_names, spec_centers)

            total += len(cells)
            correct += sum(1 for p, g in zip(preds, cells) if p == g)
            total_s += len(cells)
            correct_s += sum(1 for p, g in zip(preds_s, cells) if p == g)
            all_fp.append(fp.cpu())
            all_cell.extend(cells)

    acc = correct / max(total, 1)
    acc_s = correct_s / max(total_s, 1)

    try:
        import matplotlib.pyplot as plt
        X = torch.cat(all_fp, dim=0).numpy()
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = torch.linalg.svd(torch.tensor(Xc), full_matrices=False)
        PCs = (torch.tensor(Xc) @ Vt[:, :2]).numpy()
        uniq = sorted(set(all_cell))
        colors = {c: i for i, c in enumerate(uniq)}
        cmap = plt.get_cmap("tab20")
        plt.figure(figsize=(8, 6))
        for c in uniq:
            idxs = [i for i, cc in enumerate(all_cell) if cc == c]
            pts = PCs[idxs]
            plt.scatter(pts[:, 0], pts[:, 1], s=12, color=cmap(colors[c] % 20), label=c)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Footprint KB Val PCA")
        plt.legend(markerscale=1.5, fontsize=8)
        out_path = str(cfg.root / "save_model/prism/footprint_kb_val_pca.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
    except Exception:
        pass

    return {"accuracy_com": acc, "accuracy_spec": acc_s}


if __name__ == "__main__":
    cfg = TestConfig()
    res = evaluate(cfg)
    print(f"Com Accuracy: {res['accuracy_com']:.4f}")
    print(f"Spec Accuracy: {res['accuracy_spec']:.4f}")
