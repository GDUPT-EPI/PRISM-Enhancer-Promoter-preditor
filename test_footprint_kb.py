import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from pathlib import Path

from models.pleat.embedding import KMerTokenizer, create_dna_embedding_layer
from models.layers.footprint import FootprintExpert
from data_loader import load_prism_data, PRISMDataset


class TestConfig:
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 尝试自动查找最新的KB
        kb_dir = self.root / "save_model/prism"
        kbs = sorted(list(kb_dir.glob("footprint_kb_epoch_*.pt")), key=os.path.getmtime)
        if kbs:
            self.kb_path = str(kbs[-1])
        else:
            self.kb_path = str(self.root / "save_model/prism/footprint_kb_epoch_1.pt")
            
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


def compute_cell_means(split: str, cfg: TestConfig, extractor: EnhancerExtractor, footprint: FootprintExpert) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    pairs_df, e_seqs, p_seqs = load_prism_data(split)
    ds = PRISMDataset(pairs_df, e_seqs, p_seqs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(b, cfg))
    
    sums_com: Dict[str, torch.Tensor] = {}
    sums_spec: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    
    with torch.no_grad():
        for enh_ids, prom_ids, cells, labels in dl:
            enh_ids = enh_ids.to(cfg.device)
            feats = extractor(enh_ids)
            # Expert forward: seq_out, sample_vec, z_com, z_spec
            _, _, z_com, z_spec = footprint(feats)
            
            for i, c in enumerate(cells):
                v_com = z_com[i].detach()
                v_spec = z_spec[i].detach()
                
                if c not in counts:
                    sums_com[c] = torch.zeros_like(v_com)
                    sums_spec[c] = torch.zeros_like(v_spec)
                    counts[c] = 0
                
                sums_com[c] += v_com
                sums_spec[c] += v_spec
                counts[c] += 1
                
    means_com = {c: (sums_com[c] / max(counts[c], 1)).to(cfg.device) for c in sums_com}
    means_spec = {c: (sums_spec[c] / max(counts[c], 1)).to(cfg.device) for c in sums_spec}
    return means_com, means_spec


def evaluate(cfg: TestConfig) -> Dict[str, float]:
    print(f"Loading KB from: {cfg.kb_path}")
    kb = load_kb(cfg.kb_path)
    
    # 加载中心点
    com_names_all, com_centers_all = centers_to_tensor(kb["com_centers"], cfg.device)
    spec_names_all, spec_centers_all = centers_to_tensor(kb["spec_centers"], cfg.device)

    extractor = EnhancerExtractor(cfg).to(cfg.device)
    
    # 初始化专家并加载权重
    footprint = FootprintExpert(d_model=cfg.out_channels).to(cfg.device)
    
    if 'model_state' in kb and kb['model_state'] is not None:
        print("Loading expert weights from KB...")
        # 过滤出 enhancer_footprint 相关的权重
        expert_dict = {k.replace("enhancer_footprint.", ""): v 
                       for k, v in kb['model_state'].items() 
                       if k.startswith("enhancer_footprint.")}
        footprint.load_state_dict(expert_dict, strict=False)
    else:
        print("WARNING: No model state found in KB, using random weights!")

    extractor.eval()
    footprint.eval()

    # 不再需要 solve_map，因为我们直接有投影层
    # 我们直接评估验证集上的表现

    pairs_df, e_seqs, p_seqs = load_prism_data("val")
    ds = PRISMDataset(pairs_df, e_seqs, p_seqs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(b, cfg))

    total = 0
    correct_com = 0
    correct_spec = 0
    
    all_z_spec = []
    all_cell = []

    with torch.no_grad():
        for enh_ids, prom_ids, cells, labels in dl:
            enh_ids = enh_ids.to(cfg.device)
            feats = extractor(enh_ids)
            
            # 直接通过专家头获得投影
            _, _, z_com, z_spec = footprint(feats)
            
            # 分类: 找最近的中心
            preds_com = classify(z_com, com_names_all, com_centers_all)
            preds_spec = classify(z_spec, spec_names_all, spec_centers_all)

            total += len(cells)
            correct_com += sum(1 for p, g in zip(preds_com, cells) if p == g)
            correct_spec += sum(1 for p, g in zip(preds_spec, cells) if p == g)
            
            all_z_spec.append(z_spec.cpu())
            all_cell.extend(cells)

    acc_com = correct_com / max(total, 1)
    acc_spec = correct_spec / max(total, 1)

    try:
        import matplotlib.pyplot as plt
        X = torch.cat(all_z_spec, dim=0).numpy()
        # PCA可视化特异性空间
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = torch.linalg.svd(torch.tensor(Xc), full_matrices=False)
        PCs = (torch.tensor(Xc) @ Vt[:, :2]).numpy()
        
        uniq = sorted(set(all_cell))
        colors = {c: i for i, c in enumerate(uniq)}
        cmap = plt.get_cmap("tab20")
        
        plt.figure(figsize=(10, 8))
        for c in uniq:
            idxs = [i for i, cc in enumerate(all_cell) if cc == c]
            if not idxs: continue
            pts = PCs[idxs]
            plt.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.7, color=cmap(colors[c] % 20), label=c)
            
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Footprint Expert Spec-Space PCA (Acc: {acc_spec:.2%})")
        plt.legend(markerscale=1.5, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        out_path = str(cfg.root / "save_model/prism/footprint_expert_val_pca.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved PCA plot to {out_path}")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

    return {"accuracy_com": acc_com, "accuracy_spec": acc_spec}


if __name__ == "__main__":
    cfg = TestConfig()
    res = evaluate(cfg)
    print(f"Com Accuracy: {res['accuracy_com']:.4f}")
    print(f"Spec Accuracy: {res['accuracy_spec']:.4f}")
