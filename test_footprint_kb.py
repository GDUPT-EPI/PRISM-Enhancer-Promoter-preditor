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
        self.batch_size = 16
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


def evaluate(cfg: TestConfig) -> Dict[str, float]:
    pairs_df, e_seqs, p_seqs = load_prism_data("val")
    ds = PRISMDataset(pairs_df, e_seqs, p_seqs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(b, cfg))

    kb = load_kb(cfg.kb_path)
    com_names, com_centers = centers_to_tensor(kb["com_centers"], cfg.device)

    extractor = EnhancerExtractor(cfg).to(cfg.device)
    footprint = LCWnetFootprint(d_model=cfg.out_channels).to(cfg.device)
    extractor.eval()
    footprint.eval()

    total = 0
    correct = 0
    all_fp = []
    all_cell = []

    with torch.no_grad():
        for enh_ids, prom_ids, cells, labels in dl:
            enh_ids = enh_ids.to(cfg.device)
            feats = extractor(enh_ids)
            fp = footprint.forward_vector(feats)
            preds = classify(fp, com_names, com_centers)

            total += len(cells)
            correct += sum(1 for p, g in zip(preds, cells) if p == g)
            all_fp.append(fp.cpu())
            all_cell.extend(cells)

    acc = correct / max(total, 1)

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

    return {"accuracy": acc}


if __name__ == "__main__":
    cfg = TestConfig()
    res = evaluate(cfg)
    print(f"Accuracy: {res['accuracy']:.4f}")
