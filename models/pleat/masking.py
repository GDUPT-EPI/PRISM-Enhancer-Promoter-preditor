import torch
from typing import Dict, Tuple


class SegmentMaskBuilder:
    def __init__(self, kernel_size: int, pool_size: int,
                 pad_id: int, cls_id: int, sep_id: int):
        self.K = kernel_size
        self.P = pool_size
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id

    def concat_with_specials(self, enh_ids: torch.Tensor, pr_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = enh_ids.size(0)
        device = enh_ids.device

        cls = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        sep = torch.full((B, 1), self.sep_id, dtype=torch.long, device=device)

        concat_ids = torch.cat([cls, enh_ids, sep, cls, pr_ids, sep], dim=1)

        L = concat_ids.size(1)
        len_en = enh_ids.size(1)
        len_pr = pr_ids.size(1)

        en_start = torch.zeros(B, dtype=torch.long, device=device)
        en_end = torch.full((B,), 1 + len_en - 1, dtype=torch.long, device=device)
        pr_start = torch.full((B,), 1 + len_en + 1, dtype=torch.long, device=device)
        pr_end = torch.full((B,), pr_start[0].item() + len_pr - 1, dtype=torch.long, device=device)

        is_pad = (concat_ids == self.pad_id)
        is_sep = (concat_ids == self.sep_id)

        is_en = torch.zeros(B, L, dtype=torch.bool, device=device)
        is_pr = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            is_en[b, en_start[b]:en_end[b] + 1] = True
            is_en[b, en_start[b]] = True  # include CLS for enhancer segment
            is_pr[b, pr_start[b]:pr_end[b] + 1] = True
            is_pr[b, pr_start[b]] = True  # include CLS for promoter segment

        meta = {
            'is_pad': is_pad,
            'is_sep': is_sep,
            'is_en': is_en,
            'is_pr': is_pr,
            'en_start': en_start,
            'en_end': en_end,
            'pr_start': pr_start,
            'pr_end': pr_end,
            'L_orig': torch.full((B,), L, dtype=torch.long, device=device),
        }

        return concat_ids, meta

    def build_pooled_segment_masks(self, meta: Dict[str, torch.Tensor], pooled_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B = meta['is_pad'].size(0)
        L = meta['is_pad'].size(1)
        is_pad = meta['is_pad']
        is_en = meta['is_en']
        is_pr = meta['is_pr']

        enh_mask = torch.zeros(B, pooled_len, dtype=torch.bool, device=is_pad.device)
        pr_mask = torch.zeros(B, pooled_len, dtype=torch.bool, device=is_pad.device)

        for j in range(pooled_len):
            s = j * self.P
            e = min(s + self.P + self.K - 2, L - 1)
            region_pad_all = is_pad[:, s:e + 1].all(dim=-1)
            region_en_any = is_en[:, s:e + 1].any(dim=-1)
            region_pr_any = is_pr[:, s:e + 1].any(dim=-1)
            enh_mask[:, j] = region_pad_all | (~region_en_any)
            pr_mask[:, j] = region_pad_all | (~region_pr_any)

        return enh_mask, pr_mask

    def build_segment_attention_mask(self, meta: Dict[str, torch.Tensor], pooled_len: int) -> torch.Tensor:
        B = meta['is_pad'].size(0)
        L = meta['is_pad'].size(1)
        is_pad = meta['is_pad']
        is_en = meta['is_en']
        is_pr = meta['is_pr']
        device = is_pad.device

        seg_en = torch.zeros(B, pooled_len, dtype=torch.bool, device=device)
        seg_pr = torch.zeros(B, pooled_len, dtype=torch.bool, device=device)
        seg_pad = torch.zeros(B, pooled_len, dtype=torch.bool, device=device)

        for j in range(pooled_len):
            s = j * self.P
            e = min(s + self.P + self.K - 2, L - 1)
            seg_en[:, j] = is_en[:, s:e + 1].any(dim=-1)
            seg_pr[:, j] = is_pr[:, s:e + 1].any(dim=-1)
            seg_pad[:, j] = is_pad[:, s:e + 1].all(dim=-1)

        same_seg = (seg_en.unsqueeze(-1) & seg_en.unsqueeze(-2)) | (seg_pr.unsqueeze(-1) & seg_pr.unsqueeze(-2))
        same_seg = same_seg.unsqueeze(1)

        attn_mask = torch.zeros(B, 1, pooled_len, pooled_len, device=device, dtype=torch.float32)
        attn_mask.masked_fill_(~same_seg, float('-inf'))

        pad_cols = seg_pad.unsqueeze(1).unsqueeze(-2).expand(B, 1, pooled_len, pooled_len)
        attn_mask.masked_fill_(pad_cols, float('-inf'))

        allowed = same_seg & (~pad_cols)
        row_has_allowed = allowed.any(dim=-1)
        for b in range(B):
            for j in range(pooled_len):
                if not row_has_allowed[b, 0, j]:
                    attn_mask[b, 0, j, j] = 0.0

        return attn_mask


def create_mlm_mask(token_ids: torch.Tensor, mask_prob: float, mask_token_id: int,
                    vocab_size: int, pad_token_id: int, block_mask: bool = False,
                    block_size: int = 3):
    B, L = token_ids.shape
    device = token_ids.device
    masked_ids = token_ids.clone()
    original_ids = token_ids.clone()
    is_valid = (token_ids != pad_token_id) & (token_ids < vocab_size) & (token_ids >= 0)
    if block_mask:
        mask_positions = torch.zeros(B, L, dtype=torch.bool, device=device)
        rand = torch.rand(B, L, device=device)
        start_positions = (rand < mask_prob) & is_valid
        for b in range(B):
            starts = torch.nonzero(start_positions[b], as_tuple=False).flatten()
            for s in starts:
                e = min(s + block_size, L)
                mask_positions[b, s:e] = True
        mask_positions = mask_positions & is_valid
    else:
        rand = torch.rand(B, L, device=device)
        mask_positions = (rand < mask_prob) & is_valid
    mask_rand = torch.rand(B, L, device=device)
    replace_with_mask = mask_positions & (mask_rand < 0.8)
    masked_ids[replace_with_mask] = mask_token_id
    replace_with_random = mask_positions & (mask_rand >= 0.8) & (mask_rand < 0.9)
    random_tokens = torch.randint(1, vocab_size - 1, (B, L), device=device)
    masked_ids[replace_with_random] = random_tokens[replace_with_random]
    return masked_ids, mask_positions, original_ids


def compute_mlm_loss(mlm_logits: torch.Tensor, original_ids: torch.Tensor,
                     mask_positions: torch.Tensor, cnn_kernel_size: int,
                     pool_kernel_size: int):
    import torch.nn.functional as F
    B, L_cnn, V = mlm_logits.shape
    L_orig = original_ids.size(1)
    if L_cnn != L_orig:
        K = cnn_kernel_size
        P = pool_kernel_size
        device = mlm_logits.device
        mask_positions_cnn = torch.zeros(B, L_cnn, dtype=torch.bool, device=device)
        original_ids_cnn = torch.zeros(B, L_cnn, dtype=torch.long, device=device)
        for j in range(L_cnn):
            start = j * P
            end = min(start + P + K - 2, L_orig - 1)
            region_mask = mask_positions[:, start:end+1].any(dim=-1)
            center = start + (end - start) // 2
            original_ids_cnn[:, j] = original_ids[:, center]
            mask_positions_cnn[:, j] = region_mask
        mask_positions = mask_positions_cnn
        original_ids = original_ids_cnn
    mlm_logits_flat = mlm_logits.view(-1, V)
    original_ids_flat = original_ids.view(-1)
    mask_positions_flat = mask_positions.view(-1)
    masked_logits = mlm_logits_flat[mask_positions_flat]
    masked_labels = original_ids_flat[mask_positions_flat]
    if masked_logits.size(0) == 0:
        return torch.tensor(0.0, device=mlm_logits.device, requires_grad=True), 0.0
    loss = F.cross_entropy(masked_logits, masked_labels, label_smoothing=0.1)
    with torch.no_grad():
        predictions = masked_logits.argmax(dim=-1)
        accuracy = (predictions == masked_labels).float().mean().item()
    return loss, accuracy
