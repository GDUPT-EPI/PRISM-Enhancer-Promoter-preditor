import torch
import torch.nn as nn
import math
from config import POS_ENCODING_MAX_LEN

# 位置编码
class SinPE(nn.Module):
    def __init__(self, d_model, max_len=POS_ENCODING_MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :, :]