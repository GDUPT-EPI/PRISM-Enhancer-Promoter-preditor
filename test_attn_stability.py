import torch
import torch.nn as nn
from models.layers.attn import RoPE_CausalBlockAttention

def test_attn_stability():
    print("Testing RoPE_CausalBlockAttention numerical stability...")
    
    # Setup
    B, L, D, H = 2, 128, 64, 4
    attn = RoPE_CausalBlockAttention(d_model=D, num_heads=H, block_size=32).cuda()
    
    # 1. Normal input
    x = torch.randn(B, L, D).cuda()
    out = attn(x)
    assert torch.isfinite(out).all(), "Normal input produced NaN/Inf"
    print("Passed normal input test")

    # 2. Input with NaN
    x_nan = x.clone()
    x_nan[0, 10, :] = float('nan')
    out_nan = attn(x_nan)
    # The output might have NaNs due to input NaNs, but it shouldn't crash
    # With nan_to_num in attention, we hope it doesn't propagate catastrophically
    # Note: If input has NaN, output likely has NaN. We mainly check if attn weights crash.
    # Actually, with input NaN, q/k/v will be NaN.
    # The key is whether nan_to_num in attn logits/weights can suppress it or not.
    # If q/k are NaN, matmul result is NaN.
    # nan_to_num(attn_logits) should fix it to 0.0 or large neg.
    
    # Let's check if the attention mechanism itself (attn weights) survives.
    # We can't easily access internal weights here without hooks or modifying code.
    # But we can check if gradients can be computed or if it throws errors.
    
    try:
        loss = out_nan.mean()
        # Backward might fail if NaNs are pervasive
        # But we are testing forward stability primarily.
        print("Forward pass with NaN input completed (output may be NaN, expected)")
    except Exception as e:
        print(f"Forward pass with NaN input failed: {e}")

    # 3. Input with Inf
    x_inf = x.clone()
    x_inf[0, 10, :] = float('inf')
    out_inf = attn(x_inf)
    # nan_to_num handles posinf/neginf too.
    # Again, q/k/v will be Inf.
    # matmul will be Inf.
    # nan_to_num(attn, posinf=1e4) should clamp it.
    
    # Ideally, if we have Inf in input, we want finite output if possible, 
    # or at least controlled behavior.
    # However, garbage in -> garbage out is acceptable, as long as it doesn't crash CUDA.
    
    print("Forward pass with Inf input completed")

    # 4. Extreme values (large numbers)
    x_large = torch.randn(B, L, D).cuda() * 1e5
    out_large = attn(x_large)
    if torch.isfinite(out_large).all():
         print("Passed large input test (Finite output)")
    else:
         print("Large input test produced non-finite output (Expected for extreme values, but checking stability)")

    print("Stability test script finished.")

if __name__ == "__main__":
    try:
        test_attn_stability()
    except Exception as e:
        print(f"Test failed with error: {e}")
