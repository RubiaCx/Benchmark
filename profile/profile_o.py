
import torch
import os
import time
import json
import argparse

import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

from triton_attn import attention

from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_v2

class add_ld_library_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.os_environ = os.environ.copy()
        library_path = os.environ.get("LD_LIBRARY_PATH")
        if not library_path:
            os.environ["LD_LIBRARY_PATH"] = self.path
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{library_path}:{self.path}"

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ = self.os_environ.copy()

torch_lib_path = os.path.join(os.path.dirname(__file__), "lib")
with add_ld_library_path(torch_lib_path):
    from flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_v3

def run_single_config_analysis(n_ctx, provider):
    print(f"\n{'='*80}")
    print(f"详细分析: N_CTX={n_ctx}, provider={provider}")
    print(f"{'='*80}")
    
    BATCH, H, HEAD_DIM = 1, 32, 128
    dtype = torch.float16
    device = DEVICE
    
    if "triton" in provider:
        # 创建测试数据
        q = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        
        fn = lambda: attention(q, k, v, False, sm_scale, False)
    elif provider == "flash-v3":
        qkv = torch.randn((BATCH, n_ctx, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_v3(qkv, causal=False)
    elif provider == "flash-v2":
        qkv = torch.randn((BATCH, n_ctx, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_v2(qkv, causal=False)
    elif provider == "torch-spda":
        q = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    ms = triton.testing.do_bench(fn)

    flops = 4 * BATCH * H * n_ctx * n_ctx * HEAD_DIM
    tflops = flops / (ms / 1000) / 1e12
    
    print(f"详细性能分析:")
    print(f"  平均执行时间: {ms:.3f} ms")
    print(f"  TFLOPS: {tflops:.2f}")
    print(f"  FLOPs: {flops/1e12:.2f} T")
    
    # 内存分析
    elem_size = 2  # float16
    total_memory = 4 * BATCH * H * n_ctx * HEAD_DIM * elem_size  # q,k,v,o
    bandwidth_gb_s = total_memory / (ms / 1000) / 1e9
    print(f"  内存带宽: {bandwidth_gb_s:.2f} GB/s")
    print(f"  总内存使用: {total_memory/1e6:.2f} MB")
    
    return {
        'n_ctx': n_ctx,
        'provider': provider,
        'time_ms': ms,
        'tflops': tflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'memory_mb': total_memory/1e6
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention WS Hopper 分析工具")

    parser.add_argument("--n_ctx", type=int, help="序列长度")
    parser.add_argument("--provider", type=str, help="提供者")
    
    args = parser.parse_args()
    
    run_single_config_analysis(args.n_ctx, args.provider)
