
import torch
import os
import time
import json
import argparse

import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

from ws_attn import attention_ws, attention_ws_pp, print_best_config, bench_flash_attention

def run_single_config_analysis(n_ctx, provider):
    print(f"\n{'='*80}")
    print(f"详细分析: N_CTX={n_ctx}, provider={provider}")
    print(f"{'='*80}")
    
    BATCH, H, HEAD_DIM = 1, 32, 128
    dtype = torch.float16
    device = DEVICE
    
    # 创建测试数据
    q = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1.3
    
    # 选择函数
    if provider == "triton-ws-fp16":
        fn = lambda: attention_ws(q, k, v, sm_scale)
    elif provider == "triton-ws-pp-fp16":
        fn = lambda: attention_ws_pp(q, k, v, sm_scale)
    else:
        print(f"未支持的provider: {provider}")
        return
    
    # 预热
    for _ in range(10):
        _ = fn()
    
    # 计时
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = fn()
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / 100
    
    # 打印配置信息
    print_best_config(provider, BATCH, H, n_ctx, HEAD_DIM, avg_time_ms)
    
    # 计算性能指标
    flops = 4 * BATCH * H * n_ctx * n_ctx * HEAD_DIM
    tflops = flops / (avg_time_ms / 1000) / 1e12
    
    print(f"详细性能分析:")
    print(f"  平均执行时间: {avg_time_ms:.3f} ms")
    print(f"  TFLOPS: {tflops:.2f}")
    print(f"  FLOPs: {flops/1e12:.2f} T")
    
    # 内存分析
    elem_size = 2  # float16
    total_memory = 4 * BATCH * H * n_ctx * HEAD_DIM * elem_size  # q,k,v,o
    bandwidth_gb_s = total_memory / (avg_time_ms / 1000) / 1e9
    print(f"  内存带宽: {bandwidth_gb_s:.2f} GB/s")
    print(f"  总内存使用: {total_memory/1e6:.2f} MB")
    
    return {
        'n_ctx': n_ctx,
        'provider': provider,
        'time_ms': avg_time_ms,
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