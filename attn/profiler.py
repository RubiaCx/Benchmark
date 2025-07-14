
import torch
import os
import time
import json
import argparse

import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# from ws_attn import attention_ws, attention_ws_pp, print_best_config, bench_flash_attention

from attn_triton import attention

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
    elif provider == "triton-fp16":
        fn = lambda: attention(q, k, v, False, sm_scale, False)
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


def run_single_analysis(n_ctx, provider):
    BATCH, H, HEAD_DIM = 1, 32, 128
    dtype = torch.float16
    device = DEVICE
    
    q = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, n_ctx, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1.3
    
    if provider == "triton-ws-fp16":
        fn = lambda: attention_ws(q, k, v, sm_scale)
    elif provider == "triton-ws-pp-fp16":
        fn = lambda: attention_ws_pp(q, k, v, sm_scale)
    elif provider == "triton-fp16":
        fn = lambda: attention(q, k, v, False, sm_scale, False)
    else:
        print(f"未支持的provider: {provider}")
        return
    
    # 预热
    for _ in range(10):
        _ = fn()

def run_comprehensive_analysis():
    """运行综合对比分析"""
    print("\n" + "="*100)
    print("综合对比分析")
    print("="*100)
    
    configs = [128, 8192]
    providers = ["triton-ws-fp16", "triton-ws-pp-fp16", "triton-fp16"]
    
    results = []
    
    for n_ctx in configs:
        for provider in providers:
            result = run_single_config_analysis(n_ctx, provider)
            results.append(result)
    
    # 生成对比报告
    print("\n" + "="*100)
    print("对比报告")
    print("="*100)
    
    print(f"{'Provider':<20} {'N_CTX':<10} {'Time(ms)':<12} {'TFLOPS':<10} {'Bandwidth(GB/s)':<15} {'Memory(MB)':<12}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['provider']:<20} {result['n_ctx']:<10} {result['time_ms']:<12.3f} {result['tflops']:<10.2f} {result['bandwidth_gb_s']:<15.2f} {result['memory_mb']:<12.2f}")
    
    # 性能差异分析
    print("\n性能差异分析:")
    print("-" * 50)
    
    # 比较相同N_CTX下不同provider的性能
    for n_ctx in configs:
        ws_result = next((r for r in results if r['n_ctx'] == n_ctx and r['provider'] == 'triton-ws-fp16'), None)
        pp_result = next((r for r in results if r['n_ctx'] == n_ctx and r['provider'] == 'triton-ws-pp-fp16'), None)
        
        if ws_result and pp_result:
            speedup = ws_result['time_ms'] / pp_result['time_ms']
            print(f"N_CTX={n_ctx}: Pipeline版本相对基础版本的加速比: {speedup:.2f}x")
    
    # 比较不同N_CTX下的扩展性
    for provider in providers:
        small_result = next((r for r in results if r['provider'] == provider and r['n_ctx'] == 128), None)
        large_result = next((r for r in results if r['provider'] == provider and r['n_ctx'] == 8192), None)
        
        if small_result and large_result:
            efficiency = (large_result['tflops'] / small_result['tflops']) / (8192 / 128)
            print(f"{provider}: 长序列扩展效率: {efficiency:.2f}")
    
    # 保存结果到JSON
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n详细结果已保存到 analysis_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention WS Hopper 分析工具")
    parser.add_argument("--mode", choices=["benchmark", "nvtx", "ncu", "analysis"], 
                       default="benchmark", help="运行模式")
    parser.add_argument("--n_ctx", type=int, help="序列长度")
    parser.add_argument("--provider", type=str, help="提供者")
    
    args = parser.parse_args()
    
    if args.mode == "analysis":
        run_comprehensive_analysis()
    elif args.mode == "ncu" and args.n_ctx and args.provider:
        run_single_analysis(args.n_ctx, args.provider)
    elif args.mode == "nvtx":
        os.environ['NVTX_ENABLE'] = '1'
        bench_flash_attention.run(save_path=".", print_data=True)
    else:
        bench_flash_attention.run(save_path=".", print_data=True)
