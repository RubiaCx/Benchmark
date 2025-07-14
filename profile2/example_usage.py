#!/usr/bin/env python3
"""
Flash Attention Profiling 使用示例
"""

import os
import sys
import torch

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ProfilingConfig, QUICK_CONFIG, FULL_CONFIG
from profiler import FlashAttentionProfiler
from runner import ProfilingRunner
from analyzer import ProfileResultAnalyzer
from utils import create_performance_report

def example_quick_profiling():
    """示例1: 快速profiling"""
    print("=== 示例1: 快速profiling ===")
    
    # 使用快速配置
    config = QUICK_CONFIG
    profiler = FlashAttentionProfiler(config)
    
    shape = {'batch': 1, 'heads': 8, 'seq_len': 512, 'head_dim': 64}
    
    print(f"测试形状: {shape}")
    result = profiler.find_best_config(shape)
    
    if result and result['best_config']['success']:
        print(f"最优配置: {result['best_config']['config']}")
        print(f"性能: {result['best_config']['tflops']:.2f} TFLOPS")
        print(f"时间: {result['best_config']['avg_time_ms']:.2f} ms")
    else:
        print("测试失败")

def example_custom_shape_profiling():
    """示例2: 自定义形状profiling"""
    print("\n=== 示例2: 自定义形状profiling ===")
    
    # 自定义配置
    config = ProfilingConfig(
        batch_sizes=[2, 4],
        num_heads=[16, 32],
        seq_lengths=[1024, 2048],
        head_dims=[128],
        warmup_trials=5,
        benchmark_trials=20
    )
    
    runner = ProfilingRunner(config)
    
    # 运行单个形状测试
    result = runner.run_shape_profiling(batch=2, heads=16, seq_len=1024, head_dim=128)
    
    if result:
        print("形状profiling成功!")
        print(f"最优性能: {result['best_config']['tflops']:.2f} TFLOPS")
    else:
        print("形状profiling失败")

def example_full_profiling():
    """示例3: 完整profiling"""
    print("\n=== 示例3: 完整profiling ===")
    
    # 使用完整配置（但减少形状数量用于演示）
    config = ProfilingConfig(
        batch_sizes=[1, 2, 4],
        num_heads=[8, 16],
        seq_lengths=[512, 1024],
        head_dims=[64, 128],
        warmup_trials=10,
        benchmark_trials=50,
        save_best_configs=True
    )
    
    runner = ProfilingRunner(config)
    
    print("运行完整profiling...")
    results = runner.run_full_profiling()
    
    if results:
        print(f"完整profiling完成，测试了 {len(results)} 个形状")
        
        # 生成性能报告
        report = create_performance_report(results)
        print("\n性能报告预览:")
        print(report[:500] + "...")  # 只显示前500个字符
    else:
        print("完整profiling失败")

def example_result_analysis():
    """示例4: 结果分析"""
    print("\n=== 示例4: 结果分析 ===")
    
    # 假设已有结果文件
    results_file = "results/full_profiling_results.json"
    
    if os.path.exists(results_file):
        # 创建分析器
        analyzer = ProfileResultAnalyzer(results_file)
        
        # 分析性能热点
        hotspots = analyzer.find_performance_hotspots(5)
        print("性能热点 (Top 5):")
        for hotspot in hotspots:
            print(f"  {hotspot['rank']}. {hotspot['shape']}: {hotspot['tflops']:.2f} TFLOPS")
        
        # 分析配置模式
        config_patterns = analyzer.analyze_config_patterns()
        print("\n配置模式分析:")
        if 'parameter_distribution' in config_patterns:
            for param, distribution in config_patterns['parameter_distribution'].items():
                most_common = max(distribution.items(), key=lambda x: x[1])
                print(f"  {param}: {most_common[0]} (最常用)")
        
        # 生成综合报告
        analyzer.generate_comprehensive_report("analysis_output/example_report.txt")
        print("\n综合报告已生成: analysis_output/example_report.txt")
    else:
        print(f"结果文件不存在: {results_file}")
        print("请先运行完整profiling")

def example_flash_attention_adapter():
    """示例5: Flash Attention适配器"""
    print("\n=== 示例5: Flash Attention适配器 ===")
    
    try:
        from flash_attention_adapter import FlashAttentionAdapter
        
        # 创建适配器
        adapter = FlashAttentionAdapter()
        
        # 测试数据
        batch, heads, seq_len, head_dim = 1, 8, 512, 64
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        sm_scale = 0.5
        
        # 测试自定义配置
        custom_config = {
            'BLOCK_M': 64,
            'BLOCK_N': 64,
            'NUM_BUFFERS': 2,
            'NUM_MMA_WARPS': 4,
            'NUM_MMA_GROUPS': 1
        }
        
        # 基准测试
        shape = (batch, heads, seq_len, head_dim)
        result = adapter.benchmark_config(custom_config, shape)
        
        if result['success']:
            print(f"自定义配置测试成功!")
            print(f"配置: {result['config']}")
            print(f"性能: {result['tflops']:.2f} TFLOPS")
            print(f"时间: {result['avg_time_ms']:.2f} ms")
        else:
            print("自定义配置测试失败")
            
    except Exception as e:
        print(f"Flash Attention适配器测试失败: {e}")
        print("请确保原始flash attention文件存在")

def main():
    """主函数"""
    print("Flash Attention Profiling 使用示例")
    print("=" * 50)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，某些示例可能无法运行")
        return
    
    print("✅ CUDA可用，开始运行示例...")
    
    try:
        # 运行示例
        example_quick_profiling()
        example_custom_shape_profiling()
        # example_full_profiling()  # 注释掉，因为耗时较长
        example_result_analysis()
        example_flash_attention_adapter()
        
        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成!")
        print("\n要运行完整的profiling，请使用:")
        print("  python runner.py --mode full --config full")
        print("\n要使用自动化脚本，请运行:")
        print("  bash scripts/quick_start.sh")
        print("  bash scripts/run_all_tests.sh")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        print("请检查环境配置和依赖安装")

if __name__ == "__main__":
    main() 