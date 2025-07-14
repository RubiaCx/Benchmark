#!/usr/bin/env python3
"""
Flash Attention Profiling 自动化运行器
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, Optional

# 添加父目录到path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ProfilingConfig, QUICK_CONFIG, FULL_CONFIG, LARGE_CONFIG
from profiler import FlashAttentionProfiler
from utils import (
    setup_logging, save_results, load_results, 
    create_performance_report, check_system_requirements,
    export_to_csv, format_time, format_memory_size
)

class ProfilingRunner:
    """Profiling运行器"""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.profiler = FlashAttentionProfiler(config)
        self.logger = setup_logging(config.get_output_paths()['logs'])
    
    def run_system_check(self):
        """运行系统检查"""
        self.logger.info("Running system requirements check...")
        requirements = check_system_requirements()
        
        print("\n" + "="*60)
        print("系统需求检查")
        print("="*60)
        
        status_items = [
            ("CUDA", requirements['cuda_available']),
            ("Triton", requirements['triton_available']),
            ("NSys", requirements['nsys_available']),
            ("NCU", requirements['ncu_available'])
        ]
        
        for name, available in status_items:
            status = "✅" if available else "❌"
            print(f"{status} {name}: {'可用' if available else '不可用'}")
        
        if requirements['device_info']:
            device = requirements['device_info']
            print(f"\n设备信息:")
            print(f"  GPU: {device['name']}")
            print(f"  计算能力: {device['compute_capability']}")
            print(f"  内存: {device['memory_gb']:.1f} GB")
        
        # 检查关键依赖
        if not requirements['cuda_available']:
            self.logger.error("CUDA不可用，无法运行profiling")
            return False
        
        if not requirements['triton_available']:
            self.logger.error("Triton不可用，无法运行profiling")
            return False
        
        self.logger.info("系统检查通过")
        return True
    
    def run_quick_test(self):
        """运行快速测试"""
        self.logger.info("Running quick test...")
        
        # 使用小形状进行快速测试
        test_shape = {'batch': 1, 'heads': 8, 'seq_len': 512, 'head_dim': 64}
        
        try:
            result = self.profiler.find_best_config(test_shape)
            if result and result['best_config']['success']:
                self.logger.info(f"快速测试通过: {result['best_config']['tflops']:.2f} TFLOPS")
                return True
            else:
                self.logger.error("快速测试失败")
                return False
        except Exception as e:
            self.logger.error(f"快速测试异常: {str(e)}")
            return False
    
    def run_full_profiling(self):
        """运行完整的profiling"""
        self.logger.info("Starting full profiling...")
        
        start_time = time.time()
        results = self.profiler.profile_all_shapes()
        end_time = time.time()
        
        total_time = end_time - start_time
        self.logger.info(f"Full profiling completed in {format_time(total_time * 1000)}")
        
        # 保存结果
        paths = self.config.get_output_paths()
        results_file = os.path.join(paths['base'], 'full_profiling_results.json')
        save_results(results_file, results)
        
        # 生成报告
        report = create_performance_report(results)
        report_file = os.path.join(paths['base'], 'performance_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        # 导出CSV
        csv_file = os.path.join(paths['base'], 'results.csv')
        export_to_csv(results, csv_file)
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Report saved to {report_file}")
        self.logger.info(f"CSV exported to {csv_file}")
        
        return results
    
    def run_shape_profiling(self, batch: int, heads: int, seq_len: int, head_dim: int):
        """运行单个形状的profiling"""
        shape = {
            'batch': batch,
            'heads': heads,
            'seq_len': seq_len,
            'head_dim': head_dim
        }
        
        self.logger.info(f"Profiling shape: {shape}")
        
        try:
            result = self.profiler.find_best_config(shape)
            if result:
                paths = self.config.get_output_paths()
                shape_key = f"b{batch}_h{heads}_s{seq_len}_d{head_dim}"
                result_file = os.path.join(paths['base'], f"{shape_key}_detailed.json")
                save_results(result_file, result)
                
                self.logger.info(f"Shape profiling completed. Results saved to {result_file}")
                return result
            else:
                self.logger.error("Shape profiling failed")
                return None
        except Exception as e:
            self.logger.error(f"Shape profiling failed: {str(e)}")
            return None
    
    def run_nsys_profiling(self, config_file: Optional[str] = None):
        """运行NSys profiling"""
        self.logger.info("Starting NSys profiling...")
        
        if config_file:
            config_data = load_results(config_file)
            if not config_data:
                self.logger.error(f"Failed to load config from {config_file}")
                return False
            
            shape = config_data['shape']
        else:
            # 使用默认形状
            shape = {'batch': 4, 'heads': 16, 'seq_len': 1024, 'head_dim': 128}
        
        output_file = self.profiler.run_nsys_profiling(shape, config_file, self.config.nsys_duration)
        
        if output_file:
            self.logger.info(f"NSys profiling completed: {output_file}")
            return True
        else:
            self.logger.error("NSys profiling failed")
            return False
    
    def run_ncu_profiling(self, config_file: Optional[str] = None):
        """运行NCU profiling"""
        self.logger.info("Starting NCU profiling...")
        
        if config_file:
            config_data = load_results(config_file)
            if not config_data:
                self.logger.error(f"Failed to load config from {config_file}")
                return False
            
            shape = config_data['shape']
        else:
            # 使用默认形状
            shape = {'batch': 4, 'heads': 16, 'seq_len': 1024, 'head_dim': 128}
        
        output_file = self.profiler.run_ncu_profiling(shape, config_file, self.config.ncu_kernels)
        
        if output_file:
            self.logger.info(f"NCU profiling completed: {output_file}")
            return True
        else:
            self.logger.error("NCU profiling failed")
            return False
    
    def generate_analysis_report(self, results_file: str):
        """生成详细的分析报告"""
        self.logger.info(f"Generating analysis report from {results_file}")
        
        results = load_results(results_file)
        if not results:
            self.logger.error(f"Failed to load results from {results_file}")
            return False
        
        report = create_performance_report(results)
        
        # 添加详细分析
        report += "\n\n"
        report += "详细分析:\n"
        report += "="*60 + "\n"
        
        # 性能趋势分析
        report += self._analyze_performance_trends(results)
        
        # 配置建议
        report += "\n\n配置建议:\n"
        report += "-"*40 + "\n"
        report += self._generate_config_recommendations(results)
        
        # 保存报告
        paths = self.config.get_output_paths()
        report_file = os.path.join(paths['base'], 'detailed_analysis_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Analysis report saved to {report_file}")
        return True
    
    def _analyze_performance_trends(self, results: Dict[str, Any]) -> str:
        """分析性能趋势"""
        lines = []
        
        # 按序列长度分组分析
        seq_len_groups = {}
        for shape_key, result in results.items():
            if result['best_config']['success']:
                seq_len = result['shape']['seq_len']
                if seq_len not in seq_len_groups:
                    seq_len_groups[seq_len] = []
                seq_len_groups[seq_len].append(result['best_config']['tflops'])
        
        lines.append("序列长度性能分析:")
        for seq_len in sorted(seq_len_groups.keys()):
            tflops_list = seq_len_groups[seq_len]
            avg_tflops = sum(tflops_list) / len(tflops_list)
            lines.append(f"  SeqLen {seq_len}: 平均 {avg_tflops:.2f} TFLOPS ({len(tflops_list)} 个配置)")
        
        return "\n".join(lines)
    
    def _generate_config_recommendations(self, results: Dict[str, Any]) -> str:
        """生成配置建议"""
        lines = []
        
        # 统计最优配置参数
        param_stats = {}
        for result in results.values():
            if result['best_config']['success']:
                config = result['best_config']['config']
                for param, value in config.items():
                    if param not in param_stats:
                        param_stats[param] = {}
                    param_stats[param][value] = param_stats[param].get(value, 0) + 1
        
        lines.append("基于统计的参数建议:")
        for param, value_counts in param_stats.items():
            most_common = max(value_counts.items(), key=lambda x: x[1])
            total = sum(value_counts.values())
            percentage = (most_common[1] / total) * 100
            lines.append(f"  {param}: {most_common[0]} (在 {percentage:.1f}% 的最优配置中使用)")
        
        return "\n".join(lines)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Flash Attention Profiling Runner")
    parser.add_argument('--mode', choices=['check', 'quick', 'full', 'shape', 'nsys', 'ncu', 'analyze'], 
                       default='check', help='运行模式')
    parser.add_argument('--config', choices=['quick', 'full', 'large'], default='full', 
                       help='使用的配置')
    parser.add_argument('--batch', type=int, help='Batch size (for shape mode)')
    parser.add_argument('--heads', type=int, help='Number of heads (for shape mode)')
    parser.add_argument('--seq-len', type=int, help='Sequence length (for shape mode)')
    parser.add_argument('--head-dim', type=int, help='Head dimension (for shape mode)')
    parser.add_argument('--config-file', help='Config file path (for nsys/ncu mode)')
    parser.add_argument('--results-file', help='Results file path (for analyze mode)')
    parser.add_argument('--save-configs', action='store_true', help='保存最优配置')
    
    args = parser.parse_args()
    
    # 选择配置
    config_map = {
        'quick': QUICK_CONFIG,
        'full': FULL_CONFIG,
        'large': LARGE_CONFIG
    }
    
    config = config_map[args.config]
    if args.save_configs:
        config.save_best_configs = True
    
    # 创建运行器
    runner = ProfilingRunner(config)
    
    # 运行相应模式
    if args.mode == 'check':
        success = runner.run_system_check()
        if success:
            print("\n✅ 系统检查通过，可以开始profiling")
        else:
            print("\n❌ 系统检查失败，请检查环境配置")
            sys.exit(1)
    
    elif args.mode == 'quick':
        if not runner.run_system_check():
            sys.exit(1)
        success = runner.run_quick_test()
        if success:
            print("\n✅ 快速测试通过")
        else:
            print("\n❌ 快速测试失败")
            sys.exit(1)
    
    elif args.mode == 'full':
        if not runner.run_system_check():
            sys.exit(1)
        print("\n🚀 开始完整profiling...")
        results = runner.run_full_profiling()
        if results:
            print(f"\n✅ 完整profiling完成，测试了 {len(results)} 个形状")
        else:
            print("\n❌ 完整profiling失败")
            sys.exit(1)
    
    elif args.mode == 'shape':
        if not all([args.batch, args.heads, args.seq_len, args.head_dim]):
            print("❌ Shape模式需要指定 --batch, --heads, --seq-len, --head-dim")
            sys.exit(1)
        
        if not runner.run_system_check():
            sys.exit(1)
        
        result = runner.run_shape_profiling(args.batch, args.heads, args.seq_len, args.head_dim)
        if result:
            print(f"\n✅ 形状profiling完成")
            if result['best_config']['success']:
                print(f"最优性能: {result['best_config']['tflops']:.2f} TFLOPS")
        else:
            print("\n❌ 形状profiling失败")
            sys.exit(1)
    
    elif args.mode == 'nsys':
        if not runner.run_system_check():
            sys.exit(1)
        success = runner.run_nsys_profiling(args.config_file)
        if success:
            print("\n✅ NSys profiling完成")
        else:
            print("\n❌ NSys profiling失败")
            sys.exit(1)
    
    elif args.mode == 'ncu':
        if not runner.run_system_check():
            sys.exit(1)
        success = runner.run_ncu_profiling(args.config_file)
        if success:
            print("\n✅ NCU profiling完成")
        else:
            print("\n❌ NCU profiling失败")
            sys.exit(1)
    
    elif args.mode == 'analyze':
        if not args.results_file:
            print("❌ Analyze模式需要指定 --results-file")
            sys.exit(1)
        
        success = runner.generate_analysis_report(args.results_file)
        if success:
            print("\n✅ 分析报告生成完成")
        else:
            print("\n❌ 分析报告生成失败")
            sys.exit(1)

if __name__ == "__main__":
    main() 