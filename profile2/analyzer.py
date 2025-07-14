#!/usr/bin/env python3
"""
Flash Attention Profiling 结果分析器
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_results, save_results, create_performance_report, format_time
from config import ProfilingConfig

class ProfileResultAnalyzer:
    """Profiling结果分析器"""
    
    def __init__(self, results_file: str, config: ProfilingConfig = None):
        self.results_file = results_file
        self.config = config or ProfilingConfig()
        self.results = load_results(results_file)
        
        if not self.results:
            raise ValueError(f"无法加载结果文件: {results_file}")
    
    def extract_successful_results(self) -> List[Dict[str, Any]]:
        """提取成功的结果"""
        successful = []
        for shape_key, result in self.results.items():
            if result['best_config']['success']:
                data = {
                    'shape_key': shape_key,
                    'batch': result['shape']['batch'],
                    'heads': result['shape']['heads'],
                    'seq_len': result['shape']['seq_len'],
                    'head_dim': result['shape']['head_dim'],
                    'avg_time_ms': result['best_config']['avg_time_ms'],
                    'tflops': result['best_config']['tflops'],
                    'config': result['best_config']['config'],
                    'num_warps': result['best_config']['num_warps']
                }
                successful.append(data)
        return successful
    
    def analyze_performance_by_dimension(self) -> Dict[str, Any]:
        """按维度分析性能"""
        successful_results = self.extract_successful_results()
        
        if not successful_results:
            return {}
        
        df = pd.DataFrame(successful_results)
        
        analysis = {}
        
        # 按batch size分析
        batch_analysis = df.groupby('batch').agg({
            'tflops': ['mean', 'std', 'max', 'min'],
            'avg_time_ms': ['mean', 'std']
        }).round(2)
        analysis['batch_size'] = batch_analysis.to_dict()
        
        # 按heads分析
        heads_analysis = df.groupby('heads').agg({
            'tflops': ['mean', 'std', 'max', 'min'],
            'avg_time_ms': ['mean', 'std']
        }).round(2)
        analysis['num_heads'] = heads_analysis.to_dict()
        
        # 按序列长度分析
        seq_len_analysis = df.groupby('seq_len').agg({
            'tflops': ['mean', 'std', 'max', 'min'],
            'avg_time_ms': ['mean', 'std']
        }).round(2)
        analysis['seq_length'] = seq_len_analysis.to_dict()
        
        # 按head维度分析
        head_dim_analysis = df.groupby('head_dim').agg({
            'tflops': ['mean', 'std', 'max', 'min'],
            'avg_time_ms': ['mean', 'std']
        }).round(2)
        analysis['head_dimension'] = head_dim_analysis.to_dict()
        
        return analysis
    
    def analyze_config_patterns(self) -> Dict[str, Any]:
        """分析配置模式"""
        successful_results = self.extract_successful_results()
        
        if not successful_results:
            return {}
        
        # 统计各配置参数的分布
        config_stats = {}
        for result in successful_results:
            config = result['config']
            for param, value in config.items():
                if param not in config_stats:
                    config_stats[param] = {}
                config_stats[param][value] = config_stats[param].get(value, 0) + 1
        
        # 分析最优配置组合
        config_combinations = {}
        for result in successful_results:
            config = result['config']
            key = f"BM{config['BLOCK_M']}_BN{config['BLOCK_N']}_BUF{config['NUM_BUFFERS']}_WARPS{config['NUM_MMA_WARPS']}_GROUPS{config['NUM_MMA_GROUPS']}"
            if key not in config_combinations:
                config_combinations[key] = {
                    'count': 0,
                    'avg_tflops': 0,
                    'configs': []
                }
            
            config_combinations[key]['count'] += 1
            config_combinations[key]['avg_tflops'] += result['tflops']
            config_combinations[key]['configs'].append(result)
        
        # 计算平均性能
        for key, data in config_combinations.items():
            data['avg_tflops'] /= data['count']
        
        return {
            'parameter_distribution': config_stats,
            'config_combinations': config_combinations
        }
    
    def find_performance_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """找到性能热点"""
        successful_results = self.extract_successful_results()
        
        if not successful_results:
            return []
        
        # 按TFLOPS排序
        sorted_results = sorted(successful_results, key=lambda x: x['tflops'], reverse=True)
        
        hotspots = []
        for i, result in enumerate(sorted_results[:top_n]):
            hotspot = {
                'rank': i + 1,
                'shape': f"B{result['batch']}_H{result['heads']}_S{result['seq_len']}_D{result['head_dim']}",
                'tflops': result['tflops'],
                'avg_time_ms': result['avg_time_ms'],
                'config': result['config'],
                'efficiency_score': self._calculate_efficiency_score(result)
            }
            hotspots.append(hotspot)
        
        return hotspots
    
    def _calculate_efficiency_score(self, result: Dict[str, Any]) -> float:
        """计算效率分数"""
        # 基于TFLOPS和内存使用的综合评分
        tflops = result['tflops']
        
        # 估算内存使用
        shape = result
        memory_usage = self._estimate_memory_usage(
            shape['batch'], shape['heads'], shape['seq_len'], shape['head_dim']
        )
        
        # 简单的效率分数计算
        efficiency = tflops / (memory_usage['total_gb'] + 1)  # 避免除零
        
        return efficiency
    
    def _estimate_memory_usage(self, batch: int, heads: int, seq_len: int, head_dim: int) -> Dict[str, float]:
        """估算内存使用"""
        element_size = 2  # float16
        
        input_size = 3 * batch * heads * seq_len * head_dim * element_size
        output_size = batch * heads * seq_len * head_dim * element_size
        attention_size = batch * heads * seq_len * seq_len * 4  # float32
        
        total_bytes = input_size + output_size + attention_size
        
        return {
            'input_gb': input_size / (1024**3),
            'output_gb': output_size / (1024**3),
            'attention_gb': attention_size / (1024**3),
            'total_gb': total_bytes / (1024**3)
        }
    
    def analyze_scaling_behavior(self) -> Dict[str, Any]:
        """分析扩展性行为"""
        successful_results = self.extract_successful_results()
        
        if not successful_results:
            return {}
        
        df = pd.DataFrame(successful_results)
        
        scaling_analysis = {}
        
        # 序列长度扩展性
        seq_len_scaling = []
        for seq_len in sorted(df['seq_len'].unique()):
            subset = df[df['seq_len'] == seq_len]
            seq_len_scaling.append({
                'seq_len': seq_len,
                'avg_tflops': subset['tflops'].mean(),
                'std_tflops': subset['tflops'].std(),
                'count': len(subset)
            })
        
        scaling_analysis['sequence_length'] = seq_len_scaling
        
        # Batch size扩展性
        batch_scaling = []
        for batch in sorted(df['batch'].unique()):
            subset = df[df['batch'] == batch]
            batch_scaling.append({
                'batch': batch,
                'avg_tflops': subset['tflops'].mean(),
                'std_tflops': subset['tflops'].std(),
                'count': len(subset)
            })
        
        scaling_analysis['batch_size'] = batch_scaling
        
        return scaling_analysis
    
    def generate_visualization(self, output_dir: str):
        """生成可视化图表"""
        successful_results = self.extract_successful_results()
        
        if not successful_results:
            print("没有成功的结果可以可视化")
            return
        
        df = pd.DataFrame(successful_results)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 性能热图
        self._plot_performance_heatmap(df, output_dir)
        
        # 2. 扩展性分析
        self._plot_scaling_analysis(df, output_dir)
        
        # 3. 配置分布
        self._plot_config_distribution(df, output_dir)
        
        # 4. 性能对比
        self._plot_performance_comparison(df, output_dir)
    
    def _plot_performance_heatmap(self, df: pd.DataFrame, output_dir: str):
        """绘制性能热图"""
        plt.figure(figsize=(12, 8))
        
        # 创建序列长度 vs head数量的性能热图
        pivot_data = df.pivot_table(
            values='tflops', 
            index='seq_len', 
            columns='heads', 
            aggfunc='mean'
        )
        
        plt.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        plt.colorbar(label='TFLOPS')
        plt.xlabel('Number of Heads')
        plt.ylabel('Sequence Length')
        plt.title('Flash Attention Performance Heatmap')
        
        # 设置刻度标签
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), pivot_data.index)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300)
        plt.close()
    
    def _plot_scaling_analysis(self, df: pd.DataFrame, output_dir: str):
        """绘制扩展性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 序列长度扩展性
        seq_len_data = df.groupby('seq_len')['tflops'].agg(['mean', 'std']).reset_index()
        axes[0, 0].errorbar(seq_len_data['seq_len'], seq_len_data['mean'], 
                           yerr=seq_len_data['std'], marker='o')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('TFLOPS')
        axes[0, 0].set_title('Performance vs Sequence Length')
        axes[0, 0].grid(True)
        
        # Batch扩展性
        batch_data = df.groupby('batch')['tflops'].agg(['mean', 'std']).reset_index()
        axes[0, 1].errorbar(batch_data['batch'], batch_data['mean'], 
                           yerr=batch_data['std'], marker='o')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('TFLOPS')
        axes[0, 1].set_title('Performance vs Batch Size')
        axes[0, 1].grid(True)
        
        # Head数量扩展性
        heads_data = df.groupby('heads')['tflops'].agg(['mean', 'std']).reset_index()
        axes[1, 0].errorbar(heads_data['heads'], heads_data['mean'], 
                           yerr=heads_data['std'], marker='o')
        axes[1, 0].set_xlabel('Number of Heads')
        axes[1, 0].set_ylabel('TFLOPS')
        axes[1, 0].set_title('Performance vs Number of Heads')
        axes[1, 0].grid(True)
        
        # Head维度扩展性
        head_dim_data = df.groupby('head_dim')['tflops'].agg(['mean', 'std']).reset_index()
        axes[1, 1].errorbar(head_dim_data['head_dim'], head_dim_data['mean'], 
                           yerr=head_dim_data['std'], marker='o')
        axes[1, 1].set_xlabel('Head Dimension')
        axes[1, 1].set_ylabel('TFLOPS')
        axes[1, 1].set_title('Performance vs Head Dimension')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scaling_analysis.png'), dpi=300)
        plt.close()
    
    def _plot_config_distribution(self, df: pd.DataFrame, output_dir: str):
        """绘制配置分布"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        config_params = ['BLOCK_M', 'BLOCK_N', 'NUM_BUFFERS', 'NUM_MMA_WARPS', 'NUM_MMA_GROUPS']
        
        for i, param in enumerate(config_params):
            ax = axes[i // 3, i % 3]
            
            # 提取配置参数
            param_values = [result['config'][param] for result in df.to_dict('records')]
            
            # 绘制直方图
            ax.hist(param_values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(param)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {param}')
            ax.grid(True)
        
        # 删除空的子图
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'config_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_performance_comparison(self, df: pd.DataFrame, output_dir: str):
        """绘制性能对比"""
        plt.figure(figsize=(15, 8))
        
        # 创建性能排序的条形图
        top_20 = df.nlargest(20, 'tflops')
        
        bars = plt.bar(range(len(top_20)), top_20['tflops'])
        plt.xlabel('Configuration Rank')
        plt.ylabel('TFLOPS')
        plt.title('Top 20 Performing Configurations')
        
        # 为每个条形添加标签
        for i, (bar, result) in enumerate(zip(bars, top_20.to_dict('records'))):
            shape_label = f"B{result['batch']}_H{result['heads']}_S{result['seq_len']}_D{result['head_dim']}"
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    shape_label, ha='center', va='bottom', rotation=45, fontsize=8)
        
        plt.xticks(range(len(top_20)), [f'#{i+1}' for i in range(len(top_20))])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
        plt.close()
    
    def generate_comprehensive_report(self, output_file: str):
        """生成综合报告"""
        report_lines = []
        
        # 基本信息
        report_lines.append("Flash Attention Profiling 综合分析报告")
        report_lines.append("="*80)
        report_lines.append(f"结果文件: {self.results_file}")
        report_lines.append(f"总配置数: {len(self.results)}")
        
        successful_results = self.extract_successful_results()
        report_lines.append(f"成功配置数: {len(successful_results)}")
        
        if not successful_results:
            report_lines.append("没有成功的配置，无法生成分析报告")
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            return
        
        # 性能统计
        report_lines.append("\n性能统计:")
        report_lines.append("-"*40)
        tflops_values = [r['tflops'] for r in successful_results]
        report_lines.append(f"平均TFLOPS: {np.mean(tflops_values):.2f}")
        report_lines.append(f"最大TFLOPS: {np.max(tflops_values):.2f}")
        report_lines.append(f"最小TFLOPS: {np.min(tflops_values):.2f}")
        report_lines.append(f"标准差: {np.std(tflops_values):.2f}")
        
        # 性能热点
        report_lines.append("\n性能热点 (Top 10):")
        report_lines.append("-"*40)
        hotspots = self.find_performance_hotspots(10)
        for hotspot in hotspots:
            report_lines.append(f"{hotspot['rank']:2d}. {hotspot['shape']}: {hotspot['tflops']:.2f} TFLOPS")
        
        # 维度分析
        dimension_analysis = self.analyze_performance_by_dimension()
        report_lines.append("\n维度性能分析:")
        report_lines.append("-"*40)
        
        for dimension, data in dimension_analysis.items():
            report_lines.append(f"\n{dimension}:")
            if 'tflops' in data:
                for stat, values in data['tflops'].items():
                    report_lines.append(f"  {stat}: {values}")
        
        # 配置模式
        config_analysis = self.analyze_config_patterns()
        report_lines.append("\n配置模式分析:")
        report_lines.append("-"*40)
        
        if 'parameter_distribution' in config_analysis:
            for param, distribution in config_analysis['parameter_distribution'].items():
                most_common = max(distribution.items(), key=lambda x: x[1])
                total = sum(distribution.values())
                percentage = (most_common[1] / total) * 100
                report_lines.append(f"{param}: {most_common[0]} ({percentage:.1f}%)")
        
        # 扩展性分析
        scaling_analysis = self.analyze_scaling_behavior()
        report_lines.append("\n扩展性分析:")
        report_lines.append("-"*40)
        
        if 'sequence_length' in scaling_analysis:
            report_lines.append("序列长度扩展性:")
            for entry in scaling_analysis['sequence_length']:
                report_lines.append(f"  SeqLen {entry['seq_len']}: {entry['avg_tflops']:.2f} TFLOPS")
        
        # 保存报告
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Flash Attention Profiling Result Analyzer")
    parser.add_argument('--results-file', required=True, help='结果文件路径')
    parser.add_argument('--output-dir', default='analysis_output', help='输出目录')
    parser.add_argument('--generate-report', action='store_true', help='生成综合报告')
    parser.add_argument('--generate-visualizations', action='store_true', help='生成可视化图表')
    parser.add_argument('--compare-configs', action='store_true', help='比较配置')
    parser.add_argument('--analyze-scaling', action='store_true', help='分析扩展性')
    
    args = parser.parse_args()
    
    # 创建分析器
    try:
        analyzer = ProfileResultAnalyzer(args.results_file)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成综合报告
    if args.generate_report:
        report_file = os.path.join(args.output_dir, 'comprehensive_report.txt')
        analyzer.generate_comprehensive_report(report_file)
        print(f"综合报告已保存到: {report_file}")
    
    # 生成可视化
    if args.generate_visualizations:
        try:
            analyzer.generate_visualization(args.output_dir)
            print(f"可视化图表已保存到: {args.output_dir}")
        except ImportError:
            print("警告: matplotlib不可用，无法生成可视化图表")
    
    # 配置比较
    if args.compare_configs:
        config_analysis = analyzer.analyze_config_patterns()
        config_file = os.path.join(args.output_dir, 'config_analysis.json')
        save_results(config_file, config_analysis)
        print(f"配置分析已保存到: {config_file}")
    
    # 扩展性分析
    if args.analyze_scaling:
        scaling_analysis = analyzer.analyze_scaling_behavior()
        scaling_file = os.path.join(args.output_dir, 'scaling_analysis.json')
        save_results(scaling_file, scaling_analysis)
        print(f"扩展性分析已保存到: {scaling_file}")
    
    print("分析完成！")

if __name__ == "__main__":
    main() 