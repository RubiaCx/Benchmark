"""
Flash Attention Profiling 工具函数
"""

import os
import json
import logging
import datetime
import pickle
import numpy as np
import torch
from typing import Dict, Any, Optional

def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"profiling_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('FlashAttentionProfiler')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def save_results(file_path: str, data: Dict[str, Any]):
    """保存结果到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 根据文件扩展名选择保存格式
    if file_path.endswith('.json'):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        # 默认使用JSON
        with open(file_path + '.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)

def load_results(file_path: str) -> Optional[Dict[str, Any]]:
    """从文件加载结果"""
    if not os.path.exists(file_path):
        return None
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            # 尝试JSON
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load results from {file_path}: {e}")
        return None

def format_memory_size(size_bytes: int) -> str:
    """格式化内存大小"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"

def format_time(time_ms: float) -> str:
    """格式化时间"""
    if time_ms >= 1000:
        return f"{time_ms / 1000:.2f} s"
    elif time_ms >= 1:
        return f"{time_ms:.2f} ms"
    else:
        return f"{time_ms * 1000:.2f} μs"

def calculate_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """计算Flash Attention的理论FLOPS"""
    # 简化的FLOPS计算
    # Q @ K^T: batch * heads * seq_len * seq_len * head_dim
    # softmax: batch * heads * seq_len * seq_len
    # P @ V: batch * heads * seq_len * seq_len * head_dim
    
    qk_flops = batch * heads * seq_len * seq_len * head_dim
    softmax_flops = batch * heads * seq_len * seq_len * 5  # 近似
    pv_flops = batch * heads * seq_len * seq_len * head_dim
    
    return qk_flops + softmax_flops + pv_flops

def calculate_memory_bandwidth(data_size_bytes: int, time_ms: float) -> float:
    """计算内存带宽 (GB/s)"""
    time_s = time_ms / 1000
    bandwidth_gb_s = (data_size_bytes / (1024**3)) / time_s
    return bandwidth_gb_s

def validate_attention_output(output: torch.Tensor, expected_shape: tuple) -> bool:
    """验证attention输出的正确性"""
    if output.shape != expected_shape:
        return False
    
    # 检查是否有NaN或Inf
    if torch.isnan(output).any() or torch.isinf(output).any():
        return False
    
    # 检查数值范围是否合理
    if output.abs().max() > 100:  # 过大的值
        return False
    
    return True

def generate_shape_matrix(shapes: list) -> str:
    """生成形状矩阵的文本表示"""
    lines = []
    lines.append("Shape Configuration Matrix:")
    lines.append("=" * 50)
    
    for shape in shapes:
        memory_info = estimate_memory_usage(**shape)
        lines.append(
            f"Batch: {shape['batch']:2d}, Heads: {shape['heads']:2d}, "
            f"SeqLen: {shape['seq_len']:4d}, HeadDim: {shape['head_dim']:3d} "
            f"-> {memory_info['total_gb']:.2f} GB"
        )
    
    return "\n".join(lines)

def create_performance_report(results: Dict[str, Any]) -> str:
    """生成性能报告"""
    lines = []
    lines.append("Flash Attention Performance Report")
    lines.append("=" * 60)
    lines.append(f"Total shapes tested: {len(results)}")
    lines.append("")
    
    # 按性能排序
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['best_config']['tflops'] if x[1]['best_config']['success'] else 0,
        reverse=True
    )
    
    lines.append("Top 10 Performing Configurations:")
    lines.append("-" * 40)
    
    for i, (shape_key, result) in enumerate(sorted_results[:10]):
        if result['best_config']['success']:
            shape = result['shape']
            config = result['best_config']
            lines.append(
                f"{i+1:2d}. {shape_key}: {config['tflops']:.2f} TFLOPS, "
                f"{config['avg_time_ms']:.2f}ms "
                f"(B={shape['batch']}, H={shape['heads']}, S={shape['seq_len']}, D={shape['head_dim']})"
            )
    
    lines.append("")
    lines.append("Configuration Analysis:")
    lines.append("-" * 40)
    
    # 分析最优配置的参数分布
    successful_configs = [
        result['best_config']['config']
        for result in results.values()
        if result['best_config']['success']
    ]
    
    if successful_configs:
        # 统计各参数的分布
        param_stats = {}
        for config in successful_configs:
            for param, value in config.items():
                if param not in param_stats:
                    param_stats[param] = {}
                param_stats[param][value] = param_stats[param].get(value, 0) + 1
        
        for param, value_counts in param_stats.items():
            most_common = max(value_counts.items(), key=lambda x: x[1])
            lines.append(f"{param}: {most_common[0]} (used in {most_common[1]} configs)")
    
    return "\n".join(lines)

def check_system_requirements() -> Dict[str, Any]:
    """检查系统需求"""
    requirements = {
        'cuda_available': torch.cuda.is_available(),
        'triton_available': False,
        'nsys_available': False,
        'ncu_available': False,
        'device_info': None
    }
    
    # 检查Triton
    try:
        import triton
        requirements['triton_available'] = True
        requirements['triton_version'] = triton.__version__
    except ImportError:
        pass
    
    # 检查NSys
    try:
        import subprocess
        result = subprocess.run(['nsys', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            requirements['nsys_available'] = True
            requirements['nsys_version'] = result.stdout.strip()
    except FileNotFoundError:
        pass
    
    # 检查NCU
    try:
        result = subprocess.run(['ncu', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            requirements['ncu_available'] = True
            requirements['ncu_version'] = result.stdout.strip()
    except FileNotFoundError:
        pass
    
    # 设备信息
    if requirements['cuda_available']:
        device_props = torch.cuda.get_device_properties(0)
        requirements['device_info'] = {
            'name': device_props.name,
            'compute_capability': f"{device_props.major}.{device_props.minor}",
            'memory_gb': device_props.total_memory / (1024**3)
        }
    
    return requirements

def estimate_memory_usage(batch: int, heads: int, seq_len: int, head_dim: int, dtype=torch.float16) -> Dict[str, float]:
    """估算内存使用量"""
    element_size = 2 if dtype == torch.float16 else 4
    
    # 输入张量 (Q, K, V)
    input_size = 3 * batch * heads * seq_len * head_dim * element_size
    
    # 输出张量
    output_size = batch * heads * seq_len * head_dim * element_size
    
    # 中间计算 (attention scores)
    attention_size = batch * heads * seq_len * seq_len * 4  # float32
    
    # 缓冲区 (Triton内部使用)
    buffer_size = batch * heads * seq_len * head_dim * element_size * 2  # 估算
    
    total_bytes = input_size + output_size + attention_size + buffer_size
    
    return {
        'input_gb': input_size / (1024**3),
        'output_gb': output_size / (1024**3),
        'attention_gb': attention_size / (1024**3),
        'buffer_gb': buffer_size / (1024**3),
        'total_gb': total_bytes / (1024**3)
    }

def compare_configurations(config1: Dict, config2: Dict) -> str:
    """比较两个配置"""
    lines = []
    lines.append("Configuration Comparison:")
    lines.append("=" * 40)
    
    all_keys = set(config1.keys()) | set(config2.keys())
    
    for key in sorted(all_keys):
        val1 = config1.get(key, 'N/A')
        val2 = config2.get(key, 'N/A')
        
        if val1 != val2:
            lines.append(f"{key}: {val1} vs {val2}")
        else:
            lines.append(f"{key}: {val1}")
    
    return "\n".join(lines)

def export_to_csv(results: Dict[str, Any], output_file: str):
    """导出结果到CSV文件"""
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'shape_key', 'batch', 'heads', 'seq_len', 'head_dim',
            'avg_time_ms', 'tflops', 'success',
            'block_m', 'block_n', 'num_buffers', 'num_mma_warps', 'num_mma_groups',
            'num_warps', 'memory_gb'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for shape_key, result in results.items():
            if result['best_config']['success']:
                shape = result['shape']
                config = result['best_config']
                memory_info = estimate_memory_usage(**shape)
                
                row = {
                    'shape_key': shape_key,
                    'batch': shape['batch'],
                    'heads': shape['heads'],
                    'seq_len': shape['seq_len'],
                    'head_dim': shape['head_dim'],
                    'avg_time_ms': config['avg_time_ms'],
                    'tflops': config['tflops'],
                    'success': config['success'],
                    'block_m': config['config']['BLOCK_M'],
                    'block_n': config['config']['BLOCK_N'],
                    'num_buffers': config['config']['NUM_BUFFERS'],
                    'num_mma_warps': config['config']['NUM_MMA_WARPS'],
                    'num_mma_groups': config['config']['NUM_MMA_GROUPS'],
                    'num_warps': config['num_warps'],
                    'memory_gb': memory_info['total_gb']
                }
                
                writer.writerow(row) 