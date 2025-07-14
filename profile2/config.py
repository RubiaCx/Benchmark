"""
Flash Attention Profiling 配置管理
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

@dataclass
class ProfilingConfig:
    """Profiling配置类"""
    
    # 基础路径
    base_dir: str = "profile"
    results_dir: str = "results"
    
    # 测试形状范围
    batch_sizes: List[int] = None
    num_heads: List[int] = None
    seq_lengths: List[int] = None
    head_dims: List[int] = None
    
    # Profiling参数
    warmup_trials: int = 10
    benchmark_trials: int = 100
    nsys_duration: float = 5.0  # NSys profiling持续时间(秒)
    ncu_kernels: int = 10      # NCU分析的内核数量
    
    # 工具选项
    enable_fp8: bool = False
    enable_validation: bool = True
    save_best_configs: bool = True
    
    # 硬件信息
    device: str = "cuda"
    
    def __post_init__(self):
        """初始化后设置默认值"""
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16]
        
        if self.num_heads is None:
            self.num_heads = [8, 16, 32, 64]
        
        if self.seq_lengths is None:
            self.seq_lengths = [256, 512, 1024, 2048, 4096]
        
        if self.head_dims is None:
            self.head_dims = [64, 128, 256]
    
    def get_all_shapes(self):
        """获取所有测试形状组合"""
        shapes = []
        for batch in self.batch_sizes:
            for heads in self.num_heads:
                for seq_len in self.seq_lengths:
                    for head_dim in self.head_dims:
                        shapes.append({
                            'batch': batch,
                            'heads': heads, 
                            'seq_len': seq_len,
                            'head_dim': head_dim
                        })
        return shapes
    
    def get_output_paths(self):
        """获取输出路径"""
        paths = {}
        base = os.path.join(self.base_dir, self.results_dir)
        
        paths['base'] = base
        paths['configs'] = os.path.join(base, 'configs')
        paths['nsys'] = os.path.join(base, 'nsys')
        paths['ncu'] = os.path.join(base, 'ncu')
        paths['logs'] = os.path.join(base, 'logs')
        
        # 创建目录
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        
        return paths

# 预定义的测试配置

# 快速测试配置 - 用于开发和调试
QUICK_CONFIG = ProfilingConfig(
    batch_sizes=[1, 4],
    num_heads=[16, 32],
    seq_lengths=[512, 1024],
    head_dims=[128],
    warmup_trials=5,
    benchmark_trials=20
)

# 全面测试配置 - 生产环境使用
FULL_CONFIG = ProfilingConfig(
    batch_sizes=[1, 2, 4, 8, 16, 32],
    num_heads=[8, 16, 32, 64],
    seq_lengths=[256, 512, 1024, 2048, 4096, 8192],
    head_dims=[64, 128, 256],
    warmup_trials=20,
    benchmark_trials=100
)

# 大尺寸测试配置 - 针对大模型
LARGE_CONFIG = ProfilingConfig(
    batch_sizes=[4, 8, 16],
    num_heads=[32, 64, 128],
    seq_lengths=[2048, 4096, 8192, 16384],
    head_dims=[128, 256],
    warmup_trials=10,
    benchmark_trials=50
)

# Triton配置生成器
class TritonConfigGenerator:
    """生成Triton自动调优配置"""
    
    @staticmethod
    def get_flash_attention_configs():
        """获取Flash Attention的Triton配置"""
        configs = []
        
        # 基于原始文件的配置，扩展更多选项
        config_options = {
            'BLOCK_M': [64, 128, 256],
            'BLOCK_N': [64, 128, 256], 
            'NUM_BUFFERS': [2, 4],
            'NUM_MMA_WARPS': [4, 8],
            'NUM_MMA_GROUPS': [1, 2, 4]
        }
        
        # 生成所有合理的配置组合
        import itertools
        from triton import Config
        
        for block_m in config_options['BLOCK_M']:
            for block_n in config_options['BLOCK_N']:
                for num_buffers in config_options['NUM_BUFFERS']:
                    for num_mma_warps in config_options['NUM_MMA_WARPS']:
                        for num_mma_groups in config_options['NUM_MMA_GROUPS']:
                            # 检查配置合理性
                            if block_m % num_mma_groups != 0:
                                continue
                            
                            # 计算总warp数量
                            total_warps = max(4, num_mma_warps // num_mma_groups * num_mma_groups)
                            if total_warps > 32:  # GPU warp限制
                                continue
                            
                            config = Config(
                                {
                                    'BLOCK_M': block_m,
                                    'BLOCK_N': block_n, 
                                    'NUM_BUFFERS': num_buffers,
                                    'NUM_MMA_WARPS': num_mma_warps,
                                    'NUM_MMA_GROUPS': num_mma_groups
                                },
                                num_stages=0,
                                num_warps=total_warps
                            )
                            configs.append(config)
        
        return configs
    
    @staticmethod
    def filter_configs_by_shape(configs, shape):
        """根据输入形状过滤配置"""
        filtered = []
        seq_len = shape['seq_len']
        head_dim = shape['head_dim']
        
        for config in configs:
            # 确保block大小不超过序列长度和head维度
            if (config.kwargs['BLOCK_M'] <= seq_len and 
                config.kwargs['BLOCK_N'] <= head_dim):
                filtered.append(config)
        
        return filtered

# 硬件信息获取
def get_device_info():
    """获取GPU设备信息"""
    if not torch.cuda.is_available():
        return None
    
    device_props = torch.cuda.get_device_properties(0)
    
    return {
        'name': device_props.name,
        'major': device_props.major,
        'minor': device_props.minor,
        'total_memory': device_props.total_memory,
        'multi_processor_count': device_props.multi_processor_count,
        'max_threads_per_multi_processor': device_props.max_threads_per_multi_processor,
        'max_threads_per_block': device_props.max_threads_per_block,
        'max_shared_memory_per_block': device_props.max_shared_memory_per_block
    }

# 内存估算
def estimate_memory_usage(batch, heads, seq_len, head_dim, dtype=torch.float16):
    """估算Flash Attention的内存使用量"""
    element_size = 2 if dtype == torch.float16 else 4  # bytes per element
    
    # Q, K, V张量
    qkv_size = 3 * batch * heads * seq_len * head_dim * element_size
    
    # 输出张量
    output_size = batch * heads * seq_len * head_dim * element_size
    
    # 中间结果(近似)
    intermediate_size = batch * heads * seq_len * seq_len * 4  # attention scores in fp32
    
    total_bytes = qkv_size + output_size + intermediate_size
    total_gb = total_bytes / (1024**3)
    
    return {
        'qkv_gb': qkv_size / (1024**3),
        'output_gb': output_size / (1024**3), 
        'intermediate_gb': intermediate_size / (1024**3),
        'total_gb': total_gb
    }

# 默认配置
DEFAULT_CONFIG = FULL_CONFIG 