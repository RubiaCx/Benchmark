"""
Flash Attention Profiling 核心逻辑
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
import torch
import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton.tools.tensor_descriptor import TensorDescriptor

# 添加父目录到path以导入flash attention模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ProfilingConfig, TritonConfigGenerator, get_device_info, estimate_memory_usage
from utils import setup_logging, save_results, load_results

# 导入并修改Flash Attention实现
def create_flash_attention_kernel():
    """创建可配置的Flash Attention内核"""
    
    def _host_descriptor_pre_hook(nargs):
        BLOCK_M = nargs["BLOCK_M"]
        BLOCK_N = nargs["BLOCK_N"]
        HEAD_DIM = nargs["HEAD_DIM"]
        if not isinstance(nargs["desc_q"], TensorDescriptor):
            return
        HEAD_DIM = nargs["HEAD_DIM"]
        NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
        BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
        nargs["desc_q"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]
        if nargs["FP8_OUTPUT"]:
            nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
        else:
            nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
        nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
        nargs["desc_o"].block_shape = [BLOCK_M_SPLIT, HEAD_DIM]

    def create_autotune_kernel(configs):
        @triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
        @triton.jit
        def _attn_fwd_ws(sm_scale, M,  #
                      Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                      HEAD_DIM: tl.constexpr,  #
                      BLOCK_M: tl.constexpr,  #
                      BLOCK_N: tl.constexpr,  #
                      FP8_OUTPUT: tl.constexpr,  #
                      NUM_BUFFERS: tl.constexpr,  #
                      NUM_MMA_WARPS: tl.constexpr,  #
                      NUM_MMA_GROUPS: tl.constexpr,  #
                      ):
            tl.static_assert(BLOCK_N <= HEAD_DIM)
            BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

            # allocate buffers
            q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
            k_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS)
            v_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS)

            # allocate barriers
            q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
            k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
            k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
            v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS)
            v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)

            with tlx.async_tasks():
                # producer group
                with tlx.async_task("default"):
                    # initialize offsets
                    start_m = tl.program_id(0)
                    off_hz = tl.program_id(1)
                    off_z = off_hz // H
                    off_h = off_hz % H
                    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
                    qo_offset_y = offset_y + start_m * BLOCK_M
                    lo, hi = 0, N_CTX
                    kv_offset_y = offset_y + lo

                    # load q: it will stay in SRAM throughout
                    for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                        q_full = tlx.local_view(q_fulls, cid)
                        tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                        q_tile = tlx.local_view(q_tiles, cid)
                        qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                        tlx.async_descriptor_load(desc_q, q_tile, [qo_offset_y_split, 0], q_full)

                    # loop over loading k, v
                    kv_phase = 0
                    acc_cnt = 0
                    for _ in tl.range(lo, hi, BLOCK_N):
                        buf_id = acc_cnt % NUM_BUFFERS
                        # buffers in a row share the same phase
                        kv_phase = kv_phase ^ (buf_id == 0)

                        # wait for the K buffer to be released by the consumer
                        k_empty = tlx.local_view(k_empties, buf_id)
                        tlx.barrier_wait(k_empty, kv_phase)
                        # load K
                        k_full = tlx.local_view(k_fulls, buf_id)
                        k_tile = tlx.local_view(k_tiles, buf_id)
                        tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                        tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                        # wait for the V buffer to be released by the consumer
                        v_empty = tlx.local_view(v_empties, buf_id)
                        tlx.barrier_wait(v_empty, kv_phase)
                        # load V
                        v_full = tlx.local_view(v_fulls, buf_id)
                        v_tile = tlx.local_view(v_tiles, buf_id)
                        tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                        tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                        kv_offset_y += BLOCK_N
                        acc_cnt += 1

                # consumer group
                with tlx.async_task(num_warps=NUM_MMA_WARPS // NUM_MMA_GROUPS, registers=232, replicate=NUM_MMA_GROUPS):
                    # initialize pointer to m and l
                    m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                    l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                    acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)

                    # load scales
                    qk_scale = sm_scale
                    qk_scale *= 1.44269504  # 1/log(2)

                    # wait for the Q buffer to be populated by the producer
                    cid = tlx.async_task_replica_id()
                    q_full = tlx.local_view(q_fulls, cid)
                    tlx.barrier_wait(q_full, 0)
                    q_tile = tlx.local_view(q_tiles, cid)

                    lo, hi = 0, N_CTX
                    kv_phase = 1
                    acc_cnt = 0

                    # loop over k, v and update accumulator
                    for _ in tl.range(lo, hi, BLOCK_N):
                        buf_id = acc_cnt % NUM_BUFFERS
                        # buffers in a row share the same phase
                        kv_phase = kv_phase ^ (buf_id == 0)

                        # wait for the K buffer to be populated by the producer
                        k_full = tlx.local_view(k_fulls, buf_id)
                        tlx.barrier_wait(k_full, kv_phase)
                        k_tile = tlx.local_view(k_tiles, buf_id)

                        # -- compute qk ----
                        k_tile = tlx.local_trans(k_tile)
                        qk = tlx.async_dot(q_tile, k_tile)
                        # wait for the MMA using to complete
                        qk = tlx.async_dot_wait(0, qk)
                        # release the K buffer
                        k_empty = tlx.local_view(k_empties, buf_id)
                        tlx.barrier_arrive(k_empty, 1)

                        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                        qk = qk * qk_scale - m_ij[:, None]
                        p = tl.math.exp2(qk)
                        # -- compute correction factor
                        alpha = tl.math.exp2(m_i - m_ij)
                        l_ij = tl.sum(p, 1)
                        # -- update output accumulator --
                        acc = acc * alpha[:, None]
                        # prepare p and v for the dot
                        p = p.to(tlx.dtype_of(desc_k))

                        # wait for the V buffer to be populated by the producer
                        v_full = tlx.local_view(v_fulls, buf_id)
                        tlx.barrier_wait(v_full, kv_phase)
                        v_tile = tlx.local_view(v_tiles, buf_id)
                        acc = tlx.async_dot(p, v_tile, acc)
                        # wait for the MMA using to complete
                        acc = tlx.async_dot_wait(0, acc)
                        # release the V buffer
                        v_empty = tlx.local_view(v_empties, buf_id)
                        tlx.barrier_arrive(v_empty, 1)

                        # update m_i and l_i
                        # place this at the end of the loop to reduce register pressure
                        l_i = l_i * alpha + l_ij
                        m_i = m_ij
                        acc_cnt += 1

                    # epilogue
                    start_m = tl.program_id(0)
                    off_hz = tl.program_id(1)
                    off_z = off_hz // H
                    off_h = off_hz % H
                    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
                    qo_offset_y = offset_y + start_m * BLOCK_M
                    qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                    m_i += tl.math.log2(l_i)
                    acc = acc / l_i[:, None]
                    offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                    m_ptrs = M + off_hz * N_CTX + offs_m
                    tl.store(m_ptrs, m_i)
                    desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))
        
        return _attn_fwd_ws
    
    return create_autotune_kernel, _host_descriptor_pre_hook

class FlashAttentionProfiler:
    """Flash Attention Profiler类"""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.device_info = get_device_info()
        self.logger = setup_logging(config.get_output_paths()['logs'])
        self.results = {}
        
        # 创建内核生成器
        self.kernel_creator, self.pre_hook = create_flash_attention_kernel()
        
    def generate_test_data(self, shape, dtype=torch.float16):
        """生成测试数据"""
        batch, heads, seq_len, head_dim = shape['batch'], shape['heads'], shape['seq_len'], shape['head_dim']
        device = self.config.device
        
        torch.manual_seed(42)  # 确保结果可重现
        
        q = torch.randn((batch, heads, seq_len, head_dim), dtype=dtype, device=device)
        k = torch.randn((batch, heads, seq_len, head_dim), dtype=dtype, device=device)
        v = torch.randn((batch, heads, seq_len, head_dim), dtype=dtype, device=device)
        
        return q, k, v
    
    def create_attention_function(self, configs):
        """创建attention函数"""
        kernel = self.kernel_creator(configs)
        
        class _attention(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v, sm_scale):
                HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
                HEAD_DIM_V = v.shape[-1]
                assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
                o = torch.empty_like(q)
                
                M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
                y_dim = q.shape[0] * q.shape[1] * q.shape[2]
                
                dummy_block = [1, 1]
                desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                
                def alloc_fn(size: int, align: int, _):
                    return torch.empty(size, dtype=torch.int8, device="cuda")
                
                triton.set_allocator(alloc_fn)
                
                def grid(META):
                    return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
                
                kernel[grid](
                    sm_scale, M,
                    q.shape[0], q.shape[1],
                    desc_q, desc_k, desc_v, desc_o,
                    N_CTX=q.shape[2],
                    HEAD_DIM=HEAD_DIM_K,
                    FP8_OUTPUT=q.dtype == torch.float8_e5m2,
                )
                
                return o
        
        return _attention.apply
    
    def benchmark_single_config(self, shape, config, warmup=10, trials=100):
        """对单个配置进行基准测试"""
        try:
            # 生成测试数据
            q, k, v = self.generate_test_data(shape)
            sm_scale = 0.5
            
            # 创建attention函数
            attention = self.create_attention_function([config])
            
            # Warmup
            for _ in range(warmup):
                output = attention(q, k, v, sm_scale)
                torch.cuda.synchronize()
            
            # 基准测试
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(trials):
                output = attention(q, k, v, sm_scale)
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / trials * 1000  # ms
            
            # 计算FLOPS
            batch, heads, seq_len, head_dim = shape['batch'], shape['heads'], shape['seq_len'], shape['head_dim']
            flops_per_iter = 4 * batch * heads * seq_len * seq_len * head_dim  # 近似FLOPS
            tflops = flops_per_iter / (avg_time / 1000) / 1e12
            
            return {
                'config': config.kwargs,
                'num_warps': config.num_warps,
                'avg_time_ms': avg_time,
                'tflops': tflops,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Config {config.kwargs} failed: {str(e)}")
            return {
                'config': config.kwargs,
                'num_warps': config.num_warps,
                'avg_time_ms': float('inf'),
                'tflops': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def find_best_config(self, shape):
        """为给定形状找到最优配置"""
        self.logger.info(f"Finding best config for shape: {shape}")
        
        # 生成配置
        all_configs = TritonConfigGenerator.get_flash_attention_configs()
        filtered_configs = TritonConfigGenerator.filter_configs_by_shape(all_configs, shape)
        
        if not filtered_configs:
            self.logger.warning(f"No valid configs for shape {shape}")
            return None
        
        self.logger.info(f"Testing {len(filtered_configs)} configurations")
        
        results = []
        for i, config in enumerate(filtered_configs):
            self.logger.info(f"Testing config {i+1}/{len(filtered_configs)}: {config.kwargs}")
            
            result = self.benchmark_single_config(
                shape, config, 
                warmup=self.config.warmup_trials,
                trials=self.config.benchmark_trials
            )
            
            results.append(result)
            
            if result['success']:
                self.logger.info(f"Config result: {result['avg_time_ms']:.2f}ms, {result['tflops']:.2f} TFLOPS")
        
        # 找到最优配置
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            self.logger.error(f"No successful configs for shape {shape}")
            return None
        
        best_result = min(successful_results, key=lambda x: x['avg_time_ms'])
        self.logger.info(f"Best config: {best_result['config']}, {best_result['avg_time_ms']:.2f}ms")
        
        return {
            'shape': shape,
            'best_config': best_result,
            'all_results': results,
            'device_info': self.device_info
        }
    
    def profile_all_shapes(self):
        """对所有形状进行profiling"""
        shapes = self.config.get_all_shapes()
        self.logger.info(f"Profiling {len(shapes)} shapes")
        
        all_results = {}
        
        for i, shape in enumerate(shapes):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing shape {i+1}/{len(shapes)}: {shape}")
            
            # 检查内存需求
            memory_info = estimate_memory_usage(**shape)
            self.logger.info(f"Estimated memory usage: {memory_info['total_gb']:.2f} GB")
            
            if memory_info['total_gb'] > self.device_info['total_memory'] / (1024**3) * 0.8:
                self.logger.warning(f"Skipping shape due to memory constraints")
                continue
            
            try:
                result = self.find_best_config(shape)
                if result:
                    shape_key = f"b{shape['batch']}_h{shape['heads']}_s{shape['seq_len']}_d{shape['head_dim']}"
                    all_results[shape_key] = result
                    
                    if self.config.save_best_configs:
                        self.save_config(shape_key, result)
                        
            except Exception as e:
                self.logger.error(f"Failed to profile shape {shape}: {str(e)}")
        
        self.results = all_results
        return all_results
    
    def save_config(self, shape_key, result):
        """保存最优配置"""
        paths = self.config.get_output_paths()
        config_file = os.path.join(paths['configs'], f"{shape_key}_best_config.json")
        
        save_data = {
            'shape': result['shape'],
            'best_config': result['best_config'],
            'device_info': result['device_info'],
            'timestamp': time.time()
        }
        
        save_results(config_file, save_data)
        self.logger.info(f"Saved best config to {config_file}")
    
    def run_nsys_profiling(self, shape, config_file=None, duration=5.0):
        """运行NSys profiling"""
        self.logger.info(f"Running NSys profiling for shape: {shape}")
        
        # 准备输出文件
        paths = self.config.get_output_paths()
        shape_key = f"b{shape['batch']}_h{shape['heads']}_s{shape['seq_len']}_d{shape['head_dim']}"
        output_file = os.path.join(paths['nsys'], f"{shape_key}_nsys.qdrep")
        
        # 构建命令
        cmd = [
            'nsys', 'profile',
            '--output', output_file,
            '--duration', str(duration),
            '--cuda-memory-usage', 'true',
            '--gpu-metrics-device', '0',
            'python', '-c',
            f"""
import sys
sys.path.append('{os.path.dirname(os.path.abspath(__file__))}')
from profiler import FlashAttentionProfiler
from config import ProfilingConfig

config = ProfilingConfig()
profiler = FlashAttentionProfiler(config)

# 运行测试
shape = {shape}
q, k, v = profiler.generate_test_data(shape)
# 这里需要实现具体的测试逻辑
"""
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"NSys profiling completed: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"NSys profiling failed: {e}")
            return None
    
    def run_ncu_profiling(self, shape, config_file=None, num_kernels=10):
        """运行NCU profiling"""
        self.logger.info(f"Running NCU profiling for shape: {shape}")
        
        # 准备输出文件
        paths = self.config.get_output_paths()
        shape_key = f"b{shape['batch']}_h{shape['heads']}_s{shape['seq_len']}_d{shape['head_dim']}"
        output_file = os.path.join(paths['ncu'], f"{shape_key}_ncu.ncu-rep")
        
        # 构建命令
        cmd = [
            'ncu',
            '--set', 'full',
            '--kernel-regex', '.*triton.*',
            '--launch-count', str(num_kernels),
            '--target-processes', 'all',
            '--output', output_file,
            'python', '-c',
            f"""
import sys
sys.path.append('{os.path.dirname(os.path.abspath(__file__))}')
from profiler import FlashAttentionProfiler
from config import ProfilingConfig

config = ProfilingConfig()
profiler = FlashAttentionProfiler(config)

# 运行测试
shape = {shape}
q, k, v = profiler.generate_test_data(shape)
# 这里需要实现具体的测试逻辑
"""
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"NCU profiling completed: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"NCU profiling failed: {e}")
            return None 