"""
Flash Attention 适配器
用于与原始 flash-attention-WS-hopper.py 文件接口
"""

import os
import sys
import torch
import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import Dict, Any, Tuple, Optional

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FlashAttentionAdapter:
    """Flash Attention适配器类"""
    
    def __init__(self, original_file_path: str = None):
        """
        初始化适配器
        
        Args:
            original_file_path: 原始flash attention文件路径
        """
        self.original_file_path = original_file_path or self._find_original_file()
        self.device = triton.runtime.driver.active.get_active_torch_device()
        
        # 导入原始实现
        self._import_original_implementation()
    
    def _find_original_file(self) -> str:
        """查找原始flash attention文件"""
        possible_paths = [
            "triton_meta/third_party/tlx/tutorials/flash-attention-WS-hopper.py",
            "../triton_meta/third_party/tlx/tutorials/flash-attention-WS-hopper.py",
            "../../triton_meta/third_party/tlx/tutorials/flash-attention-WS-hopper.py"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("无法找到原始flash attention文件")
    
    def _import_original_implementation(self):
        """导入原始实现"""
        # 动态导入原始模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("flash_attention_original", self.original_file_path)
        self.original_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.original_module)
        
        # 获取关键组件
        self.original_attention = self.original_module.attention
        self.original_configs = self.original_module.configs
        self.original_pre_hook = self.original_module._host_descriptor_pre_hook
    
    def create_custom_configs(self, custom_params: Dict[str, Any]) -> list:
        """
        创建自定义配置
        
        Args:
            custom_params: 自定义参数字典
            
        Returns:
            配置列表
        """
        configs = []
        
        # 基础配置参数
        base_config = {
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'NUM_BUFFERS': 2,
            'NUM_MMA_WARPS': 4,
            'NUM_MMA_GROUPS': 1
        }
        
        # 更新自定义参数
        base_config.update(custom_params)
        
        # 创建Triton配置
        config = triton.Config(
            base_config,
            num_stages=0,
            num_warps=4,
            pre_hook=self.original_pre_hook
        )
        
        configs.append(config)
        return configs
    
    def run_attention_with_config(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                 sm_scale: float, config: Dict[str, Any]) -> torch.Tensor:
        """
        使用指定配置运行attention
        
        Args:
            q: Query tensor
            k: Key tensor  
            v: Value tensor
            sm_scale: Softmax scale
            config: 配置参数
            
        Returns:
            输出tensor
        """
        # 创建自定义配置
        custom_configs = self.create_custom_configs(config)
        
        # 创建带有自定义配置的attention函数
        attention_func = self._create_attention_function(custom_configs)
        
        # 运行attention
        return attention_func(q, k, v, sm_scale)
    
    def _create_attention_function(self, configs: list):
        """创建attention函数"""
        # 复制原始内核函数并应用新配置
        @triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
        @triton.jit
        def _attn_fwd_ws_custom(sm_scale, M,  #
                              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                              HEAD_DIM: tl.constexpr,  #
                              BLOCK_M: tl.constexpr,  #
                              BLOCK_N: tl.constexpr,  #
                              FP8_OUTPUT: tl.constexpr,  #
                              NUM_BUFFERS: tl.constexpr,  #
                              NUM_MMA_WARPS: tl.constexpr,  #
                              NUM_MMA_GROUPS: tl.constexpr,  #
                              ):
            # 使用原始实现的内核代码
            # 这里复制原始的内核实现
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
        
        # 创建attention函数
        class _attention_custom(torch.autograd.Function):
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
                if q.dtype == torch.float8_e5m2:
                    desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
                else:
                    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                
                def alloc_fn(size: int, align: int, _):
                    return torch.empty(size, dtype=torch.int8, device="cuda")
                
                triton.set_allocator(alloc_fn)
                
                def grid(META):
                    return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
                
                _attn_fwd_ws_custom[grid](
                    sm_scale, M,
                    q.shape[0], q.shape[1],
                    desc_q, desc_k, desc_v, desc_o,
                    N_CTX=q.shape[2],
                    HEAD_DIM=HEAD_DIM_K,
                    FP8_OUTPUT=q.dtype == torch.float8_e5m2,
                )
                
                return o
        
        return _attention_custom.apply
    
    def benchmark_config(self, config: Dict[str, Any], shape: Tuple[int, int, int, int], 
                        trials: int = 100, warmup: int = 10) -> Dict[str, Any]:
        """
        基准测试单个配置
        
        Args:
            config: 配置参数
            shape: 输入形状 (batch, heads, seq_len, head_dim)
            trials: 测试次数
            warmup: 预热次数
            
        Returns:
            基准测试结果
        """
        batch, heads, seq_len, head_dim = shape
        
        # 生成测试数据
        torch.manual_seed(42)
        q = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=self.device)
        k = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=self.device)
        v = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device=self.device)
        sm_scale = 0.5
        
        try:
            # 预热
            for _ in range(warmup):
                output = self.run_attention_with_config(q, k, v, sm_scale, config)
                torch.cuda.synchronize()
            
            # 基准测试
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            for _ in range(trials):
                output = self.run_attention_with_config(q, k, v, sm_scale, config)
            end_time.record()
            
            torch.cuda.synchronize()
            avg_time = start_time.elapsed_time(end_time) / trials
            
            # 计算FLOPS
            flops = 4 * batch * heads * seq_len * seq_len * head_dim
            tflops = flops / (avg_time / 1000) / 1e12
            
            return {
                'config': config,
                'avg_time_ms': avg_time,
                'tflops': tflops,
                'success': True
            }
            
        except Exception as e:
            return {
                'config': config,
                'avg_time_ms': float('inf'),
                'tflops': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def validate_output(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                       sm_scale: float, config: Dict[str, Any]) -> bool:
        """
        验证输出正确性
        
        Args:
            q, k, v: 输入张量
            sm_scale: Softmax scale
            config: 配置参数
            
        Returns:
            是否通过验证
        """
        try:
            # 使用自定义配置
            custom_output = self.run_attention_with_config(q, k, v, sm_scale, config)
            
            # 使用原始实现
            reference_output = self.original_attention(q, k, v, sm_scale)
            
            # 比较结果
            atol = 1e-2
            rtol = 1e-2
            
            return torch.allclose(custom_output, reference_output, atol=atol, rtol=rtol)
            
        except Exception as e:
            print(f"验证失败: {e}")
            return False
    
    def get_original_configs(self) -> list:
        """获取原始配置"""
        return self.original_configs
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            return {
                'name': device_props.name,
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'memory_gb': device_props.total_memory / (1024**3),
                'multi_processor_count': device_props.multi_processor_count
            }
        return {} 