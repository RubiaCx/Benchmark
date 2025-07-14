import pytest
import torch
import os
import time
import json
import argparse
from pathlib import Path

import triton
import triton.language as tl
import triton.tlx.language as tlx
from triton.tools.tensor_descriptor import TensorDescriptor


DEVICE = triton.runtime.driver.active.get_active_torch_device()

# 全局变量用于记录autotune结果
_autotune_results = {}
_config_printed = set()

def log_autotune_result(provider, batch, h, n_ctx, head_dim, config_info):
    """记录autotune结果"""
    key = f"{provider}_{batch}_{h}_{n_ctx}_{head_dim}"
    _autotune_results[key] = config_info
    
def get_best_config_info(provider, batch, h, n_ctx, head_dim):
    """获取最佳配置信息"""
    key = f"{provider}_{batch}_{h}_{n_ctx}_{head_dim}"
    return _autotune_results.get(key, None)


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

configs = [
    # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_BUFFERS': 2, 'NUM_MMA_WARPS': 4, 'NUM_MMA_GROUPS': 1}, num_stages=0, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_BUFFERS': 2, 'NUM_MMA_WARPS': 8, 'NUM_MMA_GROUPS': 2}, num_stages=0, num_warps=4, pre_hook=_host_descriptor_pre_hook),
]


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

class _attention_ws(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # NVTX标记开始
        torch.cuda.nvtx.range_push(f"attention_ws_B{q.shape[0]}_H{q.shape[1]}_N{q.shape[2]}_D{q.shape[3]}")
        
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
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

        ctx.grid = grid
        
        # 记录配置信息
        try:
            kernel = _attn_fwd_ws
            if hasattr(kernel, 'cache') and kernel.cache:
                for key, (compiled_kernel, config, constants) in kernel.cache.items():
                    if config:
                        config_info = {
                            'BLOCK_M': config.kwargs.get('BLOCK_M', 'N/A'),
                            'BLOCK_N': config.kwargs.get('BLOCK_N', 'N/A'),
                            'NUM_BUFFERS': config.kwargs.get('NUM_BUFFERS', 'N/A'),
                            'NUM_MMA_WARPS': config.kwargs.get('NUM_MMA_WARPS', 'N/A'),
                            'NUM_MMA_GROUPS': config.kwargs.get('NUM_MMA_GROUPS', 'N/A'),
                            'num_warps': config.num_warps,
                            'num_stages': config.num_stages,
                        }
                        log_autotune_result('triton-ws', q.shape[0], q.shape[1], q.shape[2], q.shape[3], config_info)
                        break
        except Exception as e:
            pass
        
        _attn_fwd_ws[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        
        torch.cuda.nvtx.range_pop()
        return o

attention_ws = _attention_ws.apply


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
@triton.jit
def _attn_fwd_ws_pipelined(sm_scale, M,  #
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
            k_phase = 0
            v_phase = 1
            k_buf_id = 0
            v_buf_id = 0

            # wait for the K[0] buffer to be populated by the producer
            k_full = tlx.local_view(k_fulls, k_buf_id)
            tlx.barrier_wait(k_full, k_phase)
            k_tile = tlx.local_view(k_tiles, k_buf_id)

            # -- compute qk[0] ----
            k_tile = tlx.local_trans(k_tile)
            qk = tlx.async_dot(q_tile, k_tile)
            # wait for the MMA using to complete
            qk = tlx.async_dot_wait(0, qk)
            # release the K buffer
            k_empty = tlx.local_view(k_empties, k_buf_id)
            tlx.barrier_arrive(k_empty, 1)

            # -- compute m_i and l_i ----
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            # -- update output accumulator[0] --
            acc = acc * alpha[:, None]
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            acc_cnt = 1


            # loop over k, v and update accumulator
            for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                k_buf_id = acc_cnt % NUM_BUFFERS
                # buffers in a row share the same phase
                k_phase = k_phase ^ (k_buf_id == 0)

                # wait for the K buffer to be populated by the producer
                k_full = tlx.local_view(k_fulls, k_buf_id)
                tlx.barrier_wait(k_full, k_phase)
                k_tile = tlx.local_view(k_tiles, k_buf_id)

                # compute qk for the current iteration
                k_tile = tlx.local_trans(k_tile)
                qk = tlx.async_dot(q_tile, k_tile)

                # compute pv from the previous iteration
                # wait for the previous V buffer to be populated by the producer
                v_buf_id = (acc_cnt - 1) % NUM_BUFFERS
                v_phase = v_phase ^ (v_buf_id == 0)
                v_full = tlx.local_view(v_fulls, v_buf_id)
                tlx.barrier_wait(v_full, v_phase)
                v_tile = tlx.local_view(v_tiles, v_buf_id)
                # prepare p and v for the dot
                p = p.to(tlx.dtype_of(desc_k))
                acc = tlx.async_dot(p, v_tile, acc)

                # wait for the current qk MMA to complete
                qk = tlx.async_dot_wait(1, qk)
                # release the K buffer
                k_empty = tlx.local_view(k_empties, k_buf_id)
                tlx.barrier_arrive(k_empty, 1)

                # -- compute m_i and l_i ----
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                # -- compute correction factor
                alpha = tl.math.exp2(m_i - m_ij)
                l_ij = tl.sum(p, 1)
                # update m_i and l_i
                l_i = l_i * alpha + l_ij
                m_i = m_ij

                # -- update output accumulator --
                # wait for the previous pv MMA to complete
                acc = tlx.async_dot_wait(0, acc)
                # release the V buffer
                v_empty = tlx.local_view(v_empties, v_buf_id)
                tlx.barrier_arrive(v_empty, 1)
                acc = acc * alpha[:, None]
                acc_cnt += 1

            # compute pv from the last iteration
            # wait for the V buffer to be populated by the producer
            v_buf_id = (acc_cnt - 1) % NUM_BUFFERS
            v_phase = v_phase ^ (v_buf_id == 0)
            v_full = tlx.local_view(v_fulls, v_buf_id)
            tlx.barrier_wait(v_full, v_phase)
            v_tile = tlx.local_view(v_tiles, v_buf_id)
            # prepare p and v for the dot
            p = p.to(tlx.dtype_of(desc_k))
            acc = tlx.async_dot(p, v_tile, acc)
            # wait for the MMA using to complete
            acc = tlx.async_dot_wait(0, acc)
            # release the V buffer
            v_empty = tlx.local_view(v_empties, v_buf_id)
            tlx.barrier_arrive(v_empty, 1)

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

class _attention_ws_pp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # NVTX标记开始
        torch.cuda.nvtx.range_push(f"attention_ws_pp_B{q.shape[0]}_H{q.shape[1]}_N{q.shape[2]}_D{q.shape[3]}")
        
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
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

        ctx.grid = grid
        
        # 记录配置信息
        try:
            kernel = _attn_fwd_ws_pipelined
            if hasattr(kernel, 'cache') and kernel.cache:
                for key, (compiled_kernel, config, constants) in kernel.cache.items():
                    if config:
                        config_info = {
                            'BLOCK_M': config.kwargs.get('BLOCK_M', 'N/A'),
                            'BLOCK_N': config.kwargs.get('BLOCK_N', 'N/A'),
                            'NUM_BUFFERS': config.kwargs.get('NUM_BUFFERS', 'N/A'),
                            'NUM_MMA_WARPS': config.kwargs.get('NUM_MMA_WARPS', 'N/A'),
                            'NUM_MMA_GROUPS': config.kwargs.get('NUM_MMA_GROUPS', 'N/A'),
                            'num_warps': config.num_warps,
                            'num_stages': config.num_stages,
                        }
                        log_autotune_result('triton-ws-pp', q.shape[0], q.shape[1], q.shape[2], q.shape[3], config_info)
                        break
        except Exception as e:
            pass
        
        _attn_fwd_ws_pipelined[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        
        torch.cuda.nvtx.range_pop()
        return o


attention_ws_pp = _attention_ws_pp.apply


def print_best_config(provider, batch, h, n_ctx, head_dim, timing_ms):
    """打印最佳配置信息"""
    config_info = get_best_config_info(provider, batch, h, n_ctx, head_dim)
    
    print(f"\n{'='*80}")
    print(f"AutoTune 最佳配置信息 - {provider}")
    print(f"形状: BATCH={batch}, H={h}, N_CTX={n_ctx}, HEAD_DIM={head_dim}")
    print(f"执行时间: {timing_ms:.3f} ms")
    print(f"{'='*80}")
    
    if config_info:
        print(f"最佳配置参数:")
        print(f"  BLOCK_M: {config_info['BLOCK_M']}")
        print(f"  BLOCK_N: {config_info['BLOCK_N']}")
        print(f"  NUM_BUFFERS: {config_info['NUM_BUFFERS']}")
        print(f"  NUM_MMA_WARPS: {config_info['NUM_MMA_WARPS']}")
        print(f"  NUM_MMA_GROUPS: {config_info['NUM_MMA_GROUPS']}")
        print(f"  num_warps: {config_info['num_warps']}")
        print(f"  num_stages: {config_info['num_stages']}")
        
        # 计算理论性能指标
        flops = 4 * batch * h * n_ctx * n_ctx * head_dim
        tflops = flops / (timing_ms / 1000) / 1e12
        print(f"  理论FLOPS: {flops/1e12:.2f} TFLOPS")
        print(f"  实际性能: {tflops:.2f} TFLOPS")
        
        # 内存带宽分析
        elem_size = 2  # float16
        q_size = batch * h * n_ctx * head_dim * elem_size
        k_size = batch * h * n_ctx * head_dim * elem_size
        v_size = batch * h * n_ctx * head_dim * elem_size
        o_size = batch * h * n_ctx * head_dim * elem_size
        total_memory = q_size + k_size + v_size + o_size
        bandwidth_gb_s = total_memory / (timing_ms / 1000) / 1e9
        print(f"  内存带宽: {bandwidth_gb_s:.2f} GB/s")
    else:
        print("无法获取配置信息")
    
    print(f"{'='*80}\n")


@pytest.mark.parametrize("Z", [8])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("N_CTX", [1024])
@pytest.mark.parametrize("HEAD_DIM", [128])
@pytest.mark.parametrize("mode", ["fwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"])
def test_op(Z, H, N_CTX, HEAD_DIM, mode, provider, dtype=torch.float16):
    if mode == "bwd":
        pytest.skip("Backward pass not supported.")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention_ws(q, k, v, sm_scale).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM = 1, 32, 128
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        # x_vals=[2**i for i in range(8, 15)],
        x_vals=[128, 8192],
        line_arg="provider",
        line_vals=["triton-ws-fp16"] + ["triton-ws-pp-fp16"] + (["flash"] if HAS_FLASH else []),
        line_names=["Triton WS [FP16]"] + ["Triton WS PP [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name=
        f"fused-attention-ws-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": "fwd",
        },
    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    
    # 添加NVTX标记
    torch.cuda.nvtx.range_push(f"bench_{provider}_B{BATCH}_H{H}_N{N_CTX}_D{HEAD_DIM}")
    
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        if provider == "triton-ws-fp16":
            fn = lambda: attention_ws(q, k, v, sm_scale)
        elif provider == "triton-ws-pp-fp16":
            fn = lambda: attention_ws_pp(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
        
        # 打印最佳配置信息
        config_key = (BATCH, H, N_CTX, HEAD_DIM, mode, provider)
        if config_key not in _config_printed:
            print_best_config(provider, BATCH, H, N_CTX, HEAD_DIM, ms)
            _config_printed.add(config_key)

    elif provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
        
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    
    torch.cuda.nvtx.range_pop()
    return total_flops * 1e-12 / (ms * 1e-3)
