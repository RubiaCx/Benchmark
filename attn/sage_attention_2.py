import math

import torch
import triton
import triton.language as tl
from typing import Optional, Any

RCP_LN2: tl.constexpr = 1.4426950408889634 # exp(x) = exp2(x * log2(e)) = exp2(x / ln(2)) = exp2(x * RCP_LN2)
LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

@triton.jit
def quant_per_block(Input, Output, Scale, 
                    stride_ib, stride_ih, stride_is,
                    stride_ob, stride_oh, stride_os,
                    stride_sb, stride_sh,
                    sm_scale: tl.constexpr,
                    HEAD_DIM: tl.constexpr,
                    SEQ_LEN: tl.constexpr,
                    HEAD_NUM: tl.constexpr,
                    BLOCK: tl.constexpr,
                    QUANT_TYPE: tl.constexpr): 
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // HEAD_NUM
    off_h = off_hb % HEAD_NUM

    offs_n = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_k = tl.arange(0, HEAD_DIM)

    input_ptrs = Input + off_b * stride_ib + off_h * stride_ih + offs_n[:, None] * stride_is + offs_k[None, :]
    output_ptrs = Output + off_b * stride_ob + off_h * stride_oh + offs_n[:, None] * stride_os + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sb + off_h * stride_sh + start_m
    x = tl.load(input_ptrs, mask=offs_n[:, None] < SEQ_LEN)
    x = x.to(tl.float32)
    x *= sm_scale

    if QUANT_TYPE == 0 or QUANT_TYPE == 3:  # int8
        scale = tl.max(tl.abs(x)) / 127. + 1e-8
        x_quant = x / (tl.max(tl.abs(x)) / 127.0 + 1e-8)
        x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)
    elif QUANT_TYPE == 1:  # e4m3
        scale = tl.max(tl.abs(x)) / 448. + 1e-8
        x_quant = (x / scale).to(tl.float8e4nv)
    elif QUANT_TYPE == 2:  # e5m2
        scale = tl.max(tl.abs(x)) / 57344. + 1e-8
        x_quant = (x / scale).to(tl.float8e5)
    else:
        tl.static_assert(False, "Unsupported quant type")
    
    tl.store(output_ptrs, x_quant, mask=offs_n[:, None] < SEQ_LEN)
    tl.store(scale_ptrs, scale)

@triton.jit
def _fwd_kernel(
    Q, K, V, 
    Q_scale, K_scale, V_scale,
    Out, Lse, TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    stride_qb, stride_qh, stride_qs,
    nheads,
    seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, 
    EVEN_N: tl.constexpr, 
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qs + offs_d[None, :]))
    k_ptrs = (K + off_b * stride_qb + off_h * stride_qh + (offs_n[:, None] * stride_qs + offs_d[None, :]))
    v_ptrs = (V + off_b * stride_qb + off_h * stride_qh + (offs_n[:, None] * stride_qs + offs_d[None, :]))

    q_scale_ptr = Q_scale + (off_b * nheads + off_h) * tl.cdiv(seqlen_q, BLOCK_M) + start_m
    k_scale_ptr = K_scale + (off_b * nheads + off_h) * tl.cdiv(seqlen_k, BLOCK_N)
    v_scale_ptr = V_scale + (off_b * nheads + off_h) * tl.cdiv(seqlen_k, BLOCK_N)
    
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, 
                        mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, 
                        mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, 
                        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    q_scale = tl.load(q_scale_ptr)

    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M: 
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_qs)
            else:
                k = tl.load(k_ptrs + start_n * stride_qs, 
                            mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_qs,
                            mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_qs,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        k_scale = tl.load(k_scale_ptr + start_n // BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k)) * q_scale * k_scale

        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp2(m_i - m_ij)

        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        if EVEN_N & EVEN_M: 
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_qs)
            else:
                v = tl.load(v_ptrs + start_n * stride_qs, 
                            mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_qs,
                            mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_qs,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        v_scale = tl.load(v_scale_ptr + start_n // BLOCK_N)
        # p = p.to(v.dtype) #! cast p to v.dtype harm the accuracy
        if QUANT_TYPE == 0:  # int8
            p_scale = tl.max(tl.abs(p)) / 127. + 1e-8
            p_quant = (p / p_scale + 0.5 * tl.where(p >= 0, 1, -1)).to(tl.int8)
        elif QUANT_TYPE == 1:  # e4m3
            p_scale = tl.max(tl.abs(p)) / 448. + 1e-8
            p_quant = (p / p_scale).to(tl.float8e4nv)
        elif QUANT_TYPE == 2:  # e5m2
            p_scale = tl.max(tl.abs(p)) / 57344. + 1e-8
            p_quant = (p / p_scale).to(tl.float8e5)
        else:
            tl.static_assert(False, "Unsupported quant type")

        acc_o += tl.dot(p_quant, v) * v_scale * p_scale

        m_i = m_ij
        l_i_new = tl.exp2(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log2(l_i_new)

    o_scale = tl.exp2(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qs + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, 
                     mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, 
                     mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o, 
                     mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out, DO, Delta,
    stride_ob, stride_oh, stride_os,
    nheads,
    seqlen_q, seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :],
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    do = tl.load(DO + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :],
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)
    
@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q, K, V, DO, 
    Q_scale, K_scale, V_scale, DO_scale,
    DQ, DK, DV,
    LSE, D,
    softmax_scale,
    stride_qs, 
    seqlen_q, seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    start_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M

    offs_qm = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + (offs_qm[:, None] * stride_qs + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_qs + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_qs + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_qs + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_qs + offs_d[None, :])

    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    if start_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_qs + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_qs + offs_d[None, :])
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv)
                tl.store(dk_ptrs, dk)
            else:
                tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
                tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
                tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            else:
                tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
                tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
        return

    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)

    k_scale = tl.load(K_scale + start_n)
    v_scale = tl.load(V_scale + start_n)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for step_m in range(start_m, num_block_m * BLOCK_M, BLOCK_M):
        step_m = tl.multiple_of(step_m, BLOCK_M)
        offs_m_curr = step_m + offs_m

        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, 
                            mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(q_ptrs, 
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        # recompute p = softmax(qk, dim=-1).T
        q_scale  = tl.load(Q_scale  + step_m // BLOCK_M)
        do_scale = tl.load(DO_scale + step_m // BLOCK_M)
        qk = tl.dot(q, tl.trans(k)) * q_scale * k_scale
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
            
        lse_i = tl.load(LSE + offs_m_curr)
        p = tl.exp2(qk - lse_i[:, None])

        if QUANT_TYPE == 0:  # int8
            p_scale = tl.max(tl.abs(p)) / 127.
            p_quant = ((p / p_scale) + 0.5 * tl.where(p >= 0, 1, -1)).to(tl.int8)
        elif QUANT_TYPE == 1:  # e4m3
            p_scale = tl.max(tl.abs(p)) / 448.
            p_quant = (p / p_scale).to(tl.float8e4nv)
        elif QUANT_TYPE == 2:  # e5m2
            p_scale = tl.max(tl.abs(p)) / 57344.
            p_quant = (p / p_scale).to(tl.float8e5)
        else:
            tl.static_assert(False, "Unsupported quant type")

        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, 
                         mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        do_scale = tl.load(DO_scale + step_m // BLOCK_M)

        dv += tl.dot(tl.trans(p_quant), do) * p_scale * do_scale
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()

        dp = tl.dot(do, tl.trans(v)) * do_scale * v_scale
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()

        Di = tl.load(D + offs_m_curr)
        ds = (p * (dp - Di[:, None]))

        if QUANT_TYPE == 0:  # int8
            ds_scale = tl.max(tl.abs(ds)) / 127.
            ds_quant = ((ds / ds_scale) + 0.5 * tl.where(ds >= 0, 1, -1)).to(tl.int8)
        elif QUANT_TYPE == 1:  # e4m3
            ds_scale = tl.max(tl.abs(ds)) / 448.
            ds_quant = (ds / ds_scale).to(tl.float8e4nv)
        elif QUANT_TYPE == 2:  # e5m2
            ds_scale = tl.max(tl.abs(ds)) / 57344.
            ds_quant = (ds / ds_scale).to(tl.float8e5)
        else:
            tl.static_assert(False, "Unsupported quant type")

        dk += tl.dot(tl.trans(ds_quant), q) * q_scale * ds_scale
        # compute dq
        if not (EVEN_M & EVEN_HEADDIM):  
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds_quant, k) * k_scale * ds_scale
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0, eviction_policy="evict_last")
                    dq += tl.dot(ds_quant, k) * k_scale * ds_scale
                    tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q, eviction_policy="evict_last")
                else:
                    dq = tl.load(dq_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0, eviction_policy="evict_last")
                    dq += tl.dot(ds_quant, k) * k_scale * ds_scale
                    tl.store(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), eviction_policy="evict_last")
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds_quant, k) * k_scale * ds_scale
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

        dq_ptrs += BLOCK_M * stride_qs
        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_qs

    dv_ptrs = DV + (offs_n[:, None] * stride_qs + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_qs + offs_d[None, :])
    dk = dk * softmax_scale
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

@triton.jit
def _bwd_kernel(
    Q, K, V, DO, 
    Q_scale, K_scale, V_scale, DO_scale,
    DQ, DK, DV,
    LSE, D,
    softmax_scale,
    stride_qb, stride_qh, stride_qs,
    nheads,
    seqlen_q, seqlen_k, seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    adj = off_b * stride_qb + off_h * stride_qh
    Q += adj; K += adj; V += adj; DO += adj; DQ += adj; DK += adj; DV += adj;

    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded

    Q_scale_ptr = Q_scale + (off_b * nheads + off_h) * tl.cdiv(seqlen_q, BLOCK_M)
    DO_scale_ptr = DO_scale + (off_b * nheads + off_h) * tl.cdiv(seqlen_q, BLOCK_M)
    K_scale_ptr = K_scale + (off_b * nheads + off_h) * tl.cdiv(seqlen_k, BLOCK_N) 
    V_scale_ptr = V_scale + (off_b * nheads + off_h) * tl.cdiv(seqlen_k, BLOCK_N)

    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q, K, V, DO, 
                Q_scale_ptr, K_scale_ptr, V_scale_ptr, DO_scale_ptr,
                DQ, DK, DV,
                LSE, D,
                softmax_scale,
                stride_qs,
                seqlen_q, seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                QUANT_TYPE=QUANT_TYPE,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q, K, V, DO, 
            Q_scale_ptr, K_scale_ptr, V_scale_ptr, DO_scale_ptr,
            DQ, DK, DV,
            LSE, D,
            softmax_scale,
            stride_qs,
            seqlen_q, seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            QUANT_TYPE=QUANT_TYPE,
        )

QUANT_CONFIG = {
    "int8": (0, torch.int8),
    "e4m3": (1, torch.float8_e4m3fn),
    "e5m2": (2, torch.float8_e5m2),
    "none": (3, torch.int8)
}

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, softmax_scale=None):
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]

        batch, seqlen_q, nheads_q, headdim = q.shape
        _, seqlen_k, nheads_k, _ = k.shape

        quant_code, quant_dtype = QUANT_CONFIG["e5m2"]
        assert headdim <= 128, "FlashAttention only support head dimensions up to 128"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda
        softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
        BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
        BLOCK_M = 128
        BLOCK_N = 128
        EVEN_M = seqlen_q % BLOCK_M == 0
        EVEN_N = seqlen_k % BLOCK_N == 0
        EVEN_HEADDIM = headdim == BLOCK_HEADDIM

        NUM_Q_BLOCKS = (seqlen_q + BLOCK_M - 1) // BLOCK_M
        NUM_KV_BLOCKS = (seqlen_k + BLOCK_N - 1) // BLOCK_N

        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        lse = torch.empty((batch, nheads_q, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        tmp = torch.empty((batch, nheads_q, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)

        q_quant = torch.empty(q.shape, dtype=quant_dtype, device=q.device)
        k_quant = torch.empty(k.shape, dtype=quant_dtype, device=k.device)
        v_quant = torch.empty(v.shape, dtype=quant_dtype, device=v.device)

        q_scale = torch.empty((batch, nheads_q, NUM_Q_BLOCKS), device=q.device, dtype=torch.float32)
        k_scale = torch.empty((batch, nheads_q, NUM_KV_BLOCKS), device=k.device, dtype=torch.float32)
        v_scale = torch.empty((batch, nheads_q, NUM_KV_BLOCKS), device=v.device, dtype=torch.float32)

        k_mean = k.mean(dim=1, keepdim=True)
        lse_correction = torch.matmul(q.transpose(1, 2), k_mean.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
        k = k - k_mean

        grid_q = (NUM_Q_BLOCKS, batch * nheads_q)
        grid_kv = (NUM_KV_BLOCKS, batch * nheads_q)
        quant_per_block[grid_q](
            q, q_quant, q_scale,
            q.stride(0), q.stride(2), q.stride(1), 
            q_quant.stride(0), q_quant.stride(2), q_quant.stride(1),
            q_scale.stride(0), q_scale.stride(1),
            sm_scale=softmax_scale * RCP_LN2,
            HEAD_DIM=headdim,
            SEQ_LEN=seqlen_q,
            HEAD_NUM=nheads_q,
            BLOCK=BLOCK_M,
            QUANT_TYPE=quant_code
        )
        quant_per_block[grid_kv](
            k, k_quant, k_scale,
            k.stride(0), k.stride(2), k.stride(1),
            k_quant.stride(0), k_quant.stride(2), k_quant.stride(1),
            k_scale.stride(0), k_scale.stride(1),
            sm_scale=1.0,
            HEAD_DIM=headdim,
            SEQ_LEN=seqlen_k,
            HEAD_NUM=nheads_k,
            BLOCK=BLOCK_N,
            QUANT_TYPE=quant_code
        )
        quant_per_block[grid_kv](
            v, v_quant, v_scale,
            v.stride(0), v.stride(2), v.stride(1),
            v_quant.stride(0), v_quant.stride(2), v_quant.stride(1),
            v_scale.stride(0), v_scale.stride(1),
            sm_scale=1.0,
            HEAD_DIM=headdim,
            SEQ_LEN=seqlen_k,
            HEAD_NUM=nheads_k,
            BLOCK=BLOCK_N,
            QUANT_TYPE=quant_code
        )
        
        stage = 3 if causal else 1
        extra_kern_args = {}
        num_warps = 4 if headdim <= 64 else 8
        _fwd_kernel[grid_q](
            q_quant, k_quant, v_quant, 
            q_scale, k_scale, v_scale,
            o, lse, tmp,
            q_quant.stride(0), q_quant.stride(2), q_quant.stride(1),
            nheads_q,
            seqlen_q, seqlen_k, seqlen_q_rounded,
            headdim,
            seqlen_q // 32,
            seqlen_k // 32,  
            causal,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            QUANT_TYPE=quant_code,
            num_warps=num_warps,
            num_stages=stage,
        )
        ctx.softmax_scale = softmax_scale
        ctx.save_for_backward(q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, o, lse)
        ctx.causal = causal
        lse = (lse * LN2)[:, :, :seqlen_q]
        lse = lse + lse_correction * softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q_quant, k_quant, v_quant, q_scale, k_scale, v_scale, o, lse = ctx.saved_tensors
        with torch.inference_mode():
            dq = torch.empty(q_quant.shape, dtype=do.dtype, device=do.device)
            dk = torch.empty(k_quant.shape, dtype=do.dtype, device=do.device)
            dv = torch.empty(v_quant.shape, dtype=do.dtype, device=do.device)
            if do.stride(-1) != 1:
                do = do.contiguous()

            assert q_quant.stride(-1) == k_quant.stride(-1) == v_quant.stride(-1) == o.stride(-1) == 1
            assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == do.stride(-1) == 1

            batch, seqlen_q, nheads_q, headdim = q_quant.shape
            _, seqlen_k, nheads_k, _ = k_quant.shape
            assert headdim <= 128

            ctx.softmax_scale = ctx.softmax_scale or 1.0 / math.sqrt(headdim)
            
            dq_accum = torch.zeros_like(dq, dtype=torch.float32)
            delta = torch.empty_like(lse)

            BLOCK_M = 128
            BLOCK_N = 128
            BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
            seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
            assert lse.shape == (batch, nheads_q, seqlen_q_rounded)

            NUM_Q_BLOCKS = (seqlen_q + BLOCK_M - 1) // BLOCK_M
            NUM_KV_BLOCKS = (seqlen_k + BLOCK_N - 1) // BLOCK_N

            quant_code, quant_dtype = QUANT_CONFIG["e5m2"]
            do_quant = torch.empty(do.shape, dtype=quant_dtype, device=do.device)
            do_scale = torch.empty((batch, nheads_q, NUM_Q_BLOCKS), device=do.device, dtype=torch.float32)
            # k_quant = k_quant * (ctx.softmax_scale * RCP_LN2) 

            EVEN_M = seqlen_q % BLOCK_M == 0
            EVEN_N = seqlen_k % BLOCK_N == 0
            EVEN_HEADDIM = headdim == BLOCK_HEADDIM

            grid_q = (NUM_Q_BLOCKS, batch * nheads_q)
            _bwd_preprocess_do_o_dot[grid_q](
                o, do, delta,
                o.stride(0), o.stride(2), o.stride(1),
                nheads_q,
                seqlen_q, seqlen_q_rounded,
                headdim,
                BLOCK_M=BLOCK_M, BLOCK_HEADDIM=BLOCK_HEADDIM,
            )
            SEQUENCE_PARALLEL = False
            grid = (
                (NUM_KV_BLOCKS, batch * nheads_q) 
                if SEQUENCE_PARALLEL 
                else (1, batch * nheads_q)
            )
            _bwd_kernel[grid](
                q_quant, k_quant, v_quant, do_quant, 
                q_scale, k_scale, v_scale, do_scale,
                dq_accum, dk, dv,
                lse, delta,
                ctx.softmax_scale,
                q_quant.stride(0), q_quant.stride(2), q_quant.stride(1),
                nheads_q,
                seqlen_q, seqlen_k, seqlen_q_rounded,
                headdim,
                seqlen_q // 32,
                seqlen_k // 32,  # key for triton cache (limit number of compilations)
                IS_CAUSAL=ctx.causal,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                QUANT_TYPE=quant_code,
                num_warps=8,
                num_stages=1,
            )
            dq.copy_(dq_accum * LN2)
        return dq, dk, dv, None, None


@torch.compiler.disable
def sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "bhsd",
    is_causal: bool = False, 
    sm_scale: Optional[float] = None,
    **kwargs: Any):

    assert all(x.is_cuda for x in [q, k, v]), "All tensors must be on the GPU."
    assert q.dtype in [torch.float16, torch.bfloat16], "Tensors must be in FP16 or BF16."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    assert q.shape[-1] == k.shape[-1], "Head dim mismatch between Q and K"
    
    torch.cuda.set_device(q.device)
    
    qkv_format = "".join([i for i in tensor_layout.split("_")[0] if i.isalpha()])

    head_dim = q.size(-1)
    
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    if qkv_format == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    elif qkv_format == "sbhd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    
    else:
        raise ValueError(f"Invalid tensor layout: {qkv_format}")
    
    output =  FlashAttnFunc.apply(q, k, v, is_causal, sm_scale)

    if qkv_format == "bhsd": 
        output = output.permute(0, 2, 1, 3).contiguous()
    elif qkv_format == "bshd":
        output = output.permute(0, 2, 1, 3).contiguous()
    
    
    
    return output