import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import triton.profiler as proton

import triton.tlx.language as tlx


from typing import Optional


# TileLang imports
import itertools
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit

DEVICE = triton.runtime.driver.active.get_active_torch_device()

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def is_hopper():
    return torch.cuda.get_device_capability()[0] == 9

def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args.get("FP8_OUTPUT", False) else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret

HAS_TENSOR_DESC = supports_tma() and hasattr(tl, "make_tensor_descriptor")
HAS_HOST_TENSOR_DESC = supports_tma() and hasattr(triton.tools.tensor_descriptor, "TensorDescriptor")
HAS_WARP_SPECIALIZE = supports_ws() and HAS_TENSOR_DESC

# %% 
# Algorithm
# ---------
# 
# 所有 Triton 实现都首先将输出矩阵 C 按 BLOCK_SIZE_M × BLOCK_SIZE_N 分块，并按 先行后列 的 "super-grouping" 顺序遍历
#   计算行块数 ⌈M/BLOCK_SIZE_M⌉ 与列块数 ⌈N/BLOCK_SIZE_N⌉
#   每组内先连续计算 GROUP_SIZE_M 个行块，再切换列块，保证多行块共用同一列切片的 B，提升 L2 缓存命中率


# %%
# Implementations
# ---------------
#
# Naive Matmul
# ~~~~~~~~~~~
# 对每个 (pid_m,pid_n) 实例，按 K 维度切片迭代：
#   计算 A 子块 BLOCK_SIZE_M×BLOCK_SIZE_K 与 B 子块 BLOCK_SIZE_K×BLOCK_SIZE_N 指针。
#   载入并通过 tl.dot 累加到本地累加器。
#
# 充分利用线程并行度；所有内存访存通过标准 tl.load/tl.store，可利用片上 L1/L2 缓存。
# 无预取、无线程组复用，K 维块仅单次加载，带宽未充分利用。

@triton.jit
def _compute_pid(pid, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n
    
def matmul_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w, pre_hook=pre_hook) \
        for BM in [64, 128, 256] \
        for BN in [64, 128, 256] \
        for BK in [32, 64] \
        for s in ([3, 4]) \
        for w in [4, 8] \
    ]

@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_naive(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c

# %%
# Implementations
# ---------------
#
# Tensor Memory Access Matmul
# ~~~~~~~~~~~
# 使用 tl.make_tensor_descriptor 或 TensorDescriptor 描述 A、B、C 张量的块形状与内存布局
# 在每个 K 切片中，调用 TMA 指令一次性从全局内存批量加载整个块，无需逐元素 tl.load
# tl.dot(a,b.T,acc) 直接完成 M×N 和 K×N 的乘加运算

def matmul_tma_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    if EPILOGUE_SUBTILE:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]

@triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma(a_desc, b_desc, c_desc,  #
                      M, N, K,  #
                      BLOCK_SIZE_M: tl.constexpr,  #
                      BLOCK_SIZE_N: tl.constexpr,  #
                      BLOCK_SIZE_K: tl.constexpr,  #
                      GROUP_SIZE_M: tl.constexpr,  #
                      FP8_OUTPUT: tl.constexpr,  #
                      WARP_SPECIALIZE: tl.constexpr,  #
                      ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)

def matmul_tma(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    matmul_kernel_tma[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c

# %%
# Implementations
# ---------------
#
# Persistent Matmul
# ~~~~~~~~~~~~~~~~~~
# 以 GPU 上的 SM 数量（NUM_SMS）为线程组数，每组通过步长为 NUM_SMS 的循环遍历所有 tile，复用同一线程组完成多块计算
# 内部仍按 super-grouping 调度，每次循环载入一块 A、B，计算并写回
# 
# 线程组复用：减少 kernel 启动开销，在小至中等矩阵尺寸下提升效率
# 负载均衡：通过动态分配 tile 给 SM，避免浪费
# 局限：未引入预取或 TMA，对大矩阵带宽压力仍大

@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                             M, N, K,  #
                             stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn,  #
                             BLOCK_SIZE_M: tl.constexpr,  #
                             BLOCK_SIZE_N: tl.constexpr,  #
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr,  #
                             NUM_SMS: tl.constexpr,  #
                             ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)

def matmul_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
    )
    return c

# %%
# Implementations
# ---------------
#
# TMA Persistent Matmul
# ~~~~~~~~~~~~~~~~~~~~~~
# 结合 Persistent 内核与 TMA，将批量加载与线程组复用合并
#   每次循环通过 TMA 批量加载下一个 K 切片
#   同一线程组跨多 tile、跨多 K 切片循环计算
# 通过 EPILOGUE_SUBTILE 控制输出子块拆分

def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE": SUBTILE}, num_stages=s, num_warps=w, pre_hook=pre_hook)  #
        for BM in [64, 128, 256]  #
        for BN in [64, 128, 256]  #
        for BK in [32, 64]  #
        for s in ([2, 3, 4])  #
        for w in [4, 8]  #
        for SUBTILE in [True, False]  #
    ]

@triton.autotune(
    configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "WARP_SPECIALIZE"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 WARP_SPECIALIZE: tl.constexpr,  #
                                 ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)

def matmul_tma_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_M = META["BLOCK_SIZE_M"]
        BLOCK_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
    )
    return c

# %%
# Implementations
# ---------------
#
# Descriptor Persistent Matmul
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prune_invalid_configs(configs, named_args, **kwargs):
    FLATTEN = kwargs["FLATTEN"]
    return [conf for conf in configs if not (conf.kwargs.get("EPILOGUE_SUBTILE", True) and FLATTEN is False)]

@triton.autotune(
        configs=matmul_tma_persistent_get_configs(), 
        key=["M", "N", "K", "WARP_SPECIALIZE", "FLATTEN"], 
        prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    FLATTEN: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)

def matmul_descriptor_persistent(a, b, warp_specialize: bool):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    flatten = False if (warp_specialize and is_hopper()) else True
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a, b, c,  #
        M, N, K,  # 
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        FLATTEN=flatten,
    )
    return c

# %%
# Implementations
# ---------------
#
# TMA Pipelined Matmul
# ~~~~~~~~~~~~~~~~~~~~
# 

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8, "NUM_STAGES" : s}, num_warps=w) \
        for BM in [64, 128, 256] \
        for BN in [64, 128, 256] \
        for BK in [32, 64] \
        for s in [2, 3, 4] \
        for w in [8] \
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_tma_pipelined(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_STAGES: tl.constexpr  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # offset computation
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # allocate NUM_STAGES buffers
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES)

    # prefetch (pipelining) for NUM_STAGES - 1 buffers
    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a = tlx.local_view(buffers_A, i)
        b = tlx.local_view(buffers_B, i)
        token_a = tlx.async_load(a_ptrs, a, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K)
        token_b = tlx.async_load(b_ptrs, b, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        tlx.async_load_commit_group([token_a, token_b])

    # main K loop
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # identify the buffer index for the current iteration
        buf = k % NUM_STAGES
        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)

        # wait for buffers to be ready
        tlx.async_load_wait_group(NUM_STAGES - 2)

        # do the mma
        acc = tlx.async_dot(a_k, b_k, acc)

        # prefetch for i-th iteration, i.e, NUM_STAGES - 1 ahead
        i = k + NUM_STAGES - 1
        a_next = tlx.local_view(buffers_A, i % NUM_STAGES)
        b_next = tlx.local_view(buffers_B, i % NUM_STAGES)
        # wait for the previous MMA using this buffer to complete
        acc = tlx.async_dot_wait(NUM_STAGES-1, acc)
        # prefetch
        token_a = tlx.async_load(a_ptrs, a_next, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K)
        token_b = tlx.async_load(b_ptrs, b_next, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K)
        tlx.async_load_commit_group([token_a, token_b])
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # wait for last mma to complete
    acc = tlx.async_dot_wait(0, acc)
    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_pipelined(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_tma_pipelined[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c

# %%
# API
# ---
#
# cuBLAS Matmul
# ~~~~~~~~~~~~~

def cublas_matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                      {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2. * M * N * K}):
        cublas.matmul(a, b, c)
    return c

# %%
# API
# ---
#
# Torch Matmul
# ~~~~~~~~~~~~

def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    c = torch.matmul(a, b.T)
    return c

# %%
# TileLang Implementation
# ----------------------

def get_tilelang_configs(M, N, K, with_roller=False):
    """
    Generate a list of configuration dictionaries that will be used for tuning.
    
    Parameters
    ----------
    with_roller : bool
        Whether to enable bitblas roller to deduce search spaces

    Returns
    -------
    list of dict
        Each configuration dict includes various block sizes, pipeline stages,
        thread numbers, and other parameters to explore during autotuning.
    """
    if with_roller:
        from tilelang.carver.template import MatmulTemplate
        from tilelang.carver.arch import CUDA
        from tilelang.carver.roller.rasterization import NoRasterization
        arch = CUDA("cuda")
        topk = 10

        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float",
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"

        roller_hints = carve_template.recommend_hints(topk=topk)

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage
            config["thread_num"] = block_rows * block_cols * 32
            config["policy"] = T.GemmWarpPolicy.from_warp_partition(block_rows, block_cols)
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
        for config in configs:
            print(config)
    else:

        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        policy = [T.GemmWarpPolicy.Square]
        enable_rasterization = [True, False]
        _configs = list(
            itertools.product(
                block_M,
                block_N,
                block_K,
                num_stages,
                thread_num,
                policy,
                enable_rasterization,
            ))

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "policy": c[5],
                "enable_rasteration": c[6],  # keep param name for backward-compat
            } for c in _configs
        ]
    return configs


def matmul_tilelang(M, N, K, with_roller=False):
    """
    Create an autotuned matrix multiplication kernel for matrices of shape:
      - A: (M, K)
      - B: (N, K)
      - C: (M, N)

    Parameters
    ----------
    M : int
        The dimension M of the matrix multiplication.
    N : int
        The dimension N of the matrix multiplication.
    K : int
        The dimension K of the matrix multiplication.

    Returns
    -------
    Function
        The optimized TileLang kernel function
    """

    # Decorate the kernel with autotune & jit, specifying:
    #  - Tuning config list
    #  - Profiling keys
    #  - Warmup and repetition counts for better measurement
    #  - A reference program for correctness verification
    #  - The "tvm" profiler backend
    #  - HIP as the compilation target (modify as needed for your hardware)

    @autotune(
        configs=get_tilelang_configs(M, N, K, with_roller),
        warmup=3,
        rep=20,
    )
    @jit(out_idx=[2])
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        policy=None,
        enable_rasteration=None,
    ):
        """
        The actual kernel to compute C = A @ B^T.

        Parameters
        ----------
        block_M : int
            Block size in M dimension.
        block_N : int
            Block size in N dimension.
        block_K : int
            Block size in K dimension.
        num_stages : int
            Number of pipelined stages (for asynchronous load).
        thread_num : int
            Number of threads to use per block.
        enable_rasteration : bool
            Whether to enable rasterization (swizzling) optimization.
        k_pack : int
            K dimension packing factor to improve memory coalescing.

        Returns
        -------
        Function
            A TVM Tensor Language function (T.prim_func) that computes matmul.
        """
        # Use half-precision for input data to reduce memory bandwidth,
        # accumulate in float for better numerical accuracy
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            """
            The compiled TVM function for block-level matrix multiplication.

            - We divide the entire (M, N) domain into blocks of shape
              (block_M, block_N).
            - Each block has its own allocated shared memory for sub-blocks
              of A and B.
            - The partial results go into C_local, and then we copy them back
              to global memory C.
            """
            # Bind x-dimension to block index in N,
            #     y-dimension to block index in M.
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):

                # Allocate shared memory for A sub-block of shape (block_M, block_K)
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                # Allocate shared memory for B sub-block of shape (block_N, block_K)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                # Allocate a local fragment for intermediate accumulation
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # Allocate a shared memory for C sub-block of shape (block_M, block_N)
                C_shared = T.alloc_shared((block_M, block_N), dtype)

                # Enable (or disable) swizzling optimization
                T.use_swizzle(panel_size=10, enable=enable_rasteration)

                # Clear out the accumulation buffer
                T.clear(C_local)

                # Loop over sub-blocks in K dimension, pipelined by num_stages
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # Load a sub-block of A from global memory into A_shared
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    # Load a sub-block of B from global memory into B_shared
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    # Perform a partial matrix multiplication:
                    #   C_local += A_shared @ B_shared^T
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                        policy=policy,
                    )
                # Write back the results from C_local to the global memory C
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    return kernel()


# Global cache for TileLang kernels
_tilelang_kernel_cache = {}

def tilelang_matmul(a, b, with_roller=False):
    """
    TileLang matrix multiplication wrapper function.
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    # Create cache key
    cache_key = (M, N, K, with_roller)
    
    # Get or create the optimized kernel
    if cache_key not in _tilelang_kernel_cache:
        try:
            _tilelang_kernel_cache[cache_key] = matmul_tilelang(M, N, K, with_roller)
        except Exception as e:
            print(f"Failed to create TileLang kernel for size ({M}, {N}, {K}): {e}")
            # Fallback to PyTorch implementation
            return torch.matmul(a, b).to(dtype)
    
    kernel = _tilelang_kernel_cache[cache_key]
    
    # Execute the kernel - TileLang handles output allocation automatically
    try:
        result = kernel(a, b)
        return result.to(dtype)
    except Exception as e:
        print(f"TileLang kernel execution failed for size ({M}, {N}, {K}): {e}")
        # Fallback to PyTorch implementation
        return torch.matmul(a, b).to(dtype)


# %%
# Benchmark
# ---------

def run_test(expect, fn, a, b, label, enabled=True):
    print(f"  {label}: ...", end="")
    if enabled:
        actual = fn(a, b)
        passed = torch.allclose(expect, actual.to(expect.dtype), atol=1.0)
        icon = "✅" if passed else "❌"
    else:
        icon = "⭕"
    print(f"\r  {label}: {icon}  ")


if __name__ == "__main__":
    torch.manual_seed(0)
    TORCH_HAS_FP8 = False

    configs = []
    for fp8_inputs in [False]:
        if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["K"],  # 只变化 K
                x_vals=[1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],  # K 的值
                line_arg="provider",  
                line_vals=["naive", "persistent", "tma", "tma_persistent", "descriptor_persistent"] if fp8_inputs \
                    else ['cublas', "naive", "tma", "persistent", "tma_persistent", "descriptor_persistent", "tilelang"], 
                line_names=["Naive", "Persistent", "TMA", "TMA Persistent", "Descriptor Persistent"] if fp8_inputs \
                    else ['cuBLAS', "Naive", "TMA", "Persistent", "TMA Persistent", "Descriptor Persistent", "TileLang"],  
                styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("orange", "-"), ("purple", "-"), ("yellow", "-"), ("cyan", "-")],
                ylabel="TFLOPS", 
                plot_name="mat-M8192-N2K-" + ("fp16" if not fp8_inputs else "fp8"),  # 修改图表名称
                # plot_type="bar",
                args={"fp8_inputs": fp8_inputs},
            ))

    @triton.testing.perf_report(configs)
    def benchmark(K, provider, fp8_inputs):  # 函数签名改为只接收 K
        M = 8192  # 固定 M
        N = K * 2  # N = K * 2
        
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        
        # 根据provider选择正确的矩阵形状
        if provider in ['tma', 'tma_persistent', 'descriptor_persistent']:
            # 这些函数期望 b 是转置的，形状为 (N, K)
            b = torch.randn((N, K), device=DEVICE, dtype=torch.float16)
        else:
            # 标准形状 (K, N)
            b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        
        if TORCH_HAS_FP8 and fp8_inputs:
            a = a.to(torch.float8_e5m2)
            if provider not in ['tma', 'tma_persistent', 'descriptor_persistent']:
                b = b.T  # 对于标准形状的b，需要转置
            b = b.to(torch.float8_e5m2)
            
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'cublas':
            if b.shape == (N, K):  # 如果b是转置形状，需要转置回来
                b_for_cublas = b.T
            else:
                b_for_cublas = b
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: torch.matmul(a, b_for_cublas), quantiles=quantiles, return_mode="max")
        elif provider == 'naive':
            if b.shape == (N, K):  # 如果b是转置形状，需要转置回来
                b_for_naive = b.T
            else:
                b_for_naive = b
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: matmul_naive(a, b_for_naive), quantiles=quantiles, return_mode="max")
        elif provider == 'tma':
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: matmul_tma(a, b, warp_specialize=True), quantiles=quantiles, return_mode="max")
        elif provider == 'persistent':
            if b.shape == (N, K):  # 如果b是转置形状，需要转置回来
                b_for_persistent = b.T
            else:
                b_for_persistent = b
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: matmul_persistent(a, b_for_persistent), quantiles=quantiles, return_mode="max")
        elif provider == 'tma_persistent':
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: matmul_tma_persistent(a, b, warp_specialize=True), quantiles=quantiles, return_mode="max")
        elif provider == 'descriptor_persistent':
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: matmul_descriptor_persistent(a, b, warp_specialize=True), quantiles=quantiles, return_mode="max")
        # elif provider == 'pipelined':
        #     if b.shape == (N, K):  # 如果b是转置形状，需要转置回来
        #         b_for_pipelined = b.T
        #     else:
        #         b_for_pipelined = b
        #     ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: matmul_pipelined(a, b_for_pipelined), quantiles=quantiles, return_mode="max")
        elif provider == 'tilelang':
            if b.shape == (N, K):  # 如果b是转置形状，需要转置回来
                b_for_tilelang = b.T
            else:
                b_for_tilelang = b
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(lambda: tilelang_matmul(a, b_for_tilelang), quantiles=quantiles, return_mode="max")

        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    # Run correctness tests first
    # print("\n=== Correctness Tests ===")
    # M, N, K = 256, 256, 256  # Use same size as benchmark for TileLang compatibility
    # a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    # b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    
    # # Compute reference result
    # ref_result = torch.matmul(a, b)
    
    # # Test each implementation
    # run_test(ref_result, lambda a, b: torch.matmul(a, b), a, b, "PyTorch", enabled=True)
    # run_test(ref_result, matmul_naive, a, b, "Triton Naive", enabled=True)
    # run_test(ref_result, lambda a, b: matmul_tma(a, b, warp_specialize=True), a, b, "Triton TMA", enabled=supports_tma())
    # run_test(ref_result, matmul_persistent, a, b, "Triton Persistent", enabled=True)
    # run_test(ref_result, lambda a, b: matmul_tma_persistent(a, b, warp_specialize=True), a, b, "Triton TMA Persistent", enabled=supports_tma())
    # run_test(ref_result, lambda a, b: matmul_descriptor_persistent(a, b, warp_specialize=True), a, b, "Triton Descriptor Persistent", enabled=supports_ws())
    # run_test(ref_result, matmul_pipelined, a, b, "Triton Pipelined", enabled=True)
    
    # # Test TileLang implementation
    # try:
    #     run_test(ref_result, tilelang_matmul, a, b, "TileLang", enabled=True)
    # except Exception as e:
    #     print(f"  TileLang: ❌ (Error: {e})")


    print("\n=== Running Benchmarks ===")
    benchmark.run(show_plots=True, print_data=True, save_path=".")
