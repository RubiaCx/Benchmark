import pytest
import torch
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

class add_ld_library_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.os_environ = os.environ.copy()
        library_path = os.environ.get("LD_LIBRARY_PATH")
        if not library_path:
            os.environ["LD_LIBRARY_PATH"] = self.path
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{library_path}:{self.path}"

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ = self.os_environ.copy()

try:
    torch_lib_path = os.path.join(os.path.dirname(__file__), "lib")
    with add_ld_library_path(torch_lib_path):
        from flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_v3
    HAS_FLASH_V3 = True
except (ImportError, IOError, AttributeError):
    try:
        from fa3.hopper.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_v3

        HAS_FLASH_V3 = True
    except (ImportError, IOError, AttributeError):
        HAS_FLASH_V3 = False

try:
    from sage_attention import sage_attention
    HAS_SAGE = True
except (ImportError, IOError, AttributeError):
    HAS_SAGE = False

try:
    from sage_attention_2 import sage_attention as sage_attention_2
    HAS_SAGE_2 = True
except (ImportError, IOError, AttributeError):
    HAS_SAGE_2 = False

try:
    import sageattention._qattn_sm90 as sage_cuda
    HAS_SAGE_CUDA_ATTN = True
    sage_cuda_attn = sage_cuda.qk_int8_sv_f8_accum_f32_attn_inst_buf
except (ImportError, IOError, AttributeError):
    HAS_SAGE_CUDA_ATTN = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
TORCH_HAS_SPDA = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
BATCH, N_HEADS = 1, 48
# vary seq length for fixed head and batch=4
configs = []
for HEAD_DIM in [64]:
    for mode in ["fwd"]:
        for causal in [True, False]:
            for warp_specialize in [False, True]: # if is_blackwell() else [False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2**i for i in range(8, 15)],
                        line_arg="provider",
                        line_vals=["flash"] 
                                 +(["flash-v3"] if HAS_FLASH_V3 else [])
                                 +(["torch-spda"] if TORCH_HAS_SPDA else [])
                                 +(["sage"] if HAS_SAGE else [])
                                 +(["sage-cuda"] if HAS_SAGE_CUDA_ATTN else []),
                        line_names=["Flash-2"]
                                  +(["Flash-3"] if HAS_FLASH_V3 else [])
                                  +(["Torch SPDA"] if TORCH_HAS_SPDA else [])
                                  +(["Sage"] if HAS_SAGE else [])
                                  +(["Sage-cuda"] if HAS_SAGE_CUDA_ATTN else []),
                        plot_type="bar",
                        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-"), ("purple", "-")],
                        # ylabel="TFLOPS",
                        ylabel="Time (ms)",
                        plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                        args={
                            "H": N_HEADS,
                            "BATCH": BATCH,
                            "HEAD_DIM": HEAD_DIM,
                            "mode": mode,
                            "causal": causal,
                            "warp_specialize": warp_specialize,
                        },
                    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash-v3":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_v3(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "torch-spda":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "sage":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: sage_attention(q, k, v, tensor_layout="bhsd", is_causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "sage-cuda":
        q = torch.randint(-95, 95, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device)
        k = torch.randint(-95, 95, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device)
        v = torch.randn((BATCH, H, HEAD_DIM, N_CTX), dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
        o = torch.empty((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=device)
        q_scale = torch.randn(BATCH, H, N_CTX // 64 * 4, dtype=torch.float, device="cuda")
        k_scale = torch.randn(BATCH, H, N_CTX // 128, dtype=torch.float, device="cuda")
        v_scale = torch.randn(BATCH, H, N_CTX, dtype=torch.float, device="cuda")
        fn = lambda: sage_cuda_attn(q, k, v, o, q_scale, k_scale, 1, causal, 2, 1.3, 0)
        ms = triton.testing.do_bench(fn)
    # flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    # total_flops = 2 * flops_per_matmul
    # if causal:
    #     total_flops *= 0.5
    # if mode == "bwd":
    #     total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    # return total_flops * 1e-12 / (ms * 1e-3)
    return ms


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)