# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ref_program(A, B):
    """
    A reference matrix multiplication program, used to compare performance.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix with shape (M, K).
    B : numpy.ndarray
        The matrix with shape (N, K).

    Returns
    -------
    np.ndarray
        The result of A @ B.T, shape (M, N).
    """
    return A @ B.T


def benchmark_ref_program(A, B, warmup=10, rep=100):
    """
    Benchmark the reference implementation.
    
    Parameters
    ----------
    A, B : numpy arrays
        Input matrices
    warmup : int
        Number of warmup iterations
    rep : int
        Number of benchmark iterations
        
    Returns
    -------
    float
        Average execution time in seconds
    """
    # Warmup
    for _ in range(warmup):
        _ = ref_program(A, B)
    
    # Benchmark
    times = []
    for _ in range(rep):
        start_time = time.perf_counter()
        _ = ref_program(A, B)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.mean(times)


def get_configs(M, N, K, with_roller=False):
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


def create_matmul_kernel(M, N, K, with_roller):
    """
    Create an autotuned matrix multiplication kernel.
    
    Returns
    -------
    Compiled kernel function
    """
    @autotune(
        configs=get_configs(M, N, K, with_roller),
        warmup=3,
        rep=20,
    )
    @jit(out_idx=[2],)
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
        """
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):

                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)

                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                        policy=policy,
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    return kernel()


def benchmark_tilelang_kernel(kernel_func, A, B, C, warmup=10, rep=100):
    """
    Benchmark the TileLang kernel.
    
    Parameters
    ----------
    kernel_func : compiled kernel
        The TileLang kernel function
    A, B, C : arrays
        Input and output tensors
    warmup : int
        Number of warmup iterations
    rep : int
        Number of benchmark iterations
        
    Returns
    -------
    float
        Average execution time in seconds
    """
    # Warmup
    for _ in range(warmup):
        kernel_func(A, B, C)
    
    # Benchmark
    times = []
    for _ in range(rep):
        start_time = time.perf_counter()
        kernel_func(A, B, C)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.mean(times)


def benchmark_single_size(M, N, K, with_roller=False, warmup=10, rep=100):
    """
    Benchmark a single matrix size.
    
    Returns
    -------
    dict
        Results containing latencies and configurations
    """
    print(f"\nBenchmarking M={M}, N={N}, K={K}")
    
    # Create test data
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(N, K).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    
    # Benchmark reference implementation
    print("  Testing reference implementation...")
    try:
        ref_latency = benchmark_ref_program(A, B, warmup=warmup, rep=rep)
        print(f"  Reference latency: {ref_latency:.6f}s")
    except Exception as e:
        print(f"  Reference failed: {e}")
        ref_latency = None
    
    # Create and benchmark TileLang kernel
    print("  Creating TileLang kernel...")
    try:
        kernel_result = create_matmul_kernel(M, N, K, with_roller)
        
        # Extract the best configuration and latency
        if hasattr(kernel_result, 'latency'):
            best_latency = kernel_result.latency
            best_config = kernel_result.config
        else:
            # If autotune doesn't return timing info, benchmark manually
            print("  Benchmarking TileLang kernel manually...")
            best_latency = benchmark_tilelang_kernel(kernel_result, A, B, C, warmup=warmup, rep=rep)
            best_config = "Unknown"
        
        print(f"  TileLang latency: {best_latency:.6f}s")
        print(f"  Best config: {best_config}")
        
    except Exception as e:
        print(f"  TileLang failed: {e}")
        best_latency = None
        best_config = None
    
    return {
        'M': M, 'N': N, 'K': K,
        'tilelang_latency': best_latency,
        'ref_latency': ref_latency,
        'best_config': best_config
    }


def benchmark_multiple_sizes(sizes, with_roller=False, save_plot=True):
    """
    Benchmark GEMM performance across multiple matrix sizes.
    
    Parameters
    ----------
    sizes : list of int
        List of matrix sizes to test (M=N=K for each size)
    with_roller : bool
        Whether to use BitBLAS roller for configuration search
    save_plot : bool
        Whether to save performance plot
        
    Returns
    -------
    dict
        Dictionary containing benchmark results
    """
    results = {
        'sizes': [],
        'tilelang_tflops': [],
        'ref_tflops': [],
        'speedup': [],
        'best_configs': []
    }
    
    print("=== TileLang GEMM Benchmark ===")
    print(f"Testing sizes: {sizes}")
    print(f"With roller: {with_roller}")
    print("-" * 50)
    
    for size in sizes:
        M = N = K = size
        total_flops = 2 * M * N * K
        
        try:
            # Run benchmark for current size
            result = benchmark_single_size(M, N, K, with_roller)
            
            tilelang_latency = result['tilelang_latency']
            ref_latency = result['ref_latency']
            best_config = result['best_config']
            
            # Calculate performance metrics
            if tilelang_latency is not None:
                tilelang_tflops = total_flops / tilelang_latency * 1e-12
            else:
                tilelang_tflops = 0
                
            if ref_latency is not None:
                ref_tflops = total_flops / ref_latency * 1e-12
                speedup = ref_latency / tilelang_latency if tilelang_latency else 0
            else:
                ref_tflops = 0
                speedup = 0
            
            # Store results
            results['sizes'].append(size)
            results['tilelang_tflops'].append(tilelang_tflops)
            results['ref_tflops'].append(ref_tflops)
            results['speedup'].append(speedup)
            results['best_configs'].append(best_config)
            
            # Print results
            print(f"\nSize {size} Results:")
            if tilelang_tflops > 0:
                print(f"  TileLang: {tilelang_tflops:.3f} TFLOPS")
            else:
                print("  TileLang: Failed")
                
            if ref_tflops > 0:
                print(f"  Reference: {ref_tflops:.3f} TFLOPS")
                print(f"  Speedup: {speedup:.2f}x")
            else:
                print("  Reference: Failed")
            
        except Exception as e:
            print(f"Error testing size {size}: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for i, size in enumerate(results['sizes']):
        if results['tilelang_tflops'][i] > 0:
            print(f"Size {size:5d}: {results['tilelang_tflops'][i]:6.2f} TFLOPS", end="")
            if results['speedup'][i] > 0:
                print(f" (Speedup: {results['speedup'][i]:5.2f}x)")
            else:
                print(" (No reference)")
        else:
            print(f"Size {size:5d}: Failed")
    
    # Create performance plot
    if save_plot and results['sizes']:
        valid_results = [(i, size) for i, size in enumerate(results['sizes']) 
                        if results['tilelang_tflops'][i] > 0]
        
        if valid_results:
            plt.figure(figsize=(12, 8))
            
            # Plot TFLOPS comparison
            plt.subplot(2, 1, 1)
            valid_indices = [i for i, _ in valid_results]
            valid_sizes = [size for _, size in valid_results]
            
            x_pos = np.arange(len(valid_sizes))
            width = 0.35
            
            tilelang_vals = [results['tilelang_tflops'][i] for i in valid_indices]
            ref_vals = [results['ref_tflops'][i] for i in valid_indices]
            
            plt.bar(x_pos - width/2, tilelang_vals, width, 
                    label='TileLang', alpha=0.8, color='blue')
            
            # Only plot reference if we have valid data
            if any(x > 0 for x in ref_vals):
                plt.bar(x_pos + width/2, ref_vals, width,
                        label='Reference', alpha=0.8, color='orange')
            
            plt.xlabel('Matrix Size (M=N=K)')
            plt.ylabel('Performance (TFLOPS)')
            plt.title('TileLang GEMM Performance Comparison')
            plt.xticks(x_pos, valid_sizes)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot speedup if we have valid reference data
            valid_speedup = [results['speedup'][i] for i in valid_indices if results['speedup'][i] > 0]
            if valid_speedup:
                plt.subplot(2, 1, 2)
                speedup_sizes = [valid_sizes[j] for j, i in enumerate(valid_indices) 
                               if results['speedup'][i] > 0]
                plt.plot(speedup_sizes, valid_speedup, 'ro-', linewidth=2, markersize=6)
                plt.xlabel('Matrix Size (M=N=K)')
                plt.ylabel('Speedup vs Reference')
                plt.title('TileLang Speedup vs Reference Implementation')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"tilelang_gemm_benchmark{'_roller' if with_roller else ''}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"\nPerformance plot saved as: {plot_filename}")
            
            # Show plot if running interactively
            try:
                plt.show()
            except:
                pass
    
    return results


if __name__ == "__main__":
    # Parse command-line arguments for matrix dimensions
    parser = argparse.ArgumentParser(description="TileLang GEMM Benchmark")
    parser.add_argument("--m", type=int, default=1024, help="Matrix dimension M (single test)")
    parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N (single test)")
    parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K (single test)")
    parser.add_argument(
        "--with_roller",
        action="store_true",
        help="Whether to enable BitBLAS roller for search space",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark across multiple sizes [2^8 to 2^14]"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="Custom list of sizes to benchmark (e.g., --sizes 256 512 1024)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    args = parser.parse_args()

    if args.benchmark or args.sizes:
        # Multi-size benchmark mode
        if args.sizes:
            sizes = args.sizes
        else:
            # Default: 2^8 to 2^14 (256 to 16384)
            sizes = [2**i for i in range(8, 15)]
        
        results = benchmark_multiple_sizes(sizes, args.with_roller)
        
    else:
        # Single size test mode
        M, N, K = args.m, args.n, args.k
        
        result = benchmark_single_size(M, N, K, args.with_roller, args.warmup, args.rep)
        
        total_flops = 2 * M * N * K
        
        # Print results
        print(f"\n=== Single Size Benchmark Results ===")
        print(f"Matrix size: {M} x {N} x {K}")
        print(f"Total FLOPs: {total_flops:.2e}")
        
        if result['tilelang_latency']:
            tilelang_tflops = total_flops / result['tilelang_latency'] * 1e-12
            print(f"TileLang: {tilelang_tflops:.3f} TFLOPS")
            print(f"Best config: {result['best_config']}")
        else:
            print("TileLang: Failed")
        
        if result['ref_latency']:
            ref_tflops = total_flops / result['ref_latency'] * 1e-12
            speedup = result['ref_latency'] / result['tilelang_latency'] if result['tilelang_latency'] else 0
            print(f"Reference: {ref_tflops:.3f} TFLOPS")
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("Reference: Failed")