# Flash Attention WS Hopper 分析工具

这是一个用于分析Flash Attention WS Hopper性能的综合工具，支持NVTX和NCU分析。

## 快速开始

### 1. 创建分析脚本

首先创建所有必要的分析脚本：

```bash
python flash-attention-WS-hopper.py --mode create_scripts
```

这将创建以下脚本：
- `nvtx_analysis.sh` - NVTX分析脚本
- `ncu_analysis.sh` - NCU分析脚本  
- `compare_analysis.sh` - 对比分析脚本

### 2. 运行基准测试

```bash
python flash-attention-WS-hopper.py --mode benchmark
```

### 3. 详细对比分析

```bash
python flash-attention-WS-hopper.py --mode analysis
```

## 四种配置对比

代码中测试了以下四种配置：

1. **Triton WS FP16 (N_CTX=128)** - 基础版本，短序列
2. **Triton WS FP16 (N_CTX=8192)** - 基础版本，长序列  
3. **Triton WS PP FP16 (N_CTX=128)** - Pipeline版本，短序列
4. **Triton WS PP FP16 (N_CTX=8192)** - Pipeline版本，长序列

## NVTX分析

### 安装NVTX

```bash
pip install nvtx
```

### 运行NVTX分析

```bash
./nvtx_analysis.sh
```

或者手动运行：

```bash
nsys profile -o nvtx_profile --trace=cuda,nvtx --force-overwrite=true \
    python flash-attention-WS-hopper.py --mode nvtx
```

### NVTX标记说明

代码中的NVTX标记：
- `bench_{provider}_B{BATCH}_H{H}_N{N_CTX}_D{HEAD_DIM}` - 基准测试标记
- `attention_ws_B{B}_H{H}_N{N}_D{D}` - 基础attention kernel标记
- `attention_ws_pp_B{B}_H{H}_N{N}_D{D}` - Pipeline attention kernel标记

### 查看NVTX结果

```bash
# 图形界面查看
nsys-ui nvtx_profile.qdrep

# 命令行查看统计信息
nsys stats nvtx_profile.nsys-rep
```

## NCU分析

### 运行NCU分析

```bash
./ncu_analysis.sh
```

这将自动分析所有四种配置，并生成详细报告。

### 手动运行单个配置

```bash
# 分析短序列基础版本
ncu --set full -o ncu_ws_128.ncu-rep --force-overwrite \
    python flash-attention-WS-hopper.py --mode ncu --n_ctx 128 --provider triton-ws-fp16

# 分析长序列Pipeline版本
ncu --set full -o ncu_ws_pp_8192.ncu-rep --force-overwrite \
    python flash-attention-WS-hopper.py --mode ncu --n_ctx 8192 --provider triton-ws-pp-fp16
```

### NCU报告分析

生成的报告包含：
- `ncu_results/ncu_{provider}_{n_ctx}.ncu-rep` - 原始NCU文件
- `ncu_results/report_{provider}_{n_ctx}.txt` - 文本报告

### 查看NCU结果

```bash
# 图形界面查看
ncu-ui ncu_results/ncu_triton-ws-fp16_128.ncu-rep

# 命令行查看详细指标
ncu -i ncu_results/ncu_triton-ws-fp16_128.ncu-rep --page raw
```

## 关键分析指标

### 性能指标

1. **执行时间 (ms)** - kernel执行时间
2. **TFLOPS** - 每秒万亿次浮点运算
3. **内存带宽 (GB/s)** - 内存访问带宽
4. **内存使用 (MB)** - 总内存使用量

### Autotune配置

对于每个配置，会显示最佳的autotune参数：
- `BLOCK_M/BLOCK_N` - 块大小
- `NUM_BUFFERS` - 缓冲区数量
- `NUM_MMA_WARPS` - MMA warp数量
- `NUM_MMA_GROUPS` - MMA组数量
- `num_warps/num_stages` - warp和stage数量

### 对比分析

1. **加速比** - Pipeline版本相对基础版本的性能提升
2. **扩展效率** - 长序列相对短序列的效率
3. **资源利用率** - GPU资源利用情况

## 深度分析

### 内存访问模式

通过NCU分析可以看到：
- Global Memory Load/Store效率
- L1/L2 Cache命中率
- Memory Bank Conflicts

### 计算效率

- Tensor Core利用率
- Warp执行效率
- 指令吞吐量

### Pipeline效率

Pipeline版本的优势：
- 更好的内存访问重叠
- 减少的内存延迟
- 提高的计算资源利用率

## 示例输出

```
================================================================================
AutoTune 最佳配置信息 - triton-ws-fp16
形状: BATCH=1, H=32, N_CTX=128, HEAD_DIM=128
执行时间: 0.234 ms
================================================================================
最佳配置参数:
  BLOCK_M: 128
  BLOCK_N: 128
  NUM_BUFFERS: 2
  NUM_MMA_WARPS: 8
  NUM_MMA_GROUPS: 2
  num_warps: 4
  num_stages: 0
  理论FLOPS: 0.07 TFLOPS
  实际性能: 23.34 TFLOPS
  内存带宽: 1.75 GB/s
```

## 常见问题

### Q: NVTX分析结果为空？
A: 确保已安装nvtx包，并且使用正确的nsys命令行参数。

### Q: NCU分析失败？
A: 检查是否有足够的GPU权限，某些系统需要sudo权限运行ncu。

### Q: 性能结果不一致？
A: 确保GPU处于稳定状态，关闭其他GPU程序，多次运行取平均值。

### Q: 如何解读autotune结果？
A: 更大的BLOCK_M/BLOCK_N通常意味着更好的并行性，但也会增加内存使用；NUM_MMA_GROUPS控制着并行度和资源分配。

## 进阶用法

### 自定义配置测试

修改`configs`数组来测试不同的配置参数：

```python
configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, ...}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, ...}),
    # 添加更多配置
]
```

### 自定义测试形状

修改基准测试中的形状参数：

```python
x_vals=[256, 512, 1024, 2048, 4096, 8192, 16384]  # 不同的N_CTX值
BATCH, N_HEADS, HEAD_DIM = 2, 16, 64  # 不同的形状
```

这个工具为深入理解Flash Attention的性能特征提供了全面的分析能力。 