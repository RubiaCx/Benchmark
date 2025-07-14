# Flash Attention Profiling 工具链

这是一个专门用于Flash Attention性能分析的完整工具链，支持自动化测试、最优配置发现和深度性能分析。

## 🎯 功能特性

- **全形状测试**: 自动测试不同的batch size、head数量、序列长度和head维度组合
- **最优配置发现**: 自动寻找并记录每种形状下的最优Triton配置
- **NSys分析**: 系统级性能分析，包括时间线和资源利用率
- **NCU分析**: 内核级详细分析，包括指令吞吐量和内存带宽
- **结果管理**: 自动保存和组织分析结果

## 📁 目录结构

```
profile/
├── README.md           # 本文档
├── config.py           # 配置管理
├── profiler.py         # 主要profiling逻辑
├── runner.py           # 自动化运行器
├── analyzer.py         # 结果分析工具
├── utils.py            # 工具函数
├── shapes/             # 测试形状配置
├── results/            # profiling结果
│   ├── configs/        # 最优配置
│   ├── nsys/          # NSys分析结果
│   └── ncu/           # NCU分析结果
└── scripts/           # 自动化脚本
```

## 🚀 快速开始

### 1. 运行全形状profiling
```bash
python runner.py --mode full --save-configs
```

### 2. 对特定形状进行详细分析
```bash
python runner.py --mode shape --batch 8 --heads 16 --seq-len 1024 --head-dim 128
```

### 3. 仅运行NSys分析
```bash
python runner.py --mode nsys --config-file results/configs/best_config.json
```

### 4. 仅运行NCU分析
```bash
python runner.py --mode ncu --config-file results/configs/best_config.json
```

## 📊 结果分析

使用analyzer.py来分析profiling结果：

```bash
# 生成性能报告
python analyzer.py --generate-report

# 对比不同配置
python analyzer.py --compare-configs

# 分析内存使用模式
python analyzer.py --analyze-memory
```

## ⚙️ 配置选项

主要配置参数在`config.py`中定义，包括：
- 测试形状范围
- profiling参数
- 输出路径
- 工具选项 