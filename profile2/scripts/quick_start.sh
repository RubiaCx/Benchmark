#!/bin/bash

# Flash Attention Profiling 快速启动脚本

echo "🚀 Flash Attention Profiling 快速启动"
echo "======================================"

# 切换到profile目录
cd "$(dirname "$0")/.."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请安装Python3"
    exit 1
fi

# 检查系统要求
echo "🔍 检查系统要求..."
python3 runner.py --mode check

if [ $? -ne 0 ]; then
    echo "❌ 系统检查失败，请检查环境配置"
    exit 1
fi

# 询问用户选择运行模式
echo ""
echo "请选择运行模式:"
echo "1. 快速测试 (推荐新手)"
echo "2. 完整profiling"
echo "3. 自定义形状"
echo "4. 仅NSys分析"
echo "5. 仅NCU分析"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "🧪 运行快速测试..."
        python3 runner.py --mode quick --config quick --save-configs
        ;;
    2)
        echo "🔬 运行完整profiling (这可能需要较长时间)..."
        python3 runner.py --mode full --config full --save-configs
        ;;
    3)
        echo "📏 自定义形状profiling"
        read -p "Batch size: " batch
        read -p "Number of heads: " heads
        read -p "Sequence length: " seq_len
        read -p "Head dimension: " head_dim
        
        python3 runner.py --mode shape --batch $batch --heads $heads --seq-len $seq_len --head-dim $head_dim --save-configs
        ;;
    4)
        echo "📊 运行NSys分析..."
        python3 runner.py --mode nsys
        ;;
    5)
        echo "🔍 运行NCU分析..."
        python3 runner.py --mode ncu
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 任务完成！"
    echo "📁 结果保存在 profile/results/ 目录中"
    echo "📊 你可以运行以下命令生成分析报告:"
    echo "   python3 analyzer.py --results-file profile/results/full_profiling_results.json --generate-report --generate-visualizations"
else
    echo "❌ 任务执行失败"
    exit 1
fi 