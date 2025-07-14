#!/bin/bash

# Flash Attention Profiling 全自动测试脚本

echo "🔬 Flash Attention Profiling 全自动测试"
echo "========================================="

# 切换到profile目录
cd "$(dirname "$0")/.."

# 创建时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="results/logs/full_test_${TIMESTAMP}.log"

# 重定向输出到日志文件
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "📝 日志文件: $LOG_FILE"
echo "🕐 开始时间: $(date)"

# 1. 系统检查
echo ""
echo "=== 第1步: 系统检查 ==="
python3 runner.py --mode check

if [ $? -ne 0 ]; then
    echo "❌ 系统检查失败，退出测试"
    exit 1
fi

# 2. 快速测试
echo ""
echo "=== 第2步: 快速测试 ==="
python3 runner.py --mode quick --config quick --save-configs

if [ $? -ne 0 ]; then
    echo "❌ 快速测试失败，退出测试"
    exit 1
fi

# 3. 完整profiling
echo ""
echo "=== 第3步: 完整profiling ==="
python3 runner.py --mode full --config full --save-configs

if [ $? -ne 0 ]; then
    echo "❌ 完整profiling失败，但继续执行后续步骤"
fi

# 4. 生成分析报告
echo ""
echo "=== 第4步: 生成分析报告 ==="
if [ -f "results/full_profiling_results.json" ]; then
    python3 analyzer.py --results-file results/full_profiling_results.json \
                        --output-dir analysis_output \
                        --generate-report \
                        --generate-visualizations \
                        --compare-configs \
                        --analyze-scaling
    
    if [ $? -eq 0 ]; then
        echo "✅ 分析报告生成成功"
    else
        echo "❌ 分析报告生成失败"
    fi
else
    echo "⚠️  未找到profiling结果文件，跳过分析报告生成"
fi

# 5. 针对几个关键形状进行NSys分析
echo ""
echo "=== 第5步: NSys分析 ==="

# 定义要分析的形状
declare -a shapes=(
    "1 8 1024 128"
    "4 16 2048 128"
    "8 32 4096 128"
)

for shape in "${shapes[@]}"; do
    read -r batch heads seq_len head_dim <<< "$shape"
    echo "📊 运行NSys分析: B${batch}_H${heads}_S${seq_len}_D${head_dim}"
    
    python3 runner.py --mode shape --batch $batch --heads $heads --seq-len $seq_len --head-dim $head_dim --save-configs
    
    if [ $? -eq 0 ]; then
        # 查找生成的配置文件
        config_file="results/configs/b${batch}_h${heads}_s${seq_len}_d${head_dim}_best_config.json"
        if [ -f "$config_file" ]; then
            python3 runner.py --mode nsys --config-file "$config_file"
        else
            echo "⚠️  未找到配置文件，使用默认配置运行NSys"
            python3 runner.py --mode nsys
        fi
    fi
done

# 6. NCU分析 (仅对一个形状进行，因为NCU分析较为耗时)
echo ""
echo "=== 第6步: NCU分析 ==="
echo "🔍 运行NCU分析 (仅对一个代表性形状)"

config_file="results/configs/b4_h16_s2048_d128_best_config.json"
if [ -f "$config_file" ]; then
    python3 runner.py --mode ncu --config-file "$config_file"
else
    echo "⚠️  未找到配置文件，使用默认配置运行NCU"
    python3 runner.py --mode ncu
fi

# 7. 生成最终报告
echo ""
echo "=== 第7步: 生成最终报告 ==="

FINAL_REPORT="results/final_report_${TIMESTAMP}.txt"

cat > "$FINAL_REPORT" << EOF
Flash Attention Profiling 测试报告
==================================

测试时间: $(date)
测试环境: $(python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')")

测试结果文件:
- 完整profiling结果: results/full_profiling_results.json
- 分析报告: analysis_output/comprehensive_report.txt
- 性能报告: results/performance_report.txt
- CSV导出: results/results.csv

可视化图表:
- 性能热图: analysis_output/performance_heatmap.png
- 扩展性分析: analysis_output/scaling_analysis.png
- 配置分布: analysis_output/config_distribution.png
- 性能对比: analysis_output/performance_comparison.png

Profiling文件:
- NSys结果: results/nsys/
- NCU结果: results/ncu/

日志文件:
- 详细日志: $LOG_FILE

建议后续步骤:
1. 查看 analysis_output/comprehensive_report.txt 了解性能分析结果
2. 使用 NSight Systems 打开 results/nsys/ 中的文件进行系统级分析
3. 使用 NSight Compute 打开 results/ncu/ 中的文件进行内核级分析
4. 根据分析结果优化Flash Attention配置

EOF

echo "📊 最终报告已生成: $FINAL_REPORT"

# 8. 结果总结
echo ""
echo "=== 测试完成总结 ==="
echo "🕐 结束时间: $(date)"

# 统计文件
echo ""
echo "📁 生成的文件:"
find results/ -name "*.json" -o -name "*.txt" -o -name "*.csv" | head -10

echo ""
echo "🎉 全自动测试完成！"
echo "📊 请查看以下文件了解详细结果:"
echo "   - 最终报告: $FINAL_REPORT"
echo "   - 详细日志: $LOG_FILE"
echo "   - 分析报告: analysis_output/comprehensive_report.txt" 