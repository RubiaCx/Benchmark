#!/bin/bash

# NVIDIA Nsight Compute 性能分析脚本
# 用法: ./ncu.sh [选项] <程序及其参数>

# 默认配置
DETAILED_MODE=true
IMPORT_SOURCE=true
OUTPUT_DIR="./ncu_reports"
OUTPUT_FILE=""
SECTIONS="SchedulerStats,WarpStateStats,SpeedOfLight_RooflineChart,SpeedOfLight_HierarchicalTensorRooflineChart"

# 显示帮助信息
show_help() {
    echo "NVIDIA Nsight Compute 性能分析脚本"
    echo ""
    echo "用法: $0 [选项] <程序及其参数>"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -o, --output FILE       指定输出文件名 (不含扩展名)"
    echo "  -d, --output-dir DIR    指定输出目录 (默认: ./ncu_reports)"
    echo "  -s, --sections LIST     指定要收集的分析节 (逗号分隔)"
    echo "  --simple                使用简单模式而非详细模式"
    echo "  --no-source             不导入源代码信息"
    echo ""
    echo "示例:"
    echo "  $0 python vector_add.py"
    echo "  $0 -o my_analysis ./my_cuda_program"
    echo "  $0 --simple python -c \"import torch; print(torch.cuda.is_available())\""
    echo "  $0 -s \"SchedulerStats,WarpStateStats\" ./my_program arg1 arg2"
    echo ""
    echo "常用分析节:"
    echo "  SchedulerStats                               - 调度器统计"
    echo "  WarpStateStats                              - 线程束状态统计"
    echo "  SpeedOfLight_RooflineChart                  - 性能屋顶线图"
    echo "  SpeedOfLight_HierarchicalTensorRooflineChart - 分层张量屋顶线图"
    echo "  MemoryWorkloadAnalysis                      - 内存工作负载分析"
    echo "  ComputeWorkloadAnalysis                     - 计算工作负载分析"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--sections)
            SECTIONS="$2"
            shift 2
            ;;
        --simple)
            DETAILED_MODE=false
            shift
            ;;
        --no-source)
            IMPORT_SOURCE=false
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "错误: 未知选项 $1"
            echo "使用 $0 --help 查看帮助信息"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# 检查是否提供了要分析的程序
if [[ $# -eq 0 ]]; then
    echo "错误: 请指定要分析的程序"
    echo "使用 $0 --help 查看帮助信息"
    exit 1
fi

# 检查ncu是否可用
if ! command -v ncu &> /dev/null; then
    echo "错误: 未找到 ncu 命令"
    echo "请确保已安装 NVIDIA Nsight Compute 并将其添加到 PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 生成输出文件名
if [[ -z "$OUTPUT_FILE" ]]; then
    PROGRAM_NAME=$(basename "$1")
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_FILE="${PROGRAM_NAME}_${TIMESTAMP}"
fi

OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

# 构建ncu命令
NCU_CMD="ncu --target-processes all"

if [[ "$DETAILED_MODE" == true ]]; then
    NCU_CMD="$NCU_CMD --set detailed"
fi

if [[ "$IMPORT_SOURCE" == true ]]; then
    NCU_CMD="$NCU_CMD --import-source yes"
fi

# 添加分析节
IFS=',' read -ra SECTION_ARRAY <<< "$SECTIONS"
for section in "${SECTION_ARRAY[@]}"; do
    NCU_CMD="$NCU_CMD --section $section"
done

NCU_CMD="$NCU_CMD -o $OUTPUT_PATH"

# 添加要分析的程序和参数
NCU_CMD="$NCU_CMD $*"

echo "正在运行性能分析..."
echo "命令: $NCU_CMD"
echo "输出文件: $OUTPUT_PATH.ncu-rep"
echo ""

# 执行分析
eval "$NCU_CMD"

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ 分析完成!"
    echo "报告保存在: $OUTPUT_PATH.ncu-rep"
    echo ""
    echo "查看报告的方法:"
    echo "1. 使用 GUI: ncu-ui $OUTPUT_PATH.ncu-rep"
    echo "2. 使用命令行: ncu --import $OUTPUT_PATH.ncu-rep --page details"
else
    echo ""
    echo "❌ 分析失败!"
    echo "请检查程序是否正确运行，或者是否有CUDA操作"
    exit 1
fi