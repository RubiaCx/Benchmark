#!/bin/bash
# Flash Attention WS Hopper 分析工具快速启动脚本

echo "=================================="
echo "Flash Attention WS Hopper 分析工具"
echo "=================================="
echo ""
echo "请选择要执行的操作:"
echo "1. 创建分析脚本"
echo "2. 运行基准测试"
echo "3. 详细对比分析"
echo "4. NVTX分析"
echo "5. NCU分析"
echo "6. 单个配置分析"
echo "7. 查看帮助"
echo "0. 退出"
echo ""

read -p "请输入选项 (0-7): " choice

case $choice in
    1)
        echo "正在创建分析脚本..."
        python flash-attention-WS-hopper.py --mode create_scripts
        echo "完成！可以直接使用生成的脚本进行分析。"
        ;;
    2)
        echo "正在运行基准测试..."
        python flash-attention-WS-hopper.py --mode benchmark
        ;;
    3)
        echo "正在进行详细对比分析..."
        python flash-attention-WS-hopper.py --mode analysis
        ;;
    4)
        echo "正在进行NVTX分析..."
        echo "请确保已安装nvtx: pip install nvtx"
        if command -v nsys &> /dev/null; then
            nsys profile -o nvtx_profile --trace=cuda,nvtx --force-overwrite=true \
                python flash-attention-WS-hopper.py --mode nvtx
            echo "NVTX分析完成！"
            echo "可以使用以下命令查看结果:"
            echo "  nsys-ui nvtx_profile.qdrep"
            echo "  nsys stats nvtx_profile.nsys-rep"
        else
            echo "错误: nsys未安装或不在PATH中"
            echo "请安装NVIDIA Nsight Systems"
        fi
        ;;
    5)
        echo "正在进行NCU分析..."
        if command -v ncu &> /dev/null; then
            # 创建结果目录
            mkdir -p ncu_results
            
            # 分析所有配置
            configs=("128" "8192")
            providers=("triton-ws-fp16" "triton-ws-pp-fp16")
            
            for n_ctx in "${configs[@]}"; do
                for provider in "${providers[@]}"; do
                    echo "分析配置: N_CTX=$n_ctx, provider=$provider"
                    
                    output_file="ncu_results/ncu_${provider}_${n_ctx}.ncu-rep"
                    
                    # 运行NCU分析
                    ncu --set full -o "$output_file" --force-overwrite \
                        python flash-attention-WS-hopper.py --mode ncu --n_ctx $n_ctx --provider $provider
                    
                    # 生成报告
                    echo "生成报告: $output_file"
                    ncu -i "$output_file" --page raw > "ncu_results/report_${provider}_${n_ctx}.txt"
                    
                    echo "完成: $provider N_CTX=$n_ctx"
                    echo "------------------------"
                done
            done
            
            echo "NCU分析完成！结果保存在 ncu_results/ 目录中"
        else
            echo "错误: ncu未安装或不在PATH中"
            echo "请安装NVIDIA Nsight Compute"
        fi
        ;;
    6)
        echo "单个配置分析"
        echo "请选择N_CTX值:"
        echo "1. 128"
        echo "2. 8192"
        read -p "请输入选项 (1-2): " n_ctx_choice
        
        case $n_ctx_choice in
            1) n_ctx="128" ;;
            2) n_ctx="8192" ;;
            *) echo "无效选项"; exit 1 ;;
        esac
        
        echo "请选择提供者:"
        echo "1. triton-ws-fp16 (基础版本)"
        echo "2. triton-ws-pp-fp16 (Pipeline版本)"
        read -p "请输入选项 (1-2): " provider_choice
        
        case $provider_choice in
            1) provider="triton-ws-fp16" ;;
            2) provider="triton-ws-pp-fp16" ;;
            *) echo "无效选项"; exit 1 ;;
        esac
        
        echo "正在分析配置: N_CTX=$n_ctx, provider=$provider"
        python flash-attention-WS-hopper.py --mode ncu --n_ctx $n_ctx --provider $provider
        ;;
    7)
        echo "Flash Attention WS Hopper 分析工具帮助"
        echo "========================================"
        echo ""
        echo "这个工具用于分析四种不同的Flash Attention配置:"
        echo "1. Triton WS FP16 (N_CTX=128) - 基础版本，短序列"
        echo "2. Triton WS FP16 (N_CTX=8192) - 基础版本，长序列"
        echo "3. Triton WS PP FP16 (N_CTX=128) - Pipeline版本，短序列"
        echo "4. Triton WS PP FP16 (N_CTX=8192) - Pipeline版本，长序列"
        echo ""
        echo "分析类型:"
        echo "- 基准测试: 比较不同配置的性能"
        echo "- NVTX分析: 使用NVIDIA Nsight Systems进行时间线分析"
        echo "- NCU分析: 使用NVIDIA Nsight Compute进行深度kernel分析"
        echo "- 详细对比: 生成comprehensive性能对比报告"
        echo ""
        echo "输出文件:"
        echo "- benchmark_results.txt: 基准测试结果"
        echo "- analysis_results.json: 详细分析结果"
        echo "- nvtx_profile.qdrep: NVTX分析结果"
        echo "- ncu_results/: NCU分析结果目录"
        echo ""
        echo "更多信息请查看 README_analysis.md"
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项，请输入0-7之间的数字"
        exit 1
        ;;
esac

echo ""
echo "操作完成！"
echo "如需再次运行，请执行: ./quick_start.sh" 