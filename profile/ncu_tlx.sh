configs=("128" "8192")
# providers=("triton-fp16")
providers=("triton-ws-fp16" "triton-ws-pp-fp16" "triton-fp16")

for n_ctx in "${configs[@]}"; do
    for provider in "${providers[@]}"; do
        echo "分析配置: N_CTX=$n_ctx, provider=$provider"
        
        output_file="ncu_results/${provider}_${n_ctx}.ncu-rep"
        
        # ncu --set full -o "$output_file" --force-overwrite python profile_tlx.py --n_ctx $n_ctx --provider $provider
        
        # # 生成报告
        # echo "生成报告: $output_file"
        ncu -i "$output_file" --page details  --print-summary per-kernel > "ncu_results/report_${provider}_${n_ctx}.txt"
        
        echo "完成: $provider N_CTX=$n_ctx"
    done
done
