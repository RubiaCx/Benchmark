import torch
import triton
import triton.language as tl

print("🔍 开始测试...")
print(f"PyTorch版本: {torch.__version__}")
print(f"Triton版本: {triton.__version__}")
print(f"CUDA设备: {torch.cuda.get_device_name()}")

# 测试基本的triton功能
@triton.jit
def simple_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(output_ptr + offsets, output, mask=mask)

def test_basic_triton():
    print("✅ 测试基本Triton功能...")
    size = 1024
    x = torch.randn(size, device='cuda')
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    simple_kernel[grid](x, output, size, BLOCK_SIZE=256)
    
    expected = x * 2
    if torch.allclose(output, expected):
        print("✅ 基本Triton测试通过")
        return True
    else:
        print("❌ 基本Triton测试失败")
        return False

def test_small_attention():
    print("🧪 测试小规模attention...")
    try:
        # 导入attention函数
        from attn_triton import attention
        
        # 创建小规模测试数据
        batch, heads, seq_len, head_dim = 1, 2, 128, 64
        
        q = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device='cuda')
        k = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device='cuda')
        v = torch.randn((batch, heads, seq_len, head_dim), dtype=torch.float16, device='cuda')
        
        print(f"  - 输入形状: {q.shape}")
        print(f"  - 设备: {q.device}")
        print(f"  - 数据类型: {q.dtype}")
        
        # 执行attention
        result = attention(q, k, v, causal=True, sm_scale=0.125)
        print(f"✅ 小规模attention测试通过，输出形状: {result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 小规模attention测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    print("💾 检查GPU内存使用...")
    torch.cuda.empty_cache()
    print(f"  - 总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  - 已分配: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    print(f"  - 已缓存: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")

if __name__ == "__main__":
    try:
        test_memory_usage()
        
        if test_basic_triton():
            if test_small_attention():
                print("🎉 所有测试通过！")
            else:
                print("⚠️  attention测试失败，可能需要调整配置")
        else:
            print("⚠️  基本Triton功能测试失败")
            
    except Exception as e:
        print(f"💥 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_memory_usage() 