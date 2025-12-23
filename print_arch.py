import torch

def tuple2cand(cand_tuple):
    return torch.tensor(cand_tuple).reshape(-1, 6)

def parse_architecture(cand_tuple, name):
    print(f"=== {name} ===")
    # 将元组转换为PyTorch张量并重塑为每行6个元素的矩阵
    cand_tensor = tuple2cand(cand_tuple)
    
    print("架构格式: [stage_index, block_index, channels, kernel_size, groups1, groups2]")
    print("其中:")
    print("- stage_index: 阶段索引 (0-5)")
    print("- block_index: 块索引 (在该阶段中的块位置)")
    print("- channels: 输出通道数")
    print("- kernel_size: 卷积核大小")
    print("- groups1: 第一个卷积的组数")
    print("- groups2: 第二个1x1卷积的组数")
    print()
    
    for i, row in enumerate(cand_tensor):
        stage_idx, block_idx, channels, kernel_size, groups1, groups2 = row.int().tolist()
        
        # 跳过标记为-1的无效层
        if stage_idx == -1 or block_idx == -1:
            continue
            
        print(f"层 {i}: 阶段{stage_idx}, 块{block_idx}")
        print(f"  - 输出通道数: {channels}")
        print(f"  - 卷积核大小: {kernel_size}")
        print(f"  - groups1: {groups1}")
        print(f"  - groups2: {groups2}")
        print()
    
    print(f"总共 {len([row for row in cand_tensor if not (row[0] == -1 or row[1] == -1)])} 个有效层")
    print("\n" + "="*50 + "\n")

# 定义两个架构元组
arch1 = (0, 0, 32, 3, 1, 1, 1, 0, 48, 3, 1, 1, 1, 1, 96, 3, 1, 1, 1, 2, 96, 3, 1, 1, 2, 0, 128, 3, 1, 1, 2, 1, 128, 5, 2, 1, -1, -1, 128, 3, 1, 1, 3, 0, 192, 3, 2, 1, 3, 1, 256, 3, 2, 1, -1, -1, 384, 3, 2, 1, 4, 0, 384, 5, 4, 1, 4, 1, 384, 3, 8, 1, 4, 2, 384, 3, 4, 1, 4, 3, 768, 5, 8, 1, 4, 4, 768, 5, 8, 1, 4, 5, 768, 3, 8, 1, 4, 6, 768, 3, 4, 1, 4, 7, 768, 5, 8, 1, 4, 8, 768, 3, 4, 1, 5, 0, 768, 5, 16, 1, 5, 1, 1536, 3, 16, 1, -1, -1, 1536, 3, 8, 1)

arch2 = (0, 0, 24, 3, 1, 1, 1, 0, 48, 3, 1, 1, 1, 1, 48, 3, 1, 1, 1, 2, 64, 3, 1, 1, 2, 0, 96, 3, 2, 1, 2, 1, 192, 5, 2, 1, -1, -1, 192, 3, 1, 1, 3, 0, 256, 5, 4, 1, 3, 1, 384, 5, 4, 1, -1, -1, 384, 5, 4, 1, 4, 0, 384, 5, 8, 1, 4, 1, 384, 3, 8, 1, 4, 2, 384, 3, 8, 1, 4, 3, 512, 5, 4, 1, 4, 4, 768, 5, 8, 1, 4, 5, 768, 5, 4, 1, 4, 6, 768, 3, 4, 1, 4, 7, 768, 5, 4, 1, -1, -1, 768, 3, 4, 1, 5, 0, 1536, 3, 8, 1, 5, 1, 1536, 3, 16, 1, 5, 2, 1536, 3, 8, 1)

# 解析并打印两个架构
parse_architecture(arch1, "架构1")
parse_architecture(arch2, "架构2")