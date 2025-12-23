import torch
import time
import math

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

class LayerTimer:
    """
    用于测量模型中各层执行时间的Hook类
    """
    def __init__(self):
        self.layer_times = {}
        self.layer_count = {}
        self.handles = []
        
    def register_hooks(self, model):
        """
        为模型的所有层注册前向Hook
        """
        for name, module in model.named_modules():
            if name:  # 忽略根模块
                handle = module.register_forward_hook(self.create_hook(name))
                self.handles.append(handle)
                
    def create_hook(self, layer_name):
        """
        为特定层创建Hook函数
        """
        def hook_fn(module, input, output):
            if layer_name not in self.layer_times:
                self.layer_times[layer_name] = []
                self.layer_count[layer_name] = 0
            
            # 记录当前时间
            start_time = time.time()
            
            # 保存开始时间到上下文中，以便在Hook结束时计算时间
            if not hasattr(self, 'start_times'):
                self.start_times = {}
            self.start_times[layer_name] = start_time
            
            # 为了在Hook结束后计算时间，我们需要在Hook函数结束后处理
            def record_time():
                if layer_name in self.start_times:
                    end_time = time.time()
                    layer_time = (end_time - self.start_times[layer_name]) * 1000  # 转换为毫秒
                    self.layer_times[layer_name].append(layer_time)
                    self.layer_count[layer_name] += 1
                    
            # 使用register_forward_hook时，我们不能直接在hook_fn中做最终记录
            # 所以我们记录开始时间，然后在外部处理
        return hook_fn

    def register_hooks_with_time_measurement(self, model):
        """
        为模型的所有层注册前向Hook来测量时间
        """
        self.layer_times = {}
        self.layer_count = {}
        self.start_times = {}
        self.handles = []
        
        def pre_hook_fn(module, input, module_name):
            self.start_times[module_name] = time.time()
        
        def post_hook_fn(module, input, output, module_name):
            if module_name in self.start_times:
                end_time = time.time()
                layer_time = (end_time - self.start_times[module_name]) * 1000  # 转换为毫秒
                if module_name not in self.layer_times:
                    self.layer_times[module_name] = []
                    self.layer_count[module_name] = 0
                self.layer_times[module_name].append(layer_time)
                self.layer_count[module_name] += 1
        
        for name, module in model.named_modules():
            if name:  # 忽略根模块
                # 注册pre-hook
                pre_handle = module.register_forward_pre_hook(
                    lambda module, input, name=name: pre_hook_fn(module, input, name)
                )
                # 注册post-hook
                post_handle = module.register_forward_hook(
                    lambda module, input, output, name=name: post_hook_fn(module, input, output, name)
                )
                self.handles.extend([pre_handle, post_handle])

    def remove_hooks(self):
        """
        移除所有注册的Hook
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
    def get_average_times(self):
        """
        获取各层的平均执行时间
        """
        avg_times = {}
        for layer_name, times in self.layer_times.items():
            if times:
                avg_times[layer_name] = sum(times) / len(times)
            else:
                avg_times[layer_name] = 0.0
        return avg_times

def calculate_ops_per_layer(model, cand_tuple):
    """
    计算架构中每层的FLOPs和bit-ops
    """
    print("=== 计算每层的FLOPs和bit-ops ===")
    
    # 将架构元组转换为张量
    arch_tensor = tuple2cand(cand_tuple)
    
    # 计算每层的ops
    flops_per_layer = []
    bitops_per_layer = []
    layer_names = []
    
    pre = None
    cur = None
    
    for idx, (i, j, channels, ks, groups1, groups2) in enumerate(arch_tensor):
        if i == -1 or j == -1:
            continue
            
        cur = [channels.item(), ks.item(), groups1.item(), groups2.item()]
        
        # 获取对应层的模块
        layer_module = model.features[i][j]
        
        if i == 0:  # 第一阶段，使用StemBlock
            tmp_flops, tmp_bitops = layer_module.get_flops_bitops(cur)
            
            # StemBlock只有一个conv层
            layer_name = f"features.{i}.{j}.conv"
            flops_per_layer.append(tmp_flops)
            bitops_per_layer.append(tmp_bitops)
            layer_names.append(layer_name)
            print(f"features.{i}.{j}.conv: FLOPs={tmp_flops:.6f}M, bit-ops={tmp_bitops:.6f}M")
        else:  # 非第一阶段，使用BasicBlock
            # BasicBlock包含两个卷积层：binary_conv和binary_conv1x1
            # 我们需要将总的FLOPs和bit-ops分解为这两个部分
            tmp_flops, tmp_bitops = layer_module.get_flops_bitops(pre, cur)
            
            # 从BasicBlock的get_flops_bitops源码可以看到，bitops由两部分组成：
            # bitops1 = binary_conv的bitops
            # bitops2 = binary_conv1x1的bitops
            # 而flops目前为0
            
            # 重新计算每个组件的bitops
            pre_channels = pre[0] if pre is not None else cur[0]
            wh = layer_module.binary_conv.wh
            stride = layer_module.stride
            
            # binary_conv部分的bitops (带kernel size的卷积)
            bitops1 = (ks * ks * pre_channels // groups1 * cur[0] *
                      wh // stride * wh // stride) / 1e6
            flops1 = 0.0
            
            # binary_conv1x1部分的bitops (1x1卷积)
            # 输出通道数是当前层的channels，即cur[0]，但根据架构可能有所不同
            # 实际上应该是从当前层到下一层的通道数变换，这里应该是cur[2]，但实际上是当前层输出通道
            # 根据模型结构，binary_conv1x1的输出通道是max_oup，即cur[0]
            wh1x1 = layer_module.binary_conv1x1.wh
            bitops2 = (1 * 1 * cur[0] // groups2 * cur[0] * wh1x1 * wh1x1) / 1e6
            flops2 = 0.0
            
            # 输出binary_conv部分
            print(f"features.{i}.{j}.binary_conv: FLOPs={flops1:.6f}M, bit-ops={bitops1:.6f}M")
            
            # 输出binary_conv1x1部分
            print(f"features.{i}.{j}.binary_conv1x1: FLOPs={flops2:.6f}M, bit-ops={bitops2:.6f}M")
            
            # 添加到结果列表
            flops_per_layer.extend([flops1, flops2])
            bitops_per_layer.extend([bitops1, bitops2])
            layer_names.extend([f"features.{i}.{j}.binary_conv", f"features.{i}.{j}.binary_conv1x1"])
        
        pre = cur
    
    # 计算最后的全连接层
    if pre is not None:
        fc_flops, fc_bitops = model.fc.get_flops_bitops(pre)
        flops_per_layer.append(fc_flops)
        bitops_per_layer.append(fc_bitops)
        layer_names.append("fc")
        print(f"fc: FLOPs={fc_flops:.6f}M, bit-ops={fc_bitops:.6f}M")
    
    total_flops = sum(flops_per_layer)
    total_bitops = sum(bitops_per_layer)
    total_ops = total_flops + total_bitops / 64
    
    print(f"\n总FLOPs: {total_flops:.6f}M")
    print(f"总bit-ops: {total_bitops:.6f}M")
    print(f"总OPS: {total_ops:.6f}M")
    print("\n" + "="*50 + "\n")
    
    return {
        'layer_names': layer_names,
        'flops': flops_per_layer,
        'bitops': bitops_per_layer,
        'total_flops': total_flops,
        'total_bitops': total_bitops,
        'total_ops': total_ops
    }

def measure_arch_with_layer_timing(model, cand_tuple, device='cpu', num_runs=40, warmup_runs=20):
    """
    使用Hook测量架构中各层的执行时间
    """
    print(f"=== 使用Hook测量架构执行时间 ===")
    print(f"设备: {device}, 运行次数: {num_runs}, 预热次数: {warmup_runs}")
    print()
    
    model.eval()
    model = model.to(device)
    
    # 创建LayerTimer实例
    timer = LayerTimer()
    timer.register_hooks_with_time_measurement(model)
    
    # 准备输入张量 - 这是整个网络的初始输入
    dummy_input = torch.randn((1, 3, 224, 224)).to(device)
    arch_tensor = tuple2cand(cand_tuple)
    
    # 预热运行
    print("开始预热...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(dummy_input, arch_tensor)
    
    # 同步GPU（如果使用GPU）
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("开始正式测量...")
    # 执行多次运行以获得更准确的时间测量
    for run_idx in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input, arch_tensor)
        
        # 同步GPU（如果使用GPU）
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # 获取平均时间
    avg_times = timer.get_average_times()
    
    # 移除Hook
    timer.remove_hooks()
    
    # 打印各层时间
    print("各层执行时间:")
    total_time = 0.0
    layer_count = 0
    total_time_linear = 0.0
    layer_count_linear = 0
   
    for layer_name, avg_time in avg_times.items():
        total_time += avg_time
        layer_count += 1
        if avg_time > 0:  # 只显示有测量值的层
            if 'conv' in layer_name.lower() or 'linear' in layer_name.lower() or 'fc' in layer_name.lower():
                print(f"{layer_name}: {avg_time:.4f} ms")
                layer_count_linear += 1
                total_time_linear += avg_time

    
    print()
    print(f"总执行时间 (所有层平均): {total_time:.4f} ms")
    print(f"总执行时间 (所有线性层): {total_time_linear:.4f} ms")
    if layer_count > 0:
        print(f"平均每层执行时间: {total_time/layer_count:.4f} ms (共{layer_count}个被调用的层)")
    if layer_count_linear > 0:
        print(f"平均每个线性层执行时间: {total_time/layer_count_linear:.4f} ms (共{layer_count_linear}个被调用的线性层)")

    print("\n" + "="*50 + "\n")
    
    return avg_times, total_time

def create_model_and_measure(arch_tuple, name, device='cpu', save_results=True):
    """
    创建模型实例并测量架构执行时间，同时计算ops
    """
    print(f"正在为 {name} 创建模型并测量执行时间与ops...")
    
    # 导入模型
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
    
    from models.superbnn import superbnn
    model = superbnn()
    
    # 计算每层的ops
    ops_data = calculate_ops_per_layer(model, arch_tuple)
    
    # 测量架构执行时间
    time_data, total_time = measure_arch_with_layer_timing(model, arch_tuple, device)
    
    # 创建包含ops和时间数据的字典
    combined_data = {
        'architecture_name': name,
        'ops_data': ops_data,
        'time_data': time_data,
        'total_execution_time': total_time
    }
    
    # 保存结果为pth文件
    if save_results:
        filename = f"{name.replace(' ', '_')}_results.pth"
        torch.save(combined_data, filename)
        print(f"结果已保存到 {filename}")
    
    return combined_data

# 定义两个架构元组
arch1 = (0, 0, 24, 3, 1, 1, 1, 0, 48, 3, 1, 1, 1, 1, 48, 3, 1, 1, 1, 2, 64, 3, 1, 1, 2, 0, 96, 3, 2, 1, 2, 1, 192, 5, 2, 1, -1, -1, 192, 3, 1, 1, 3, 0, 256, 5, 4, 1, 3, 1, 384, 5, 4, 1, -1, -1, 384, 5, 4, 1, 4, 0, 384, 5, 8, 1, 4, 1, 384, 3, 8, 1, 4, 2, 384, 3, 8, 1, 4, 3, 512, 5, 4, 1, 4, 4, 768, 5, 8, 1, 4, 5, 768, 5, 4, 1, 4, 6, 768, 3, 4, 1, 4, 7, 768, 5, 4, 1, -1, -1, 768, 3, 4, 1, 5, 0, 1536, 3, 8, 1, 5, 1, 1536, 3, 16, 1, 5, 2, 1536, 3, 8, 1)

arch2 = (0, 0, 24, 3, 1, 1, 1, 0, 96, 3, 1, 1, 1, 1, 96, 3, 1, 1, -1, -1, 96, 3, 1, 1, 2, 0, 96, 5, 2, 1, 2, 1, 128, 3, 1, 1, -1, -1, 192, 3, 1, 1, 3, 0, 192, 3, 4, 1, 3, 1, 256, 3, 4, 1, 3, 2, 384, 5, 4, 1, 4, 0, 512, 3, 8, 1, 4, 1, 512, 3, 4, 1, 4, 2, 768, 3, 8, 1, 4, 3, 768, 3, 4, 1, 4, 4, 768, 3, 8, 1, 4, 5, 768, 3, 8, 1, 4, 6, 768, 5, 8, 1, 4, 7, 768, 5, 4, 1, 4, 8, 768, 3, 8, 1, 5, 0, 1024, 5, 16, 1, 5, 1, 1536, 5, 16, 1, -1, -1, 1536, 3, 16, 1)

# 解析并打印两个架构
parse_architecture(arch1, "架构1")
parse_architecture(arch2, "架构2")

# 创建模型并测量，同时保存结果
result1 = create_model_and_measure(arch1, "架构1", device='cpu', save_results=True)  # 可以改为 'cuda' 如果有GPU
result2 = create_model_and_measure(arch2, "架构2", device='cpu', save_results=True)  # 可以改为 'cuda' 如果有GPU