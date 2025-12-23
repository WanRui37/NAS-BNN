import re
import torch
import matplotlib.pyplot as plt
import numpy as np

# ===================== 0. 全局绘图风格设置 =====================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,  # 增大默认字体
        "axes.linewidth": 1.5,  # 坐标轴线宽
        "xtick.major.width": 1.2,  # 刻度线宽
        "ytick.major.width": 1.2,
        "xtick.direction": "in",  # 刻度向内
        "ytick.direction": "in",
        "mathtext.fontset": "stix",  # 公式字体风格
    }
)

def parse_log_file(log_path):
    """
    解析日志文件，提取架构信息和执行时间
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割两个架构的数据
    sections = content.split('=' * 50)
    
    arch_data = {}
    
    # 解析第一个架构
    arch1_section = sections[1] if len(sections) > 1 else ""
    if "架构1" in arch1_section:
        arch1_data = parse_architecture_section(arch1_section)
        arch_data["架构1"] = arch1_data
    
    # 解析第二个架构
    arch2_section = sections[2] if len(sections) > 2 else ""
    if "架构2" in arch2_section:
        arch2_data = parse_architecture_section(arch2_section)
        arch_data["架构2"] = arch2_data
    
    return arch_data

def parse_architecture_section(section):
    """
    解析单个架构部分的数据
    """
    # 提取层执行时间
    time_pattern = r'(.+?): ([\d.]+) ms'
    time_matches = re.findall(time_pattern, section)
    
    layer_times = {}
    binary_conv_times = []
    binary_conv1x1_times = []
    
    for layer_name, time_str in time_matches:
        time_val = float(time_str)
        layer_times[layer_name] = time_val
        
        if 'binary_conv' in layer_name and 'binary_conv1x1' not in layer_name:
            binary_conv_times.append(time_val)
        elif 'binary_conv1x1' in layer_name:
            binary_conv1x1_times.append(time_val)
    
    # 计算总时间
    total_time = sum(layer_times.values())
    binary_conv_total_time = sum(binary_conv_times)
    binary_conv1x1_total_time = sum(binary_conv1x1_times)
    
    # 假设ops占比与时间占比相同（在没有实际ops数据的情况下）
    binary_conv_ops_ratio = binary_conv_total_time / total_time if total_time > 0 else 0
    binary_conv1x1_ops_ratio = binary_conv1x1_total_time / total_time if total_time > 0 else 0
    other_ops_ratio = 1 - binary_conv_ops_ratio - binary_conv1x1_ops_ratio
    
    # 计算时间占比
    binary_conv_time_ratio = binary_conv_total_time / total_time if total_time > 0 else 0
    binary_conv1x1_time_ratio = binary_conv1x1_total_time / total_time if total_time > 0 else 0
    other_time_ratio = 1 - binary_conv_time_ratio - binary_conv1x1_time_ratio
    
    return {
        'binary_conv_ops_ratio': binary_conv_ops_ratio,
        'binary_conv1x1_ops_ratio': binary_conv1x1_ops_ratio,
        'other_ops_ratio': other_ops_ratio,
        'binary_conv_time_ratio': binary_conv_time_ratio,
        'binary_conv1x1_time_ratio': binary_conv1x1_time_ratio,
        'other_time_ratio': other_time_ratio,
        'binary_conv_total_time': binary_conv_total_time,
        'binary_conv1x1_total_time': binary_conv1x1_total_time,
        'total_time': total_time
    }

def load_arch_data_from_pth():
    """
    从pth文件加载架构数据
    """
    arch_data = {}
    
    # 加载架构1的数据
    try:
        arch1_data = torch.load('架构1_results.pth')
        print(f"架构1加载成功，数据类型: {type(arch1_data)}")
        if isinstance(arch1_data, dict):
            print(f"架构1键: {list(arch1_data.keys())}")
            if 'ops_data' in arch1_data and isinstance(arch1_data['ops_data'], dict):
                print(f"架构1 ops_data keys: {list(arch1_data['ops_data'].keys())}")
                print(f"架构1 layer_names数量: {len(arch1_data['ops_data'].get('layer_names', []))}")
            if 'time_data' in arch1_data:
                print(f"架构1 time_data keys数量: {len(arch1_data['time_data'])}")
        arch_data["架构1"] = arch1_data
        print("成功加载架构1的数据")
    except FileNotFoundError:
        print("警告：未找到架构1的pth文件")
    except Exception as e:
        print(f"加载架构1时出错: {e}")
    
    # 加载架构2的数据
    try:
        arch2_data = torch.load('架构2_results.pth')
        print(f"架构2加载成功，数据类型: {type(arch2_data)}")
        if isinstance(arch2_data, dict):
            print(f"架构2键: {list(arch2_data.keys())}")
            if 'ops_data' in arch2_data and isinstance(arch2_data['ops_data'], dict):
                print(f"架构2 ops_data keys: {list(arch2_data['ops_data'].keys())}")
                print(f"架构2 layer_names数量: {len(arch2_data['ops_data'].get('layer_names', []))}")
            if 'time_data' in arch2_data:
                print(f"架构2 time_data keys数量: {len(arch2_data['time_data'])}")
        arch_data["架构2"] = arch2_data
        print("成功加载架构2的数据")
    except FileNotFoundError:
        print("警告：未找到架构2的pth文件")
    except Exception as e:
        print(f"加载架构2时出错: {e}")
    
    return arch_data

def analyze_ops_and_time_data(arch_data):
    """
    分析ops_data和time_data
    """
    analyzed_data = {}
    
    for arch_name, data in arch_data.items():
        print(f"\nAnalyzing {arch_name} ops_data and time_data...")
        
        if 'ops_data' not in data:
            print(f"Error: {arch_name} does not contain ops_data key")
            continue
            
        if 'time_data' not in data:
            print(f"Error: {arch_name} does not contain time_data key")
            continue
            
        ops_data = data['ops_data']
        time_data = data['time_data']
        
        # 检查必要的键是否存在
        if 'layer_names' not in ops_data or 'flops' not in ops_data or 'bitops' not in ops_data:
            print(f"Error: {arch_name} ops_data missing required keys (layer_names, flops, bitops)")
            continue
        
        # 检查数组长度是否一致
        layer_names = ops_data['layer_names']
        flops = ops_data['flops']
        bitops = ops_data['bitops']
        
        if len(layer_names) != len(flops) or len(layer_names) != len(bitops):
            print(f"Error: {arch_name} layer_names, flops, bitops lengths are inconsistent: {len(layer_names)}, {len(flops)}, {len(bitops)}")
            continue
        
        print(f"{arch_name} - ops_data info:")
        print(f"  Number of layer_names: {len(layer_names)}")
        print(f"  Total flops: {sum(flops)}")
        print(f"  Total bitops: {sum(bitops)}")
        print(f"  Number of time_data keys: {len(time_data)}")
        print(f"  Total time_data: {sum(time_data.values()) if time_data else 0}")
        
        # 分析ops类型 - 根据layer_names中的层类型来分类
        binary_conv_flops = 0
        binary_conv_bitops = 0
        binary_conv_time = 0
        binary_conv1x1_flops = 0
        binary_conv1x1_bitops = 0
        binary_conv1x1_time = 0
        other_flops = 0
        other_bitops = 0
        other_time = 0
        
        # 分析每层的ops和时间
        for idx, layer_name in enumerate(layer_names):
            layer_flops = flops[idx]
            layer_bitops = bitops[idx]
            
            # 根据layer_name判断层类型
            if 'binary_conv' in layer_name and 'binary_conv1x1' not in layer_name:
                # binary_conv层
                binary_conv_flops += layer_flops
                binary_conv_bitops += layer_bitops
                # 尝试在time_data中找到对应的层
                for time_layer_name, time_val in time_data.items():
                    if layer_name in time_layer_name or time_layer_name in layer_name:
                        binary_conv_time += time_val
                        break
            elif 'binary_conv1x1' in layer_name:
                # binary_conv1x1层
                binary_conv1x1_flops += layer_flops
                binary_conv1x1_bitops += layer_bitops
                # 尝试在time_data中找到对应的层
                for time_layer_name, time_val in time_data.items():
                    if layer_name in time_layer_name or time_layer_name in layer_name:
                        binary_conv1x1_time += time_val
                        break
            else:
                # 其他层 (如conv, fc等)
                other_flops += layer_flops
                other_bitops += layer_bitops
                # 尝试在time_data中找到对应的层
                for time_layer_name, time_val in time_data.items():
                    if layer_name in time_layer_name or time_layer_name in layer_name:
                        other_time += time_val
                        break
        
        # 计算总值
        total_flops = sum(flops)
        total_bitops = sum(bitops)
        total_time = sum(time_data.values()) if time_data else 0
        
        # 计算比例
        total_ops = total_flops + total_bitops / 64  # 使用FLOPs + bit-ops/64作为总ops
        binary_conv_ops = binary_conv_flops + binary_conv_bitops / 64
        binary_conv1x1_ops = binary_conv1x1_flops + binary_conv1x1_bitops / 64
        other_ops = other_flops + other_bitops / 64
        
        # ops比例
        binary_conv_ops_ratio = binary_conv_ops / total_ops if total_ops > 0 else 0
        binary_conv1x1_ops_ratio = binary_conv1x1_ops / total_ops if total_ops > 0 else 0
        other_ops_ratio = other_ops / total_ops if total_ops > 0 else 0
        
        # 时间比例
        binary_conv_time_ratio = binary_conv_time / total_time if total_time > 0 else 0
        binary_conv1x1_time_ratio = binary_conv1x1_time / total_time if total_time > 0 else 0
        other_time_ratio = other_time / total_time if total_time > 0 else 0
        
        analyzed_data[arch_name] = {
            'ops_data': {
                'total_flops': total_flops,
                'total_bitops': total_bitops,
                'binary_conv_flops': binary_conv_flops,
                'binary_conv_bitops': binary_conv_bitops,
                'binary_conv1x1_flops': binary_conv1x1_flops,
                'binary_conv1x1_bitops': binary_conv1x1_bitops,
                'other_flops': other_flops,
                'other_bitops': other_bitops,
                'total_ops': total_ops,
                'binary_conv_ops': binary_conv_ops,
                'binary_conv1x1_ops': binary_conv1x1_ops,
                'other_ops': other_ops,
                'binary_conv_ops_ratio': binary_conv_ops_ratio,
                'binary_conv1x1_ops_ratio': binary_conv1x1_ops_ratio,
                'other_ops_ratio': other_ops_ratio
            },
            'time_data': {
                'total_time': total_time,
                'binary_conv_time': binary_conv_time,
                'binary_conv1x1_time': binary_conv1x1_time,
                'other_time': other_time,
                'binary_conv_time_ratio': binary_conv_time_ratio,
                'binary_conv1x1_time_ratio': binary_conv1x1_time_ratio,
                'other_time_ratio': other_time_ratio
            }
        }
        
        print(f"{arch_name} Analysis Results:")
        print(f"  Total FLOPs: {total_flops}, Total bit-ops: {total_bitops}")
        print(f"  Binary Conv FLOPs: {binary_conv_flops}, bit-ops: {binary_conv_bitops}, Time: {binary_conv_time:.2f}")
        print(f"  Binary Conv 1x1 FLOPs: {binary_conv1x1_flops}, bit-ops: {binary_conv1x1_bitops}, Time: {binary_conv1x1_time:.2f}")
        print(f"  Other FLOPs: {other_flops}, bit-ops: {other_bitops}, Time: {other_time:.2f}")
        print(f"  Binary Conv Ops Ratio: {binary_conv_ops_ratio:.2%}, Time Ratio: {binary_conv_time_ratio:.2%}")
        print(f"  Binary Conv 1x1 Ops Ratio: {binary_conv1x1_ops_ratio:.2%}, Time Ratio: {binary_conv1x1_time_ratio:.2%}")
        print(f"  Other Ops Ratio: {other_ops_ratio:.2%}, Time Ratio: {other_time_ratio:.2%}")
    
    return analyzed_data

def create_pie_charts(analyzed_data):
    """
    创建ops和时间占比的饼图
    """
    # 使用全局字体设置 (Times New Roman)
    plt.rcParams['axes.unicode_minus'] = False
    
    num_archs = len(analyzed_data)
    if num_archs == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, num_archs, figsize=(6*num_archs, 12))
    if num_archs == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Operations and Time Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    labels = ['Binary Conv', 'Binary Conv 1x1', 'Others']
    
    for i, (arch_name, data) in enumerate(analyzed_data.items()):
        ops_data = data['ops_data']
        time_data = data['time_data']
        
        # Ops占比饼图
        ops_ratios = [
            ops_data['binary_conv_ops_ratio'],
            ops_data['binary_conv1x1_ops_ratio'],
            ops_data['other_ops_ratio']
        ]
        ops_ratios = [max(0, ratio) for ratio in ops_ratios]  # 确保非负
        
        axes[0, i].pie(ops_ratios, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, i].set_title(f'{arch_name} - Ops Ratio', fontweight='bold')
        
        # Time占比饼图
        time_ratios = [
            time_data['binary_conv_time_ratio'],
            time_data['binary_conv1x1_time_ratio'],
            time_data['other_time_ratio']
        ]
        time_ratios = [max(0, ratio) for ratio in time_ratios]  # 确保非负
        
        axes[1, i].pie(time_ratios, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, i].set_title(f'{arch_name} - Time Ratio', fontweight='bold')
    
    plt.tight_layout()
    # 保存为PDF
    plt.savefig('ops_time_analysis_pie_charts.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print("Pie charts saved to ops_time_analysis_pie_charts.pdf")

def create_comparison_bars(analyzed_data):
    """
    创建ops和时间对比柱状图
    """
    # 使用全局字体设置 (Times New Roman)
    plt.rcParams['axes.unicode_minus'] = False
    
    arch_names = list(analyzed_data.keys())
    if len(arch_names) < 2:
        print("At least two architectures are needed to create comparison chart")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Architecture Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    labels = ['Binary Conv', 'Binary Conv 1x1', 'Others']
    
    x = np.arange(len(arch_names))  # 架构数量
    width = 0.25  # 柱宽
    
    # 准备ops数据
    binary_conv_ops = [analyzed_data[arch]['ops_data']['binary_conv_ops_ratio'] for arch in arch_names]
    binary_conv1x1_ops = [analyzed_data[arch]['ops_data']['binary_conv1x1_ops_ratio'] for arch in arch_names]
    other_ops = [analyzed_data[arch]['ops_data']['other_ops_ratio'] for arch in arch_names]
    
    # 绘制ops对比图
    axes[0].bar(x - width, binary_conv_ops, width, label='Binary Conv', color=colors[0])
    axes[0].bar(x, binary_conv1x1_ops, width, label='Binary Conv 1x1', color=colors[1])
    axes[0].bar(x + width, other_ops, width, label='Others', color=colors[2])
    axes[0].set_xlabel('Architecture')
    axes[0].set_ylabel('Ops Ratio')
    axes[0].set_title('Ops Ratio Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(arch_names)
    axes[0].legend()
    
    # 准备time数据
    binary_conv_time = [analyzed_data[arch]['time_data']['binary_conv_time_ratio'] for arch in arch_names]
    binary_conv1x1_time = [analyzed_data[arch]['time_data']['binary_conv1x1_time_ratio'] for arch in arch_names]
    other_time = [analyzed_data[arch]['time_data']['other_time_ratio'] for arch in arch_names]
    
    # 绘制time对比图
    axes[1].bar(x - width, binary_conv_time, width, label='Binary Conv', color=colors[0])
    axes[1].bar(x, binary_conv1x1_time, width, label='Binary Conv 1x1', color=colors[1])
    axes[1].bar(x + width, other_time, width, label='Others', color=colors[2])
    axes[1].set_xlabel('Architecture')
    axes[1].set_ylabel('Time Ratio')
    axes[1].set_title('Time Ratio Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(arch_names)
    axes[1].legend()
    
    plt.tight_layout()
    # 保存为PDF
    plt.savefig('architecture_comparison_bars.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print("Comparison charts saved to architecture_comparison_bars.pdf")

def print_summary(analyzed_data):
    """
    打印摘要信息
    """
    print("\n=== ops_data and time_data Analysis Summary ===")
    for arch_name, data in analyzed_data.items():
        ops_data = data['ops_data']
        time_data = data['time_data']
        
        print(f"\n{arch_name}:")
        print(f"  Total FLOPs: {ops_data['total_flops']:.2f}")
        print(f"  Total bit-ops: {ops_data['total_bitops']:.2f}")
        print(f"  Total Time: {time_data['total_time']:.2f} ms")
        print(f"  Total Ops (FLOPs + bit-ops/64): {ops_data['total_ops']:.2f}")
        print(f"  Binary Conv Ops Ratio: {ops_data['binary_conv_ops_ratio']:.2%}")
        print(f"  Binary Conv 1x1 Ops Ratio: {ops_data['binary_conv1x1_ops_ratio']:.2%}")
        print(f"  Other Ops Ratio: {ops_data['other_ops_ratio']:.2%}")
        print(f"  Binary Conv Time Ratio: {time_data['binary_conv_time_ratio']:.2%}")
        print(f"  Binary Conv 1x1 Time Ratio: {time_data['binary_conv1x1_time_ratio']:.2%}")
        print(f"  Other Time Ratio: {time_data['other_time_ratio']:.2%}")

# 主函数
def main():
    print("开始分析ops_data和time_data...")
    
    # 从pth文件加载架构数据
    raw_arch_data = load_arch_data_from_pth()
    
    # 检查是否成功加载到架构数据
    if not raw_arch_data:
        print("警告：没有加载到任何架构数据，请确保架构结果的pth文件存在。")
        return
    else:
        print(f"成功加载到 {len(raw_arch_data)} 个架构的数据")
    
    # 分析ops_data和time_data
    analyzed_data = analyze_ops_and_time_data(raw_arch_data)
    
    if not analyzed_data:
        print("错误：没有成功分析任何架构数据。")
        return
    
    # 打印摘要
    print_summary(analyzed_data)
    
    # 创建饼图
    create_pie_charts(analyzed_data)
    
    # 创建对比柱状图（如果有多于一个架构）
    if len(analyzed_data) > 1:
        create_comparison_bars(analyzed_data)
    
    print("分析完成！")

if __name__ == "__main__":
    main()