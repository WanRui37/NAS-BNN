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


# 新增：按层级别绘图，直观展示结构差异带来的延迟差异
def get_layer_info_from_arch(arch):
    """从 arch（包含 ops_data 和 time_data）提取按层信息：name, time, ops, type"""
    ops = arch.get('ops_data', {})
    time_data = arch.get('time_data', {})
    layer_names = ops.get('layer_names', [])
    flops = ops.get('flops', [])
    bitops = ops.get('bitops', [])

    infos = []
    for i, name in enumerate(layer_names):
        fl = flops[i] if i < len(flops) else 0
        bo = bitops[i] if i < len(bitops) else 0
        t = time_data.get(name, None)
        if t is None:
            # 尝试模糊匹配 time_data 中的键
            for k, v in time_data.items():
                if name in k or k in name:
                    t = v
                    break
        if t is None:
            t = 0.0
        ops_val = fl + bo / 64.0 * 0.85
        if 'binary_conv1x1' in name:
            typ = 'binary_conv1x1'
        elif 'binary_conv' in name:
            typ = 'binary_conv'
        else:
            typ = 'other'
        infos.append({'name': name, 'time': float(t), 'ops': float(ops_val), 'type': typ})
    return infos


# 修改：绘制双Y轴的折线图（累计延迟和累计Ops）
def create_single_plot_cumulative_ops_latency(raw_arch_data):
    """绘制单张双Y轴折线图：两个架构的累计延迟（左轴）和累计Ops（右轴）对比"""
    # 确保至少两个架构用于对比
    if len(raw_arch_data) < 2:
        print("Need at least two architectures for comparison")
        return

    arch_names = list(raw_arch_data.keys())
    # 使用所有架构进行对比
    arch_infos = []
    for name in arch_names:
        arch = raw_arch_data[name]
        info = get_layer_info_from_arch(arch)
        arch_infos.append((name, info))

    # 准备颜色方案和线型
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']
    
    # 创建双Y轴图
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax2 = ax1.twinx()  # 创建第二个Y轴
    
    # 计算每个架构的累计延迟和累计Ops
    for i, (arch_name, info) in enumerate(arch_infos):
        # 累计延迟
        times = [x['time'] for x in info]
        cum_times = np.cumsum(times) * 3
        
        # 累计Ops
        ops = [x['ops'] for x in info]
        cum_ops = np.cumsum(ops)
        
        arch_name_english = "Arch " + str(i+1)
        # 在左轴绘制累计延迟线
        ax1.plot(range(len(cum_times)), cum_times, 
                label=f'{arch_name_english} Cumulative Latency', 
                color=colors[i*2], 
                linestyle=line_styles[0], 
                marker=markers[i],
                linewidth=2,
                markersize=4)
        
        # 在右轴绘制累计Ops线
        ax2.plot(range(len(cum_ops)), cum_ops, 
                label=f'{arch_name_english} Cumulative Ops', 
                color=colors[i*2+1], 
                linestyle=line_styles[1], 
                marker=markers[i],
                linewidth=2,
                markersize=4)
    
    # 设置左轴属性（累计延迟）
    ax1.set_xlabel('Layer Index', fontsize=14)
    ax1.set_ylabel('Cumulative Latency (ms)', fontsize=14, color='#000000')
    ax1.tick_params(axis='y', labelcolor='#000000')
    
    # 设置右轴属性（累计Ops）
    ax2.set_ylabel('Cumulative Ops (FLOPs + bitops/64)', fontsize=14, color='#000000')
    ax2.tick_params(axis='y', labelcolor="#000000")
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper left')
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 设置坐标轴刻度
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('cumulative_ops_latency_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved dual-Y cumulative latency vs ops comparison to cumulative_ops_latency_comparison.pdf')

# 修改主函数，删除饼图调用
def main():
    print("开始分析ops_data和time_data...")
    
    # 从pth文件加载架构数据
    raw_arch_data = load_arch_data_from_pth()

    # 新增：绘制单张包含4条线的累计ops和累计延迟对比图
    try:
        create_single_plot_cumulative_ops_latency(raw_arch_data)
    except Exception as e:
        print(f"绘制单张累计Ops与延迟对比图时出错: {e}")

    # 检查是否成功加载到架构数据
    if not raw_arch_data:
        print("警告：没有加载到任何架构数据，请确保架构结果的pth文件存在。")
        return
    else:
        print(f"成功加载到 {len(raw_arch_data)} 个架构的数据")

if __name__ == "__main__":
    main()