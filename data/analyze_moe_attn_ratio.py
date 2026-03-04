import re
import os
import argparse
from collections import defaultdict

def analyze_log(file_path, verbose=True):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    config_pattern = re.compile(r"配置:\s*Batch Size\s*=\s*(\d+),\s*Seq Length\s*=\s*(\d+)")
    data_pattern = re.compile(r"(L\d+)\s*\|\s*([\d.]+)\s*ms\s*\|\s*[\d.]+\s*ms\s*\|\s*([\d.]+)\s*ms")

    data = defaultdict(lambda: {'ratios': [], 'attn': [], 'moe': []})
    current_config = None

    global_max = {'ratio': -1, 'config': None, 'layer': None}
    global_min = {'ratio': float('inf'), 'config': None, 'layer': None}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if config_match := config_pattern.search(line):
                current_config = f"BS={config_match.group(1)}, Seq={config_match.group(2)}"
                continue

            if current_config and (data_match := data_pattern.search(line)):
                layer, attn_time, moe_time = data_match.group(1), float(data_match.group(2)), float(data_match.group(3))
                if moe_time > 0:
                    ratio = attn_time / moe_time
                    data[current_config]['ratios'].append(ratio)
                    data[current_config]['attn'].append(attn_time)
                    data[current_config]['moe'].append(moe_time)

                    # 更新单层全局极值
                    if ratio > global_max['ratio']: global_max = {'ratio': ratio, 'config': current_config, 'layer': layer}
                    if ratio < global_min['ratio']: global_min = {'ratio': ratio, 'config': current_config, 'layer': layer}

    print(f"分析文件: {file_path}\n")
    
    if verbose: print("=== 每个配置的详细统计 ===")
        
    stats = {}
    for config, vals in data.items():
        if not vals['ratios']:
            if verbose: print(f"配置 {config:<15}: 无有效数据 (可能 OOM)")
            continue
            
        avg_ratio = sum(vals['ratios']) / len(vals['ratios'])
        attn_ratio = max(vals['attn']) / min(vals['attn']) if min(vals['attn']) > 0 else float('inf')
        moe_ratio = max(vals['moe']) / min(vals['moe']) if min(vals['moe']) > 0 else float('inf')
        
        stats[config] = {'avg': avg_ratio, 'attn_var': attn_ratio, 'moe_var': moe_ratio}
        
        if verbose:
            print(f"配置 {config:<15}: 平均倍数={avg_ratio:>6.2f}x | Attn极值比={attn_ratio:>5.2f}x | MoE极值比={moe_ratio:>5.2f}x")

    if not stats:
        print("未找到有效数据。")
        return

    # 提取极值配置
    def get_extreme(key, is_max=True):
        cmp_func = max if is_max else min
        conf = cmp_func(stats, key=lambda k: stats[k][key])
        return stats[conf][key], conf

    print("\n=== 全局极值 ===")
    if global_max['config']:
        print(f"单层最小倍数: {global_min['ratio']:.2f}x  -> 配置 [{global_min['config']}], 层 [{global_min['layer']}]")
        print(f"单层最大倍数: {global_max['ratio']:.2f}x  -> 配置 [{global_max['config']}], 层 [{global_max['layer']}]")

        
    print("\n=== 平均倍数 ===")
    max_val, max_conf = get_extreme('avg', True)
    min_val, min_conf = get_extreme('avg', False)
    print(f"最小平均倍数: {min_val:.2f}x  -> 配置 [{min_conf}]")
    print(f"最大平均倍数: {max_val:.2f}x  -> 配置 [{max_conf}]")
        
    # print("\n=== 同一配置下不同层时间极值比 (Max/Min) ===")
    # for name, key in [("Attention", "attn_var"), ("MoE", "moe_var")]:
    #     max_val, max_conf = get_extreme(key, True)
    #     min_val, min_conf = get_extreme(key, False)
    #     print(f"{name} 最大极值比: {max_val:.2f}x  -> 配置 [{max_conf}]")
    #     print(f"{name} 最小极值比: {min_val:.2f}x  -> 配置 [{min_conf}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析日志文件中 MoE 和 Attention 的时间倍数")
    parser.add_argument("file", nargs="?", default="output-5090/NVIDIA_GeForce_RTX_5090_layers5.log", help="要分析的日志文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出每个配置的详细倍数信息")
    
    args = parser.parse_args()
    analyze_log(args.file, verbose=args.verbose)
