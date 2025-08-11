# src/analyze_results_02.py

import os
import json
import glob
import argparse
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 字体设置（尽量兼容）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_all_results(logs_dir: str) -> Dict[str, Any]:
    """
    加载 results/logs 下的所有 json 结果文件。
    返回以 exp_id 为键的结果字典。
    """
    results: Dict[str, Any] = {}
    if not os.path.exists(logs_dir):
        print(f"日志目录不存在: {logs_dir}")
        return results

    for filepath in glob.glob(os.path.join(logs_dir, "*.json")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            args = data.get('args', {})
            exp_id = f"{args.get('model','?')}_{args.get('mode','?')}_f{args.get('f','?')}_{args.get('attack_type','?')}"
            results[exp_id] = {
                'args': args,
                'final_accuracy': data.get('final_accuracy', None),
                'final_loss': data.get('final_loss', None),
                'training_log': data.get('training_log', {}),
                'filepath': filepath,
            }
        except Exception as e:
            print(f"加载失败 {filepath}: {e}")
    print(f"成功加载 {len(results)} 个实验结果")
    return results


def aggregate_accuracy_by_f(results: Dict[str, Any], model: str, attack_type: str = None,
                            modes: List[str] = None) -> Dict[str, Dict[int, List[float]]]:
    """
    按算法(mode)聚合指定 model 下不同 f 的准确率。
    返回: mode -> { f -> [accuracy,...] }
    可选按 attack_type 与 modes 过滤。
    """
    # 默认支持的算法全集
    all_modes = ['average', 'krum', 'median', 'trimmed_mean', 'bulyan']
    if modes is None:
        modes = all_modes

    agg: Dict[str, Dict[int, List[float]]] = {m: {} for m in modes}

    for _, data in results.items():
        args = data['args']
        if args.get('model') != model:
            continue
        if attack_type is not None and args.get('attack_type') != attack_type:
            continue
        mode = args.get('mode')
        if mode not in modes:
            continue
        try:
            f_val = int(args.get('f', -1))
        except Exception:
            continue
        acc = data.get('final_accuracy')
        if acc is None:
            continue
        agg.setdefault(mode, {}).setdefault(f_val, []).append(acc)

    return agg


def plot_accuracy_table(agg: Dict[str, Dict[int, List[float]]], model: str, save_path: str,
                        title_suffix: str = ""):
    """
    绘制 横轴为 f，纵轴为 准确率 的折线图。每条线代表一种算法。
    对同一 (mode,f) 多次实验，绘制均值；若存在多个 f 点，连接成线。
    横轴刻度步长固定为 1（f 为整数）。
    """
    plt.figure(figsize=(10, 6))

    # 颜色与样式
    palette = {
        'average': '#1f77b4',
        'krum': '#d62728',
        'median': '#2ca02c',
        'trimmed_mean': '#ff7f0e',
        'bulyan': '#9467bd',
    }
    markers = {
        'average': 'o',
        'krum': 's',
        'median': '^',
        'trimmed_mean': 'D',
        'bulyan': 'v',
    }

    all_f_values = set()

    for mode, f_to_accs in agg.items():
        if not f_to_accs:
            continue
        f_vals = sorted(f_to_accs.keys())
        means = [float(np.mean(f_to_accs[f])) for f in f_vals]
        all_f_values.update(f_vals)
        plt.plot(f_vals, means, label=mode, color=palette.get(mode, None),
                 marker=markers.get(mode, 'o'), linewidth=2)

    # 设置 x 轴为整数刻度，步长为 1
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if all_f_values:
        min_f, max_f = min(all_f_values), max(all_f_values)
        ax.set_xlim(min_f - 0.1, max_f + 0.1)

    title_core = f"{model.upper()} 模型：准确率随 f 变化"
    if title_suffix:
        title_core += f"（{title_suffix}）"
    plt.title(title_core)
    plt.xlabel("f (拜占庭节点数)")
    plt.ylabel("准确率")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="按 f 分析各算法准确率 (表格版)")
    parser.add_argument('--logs-dir', default='results/logs', help='日志目录路径')
    parser.add_argument('--output-dir', default='results/plots', help='输出目录路径')
    parser.add_argument('--attack-type', default=None, help='可选过滤攻击类型，如 reverse/zero/random/sign_flip')
    args = parser.parse_args()

    results = load_all_results(args.logs_dir)
    if not results:
        print("没有找到实验结果")
        return

    # 表1：average 与 krum 在 MLP 上
    modes_ak = ['average', 'krum']
    agg_mlp_ak = aggregate_accuracy_by_f(results, model='mlp', attack_type=args.attack_type, modes=modes_ak)
    save_path_mlp_ak = os.path.join(args.output_dir, 'table1_mlp_average_krum_accuracy_vs_f.png')
    plot_accuracy_table(agg_mlp_ak, model='mlp', save_path=save_path_mlp_ak, title_suffix='average vs krum')

    # 表2：average 与 krum 在 Conv 上
    agg_conv_ak = aggregate_accuracy_by_f(results, model='conv', attack_type=args.attack_type, modes=modes_ak)
    save_path_conv_ak = os.path.join(args.output_dir, 'table2_conv_average_krum_accuracy_vs_f.png')
    plot_accuracy_table(agg_conv_ak, model='conv', save_path=save_path_conv_ak, title_suffix='average vs krum')

    # 表3：krum/median/trimmed_mean/bulyan 在 MLP 上
    modes_robust = ['krum', 'median', 'trimmed_mean', 'bulyan']
    agg_mlp_robust = aggregate_accuracy_by_f(results, model='mlp', attack_type=args.attack_type, modes=modes_robust)
    save_path_mlp_robust = os.path.join(args.output_dir, 'table3_mlp_robust_accuracy_vs_f.png')
    plot_accuracy_table(agg_mlp_robust, model='mlp', save_path=save_path_mlp_robust, title_suffix='鲁棒聚合算法对比')

    # 表4：krum/median/trimmed_mean/bulyan 在 Conv 上
    agg_conv_robust = aggregate_accuracy_by_f(results, model='conv', attack_type=args.attack_type, modes=modes_robust)
    save_path_conv_robust = os.path.join(args.output_dir, 'table4_conv_robust_accuracy_vs_f.png')
    plot_accuracy_table(agg_conv_robust, model='conv', save_path=save_path_conv_robust, title_suffix='鲁棒聚合算法对比')


if __name__ == '__main__':
    main() 