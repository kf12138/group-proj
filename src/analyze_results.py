# src/analyze_results.py

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_all_results(logs_dir="results/logs"):
    """
    加载所有训练结果
    
    Args:
        logs_dir: 日志目录路径
    
    Returns:
        Dict: 所有结果的字典
    """
    results = {}
    
    if not os.path.exists(logs_dir):
        print(f"日志目录不存在: {logs_dir}")
        return results
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(logs_dir, "*.json"))
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取实验信息
            args = data['args']
            training_log = data['training_log']
            
            # 生成实验标识
            exp_id = f"{args['model']}_{args['mode']}_f{args['f']}_{args['attack_type']}"
            
            results[exp_id] = {
                'args': args,
                'training_log': training_log,
                'final_accuracy': data['final_accuracy'],
                'final_loss': data['final_loss'],
                'total_time': data['total_time'],
                'avg_time_per_epoch': data['avg_time_per_epoch'],
                'filepath': filepath
            }
            
        except Exception as e:
            print(f"加载文件失败 {filepath}: {e}")
    
    print(f"成功加载 {len(results)} 个实验结果")
    return results


def plot_loss_comparison_1(results: Dict[str, Any], save_path: str = None):
    """
    表一：average与krum在mlp与conv，f=0时的loss曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 筛选条件：f=0, attack=reverse, mode in [average, krum]
    target_experiments = []
    for exp_id, data in results.items():
        args = data['args']
        if (args['f'] == 0 and 
            args['attack_type'] == 'reverse' and 
            args['mode'] in ['average', 'krum']):
            target_experiments.append((exp_id, data))
    
    # 按模型分组
    mlp_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['model'] == 'mlp']
    conv_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['model'] == 'conv']
    
    # 绘制MLP结果
    colors = {'average': 'blue', 'krum': 'red'}
    for exp_id, data in mlp_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[0].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker='o')
    
    axes[0].set_title('MLP模型 - f=0 (无攻击)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制ConvNet结果
    for exp_id, data in conv_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[1].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker='s')
    
    axes[1].set_title('ConvNet模型 - f=0 (无攻击)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"表一已保存到: {save_path}")
    else:
        plt.show()


def plot_loss_comparison_2(results: Dict[str, Any], save_path: str = None):
    """
    表二：average与krum在mlp训练时，f=1、2时的loss曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 筛选条件：model=mlp, f in [1,2], attack=reverse, mode in [average, krum]
    target_experiments = []
    for exp_id, data in results.items():
        args = data['args']
        if (args['model'] == 'mlp' and 
            args['f'] in [1, 2] and 
            args['attack_type'] == 'reverse' and 
            args['mode'] in ['average', 'krum']):
            target_experiments.append((exp_id, data))
    
    # 按f值分组
    f1_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 1]
    f2_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 2]
    
    # 绘制f=1结果
    colors = {'average': 'blue', 'krum': 'red'}
    for exp_id, data in f1_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[0].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker='o')
    
    axes[0].set_title('MLP模型 - f=1 (1个拜占庭节点)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制f=2结果
    for exp_id, data in f2_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[1].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker='s')
    
    axes[1].set_title('MLP模型 - f=2 (2个拜占庭节点)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"表二已保存到: {save_path}")
    else:
        plt.show()


def plot_loss_comparison_3(results: Dict[str, Any], save_path: str = None):
    """
    表三：average与krum在conv训练时，f=1、2时的loss曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 筛选条件：model=conv, f in [1,2], attack=reverse, mode in [average, krum]
    target_experiments = []
    for exp_id, data in results.items():
        args = data['args']
        if (args['model'] == 'conv' and 
            args['f'] in [1, 2] and 
            args['attack_type'] == 'reverse' and 
            args['mode'] in ['average', 'krum']):
            target_experiments.append((exp_id, data))
    
    # 按f值分组
    f1_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 1]
    f2_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 2]
    
    # 绘制f=1结果
    colors = {'average': 'blue', 'krum': 'red'}
    for exp_id, data in f1_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[0].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker='o')
    
    axes[0].set_title('ConvNet模型 - f=1 (1个拜占庭节点)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制f=2结果
    for exp_id, data in f2_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[1].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker='s')
    
    axes[1].set_title('ConvNet模型 - f=2 (2个拜占庭节点)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"表三已保存到: {save_path}")
    else:
        plt.show()


def plot_loss_comparison_4(results: Dict[str, Any], save_path: str = None):
    """
    表四：krum/median/trimmed_mean/bulyan在mlp训练时，f=1、2的loss曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 筛选条件：model=mlp, f in [1,2], attack=reverse, mode in [krum, median, trimmed_mean, bulyan]
    target_experiments = []
    for exp_id, data in results.items():
        args = data['args']
        if (args['model'] == 'mlp' and 
            args['f'] in [1, 2] and 
            args['attack_type'] == 'reverse' and 
            args['mode'] in ['krum', 'median', 'trimmed_mean', 'bulyan']):
            target_experiments.append((exp_id, data))
    
    # 按f值分组
    f1_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 1]
    f2_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 2]
    
    # 绘制f=1结果
    colors = {'krum': 'red', 'median': 'green', 'trimmed_mean': 'orange', 'bulyan': 'purple'}
    markers = {'krum': 'o', 'median': 's', 'trimmed_mean': '^', 'bulyan': 'D'}
    
    for exp_id, data in f1_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[0].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker=markers[mode])
    
    axes[0].set_title('MLP模型 - f=1 (鲁棒聚合算法对比)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制f=2结果
    for exp_id, data in f2_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[1].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker=markers[mode])
    
    axes[1].set_title('MLP模型 - f=2 (鲁棒聚合算法对比)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"表四已保存到: {save_path}")
    else:
        plt.show()


def plot_loss_comparison_5(results: Dict[str, Any], save_path: str = None):
    """
    表五：krum/median/trimmed_mean/bulyan在conv训练时，f=1、2的loss曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 筛选条件：model=conv, f in [1,2], attack=reverse, mode in [krum, median, trimmed_mean, bulyan]
    target_experiments = []
    for exp_id, data in results.items():
        args = data['args']
        if (args['model'] == 'conv' and 
            args['f'] in [1, 2] and 
            args['attack_type'] == 'reverse' and 
            args['mode'] in ['krum', 'median', 'trimmed_mean', 'bulyan']):
            target_experiments.append((exp_id, data))
    
    # 按f值分组
    f1_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 1]
    f2_experiments = [(exp_id, data) for exp_id, data in target_experiments if data['args']['f'] == 2]
    
    # 绘制f=1结果
    colors = {'krum': 'red', 'median': 'green', 'trimmed_mean': 'orange', 'bulyan': 'purple'}
    markers = {'krum': 'o', 'median': 's', 'trimmed_mean': '^', 'bulyan': 'D'}
    
    for exp_id, data in f1_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[0].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker=markers[mode])
    
    axes[0].set_title('ConvNet模型 - f=1 (鲁棒聚合算法对比)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制f=2结果
    for exp_id, data in f2_experiments:
        mode = data['args']['mode']
        epochs = data['training_log']['epoch']
        losses = data['training_log']['loss']
        axes[1].plot(epochs, losses, color=colors[mode], linewidth=2, 
                     label=f"{mode} (最终: {losses[-1]:.3f})", marker=markers[mode])
    
    axes[1].set_title('ConvNet模型 - f=2 (鲁棒聚合算法对比)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"表五已保存到: {save_path}")
    else:
        plt.show()


def generate_summary_report(results: Dict[str, Any], save_path: str = None):
    """
    生成汇总报告
    """
    print("\n" + "="*60)
    print("联邦学习拜占庭容错实验汇总报告")
    print("="*60)
    
    # 按算法分组统计
    algo_stats = {}
    for exp_id, data in results.items():
        mode = data['args']['mode']
        if mode not in algo_stats:
            algo_stats[mode] = []
        algo_stats[mode].append(data)
    
    print(f"\n总共 {len(results)} 个实验结果，涉及 {len(algo_stats)} 种聚合算法")
    
    # 算法性能对比
    print("\n算法性能对比:")
    print("-" * 80)
    print(f"{'算法':<15} {'实验数':<8} {'平均准确率':<12} {'平均损失':<12} {'平均时间':<12}")
    print("-" * 80)
    
    for algo, data_list in algo_stats.items():
        accuracies = [d['final_accuracy'] for d in data_list]
        losses = [d['final_loss'] for d in data_list]
        times = [d['avg_time_per_epoch'] for d in data_list]
        
        avg_acc = np.mean(accuracies)
        avg_loss = np.mean(losses)
        avg_time = np.mean(times)
        
        print(f"{algo:<15} {len(data_list):<8} {avg_acc:.4f}±{np.std(accuracies):.4f} "
              f"{avg_loss:.4f}±{np.std(losses):.4f} {avg_time:.2f}±{np.std(times):.2f}s")
    
    # 最佳性能实验
    print("\n最佳性能实验:")
    print("-" * 80)
    best_acc = max(results.values(), key=lambda x: x['final_accuracy'])
    best_loss = min(results.values(), key=lambda x: x['final_loss'])
    fastest = min(results.values(), key=lambda x: x['avg_time_per_epoch'])
    
    print(f"最高准确率: {best_acc['final_accuracy']:.4f} ({list(results.keys())[list(results.values()).index(best_acc)]})")
    print(f"最低损失: {best_loss['final_loss']:.4f} ({list(results.keys())[list(results.values()).index(best_loss)]})")
    print(f"最快训练: {fastest['avg_time_per_epoch']:.2f}s ({list(results.keys())[list(results.values()).index(fastest)]})")
    
    # 保存报告
    if save_path:
        report_data = {
            'total_experiments': len(results),
            'algorithms': list(algo_stats.keys()),
            'algorithm_stats': {
                algo: {
                    'count': len(data_list),
                    'avg_accuracy': np.mean([d['final_accuracy'] for d in data_list]),
                    'std_accuracy': np.std([d['final_accuracy'] for d in data_list]),
                    'avg_loss': np.mean([d['final_loss'] for d in data_list]),
                    'std_loss': np.std([d['final_loss'] for d in data_list]),
                    'avg_time': np.mean([d['avg_time_per_epoch'] for d in data_list]),
                    'std_time': np.std([d['avg_time_per_epoch'] for d in data_list])
                }
                for algo, data_list in algo_stats.items()
            },
            'best_experiments': {
                'best_accuracy': {
                    'exp_id': list(results.keys())[list(results.values()).index(best_acc)],
                    'value': best_acc['final_accuracy']
                },
                'best_loss': {
                    'exp_id': list(results.keys())[list(results.values()).index(best_loss)],
                    'value': best_loss['final_loss']
                },
                'fastest': {
                    'exp_id': list(results.keys())[list(results.values()).index(fastest)],
                    'value': fastest['avg_time_per_epoch']
                }
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n汇总报告已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="分析训练结果")
    parser.add_argument('--logs-dir', default='results/logs', help='日志目录路径')
    parser.add_argument('--output-dir', default='results/plots', help='输出目录路径')
    parser.add_argument('--plot-type', choices=['all', 'table1', 'table2', 'table3', 'table4', 'table5'], 
                       default='all', help='绘图类型')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果
    results = load_all_results(args.logs_dir)
    
    if not results:
        print("没有找到实验结果，请先运行训练脚本并保存日志")
        return
    
    # 生成图表
    if args.plot_type in ['all', 'table1']:
        plot_loss_comparison_1(results, 
                             os.path.join(args.output_dir, 'table1_f0_comparison.png'))
    
    if args.plot_type in ['all', 'table2']:
        plot_loss_comparison_2(results, 
                             os.path.join(args.output_dir, 'table2_mlp_f12.png'))
    
    if args.plot_type in ['all', 'table3']:
        plot_loss_comparison_3(results, 
                             os.path.join(args.output_dir, 'table3_conv_f12.png'))
    
    if args.plot_type in ['all', 'table4']:
        plot_loss_comparison_4(results, 
                             os.path.join(args.output_dir, 'table4_mlp_robust.png'))
    
    if args.plot_type in ['all', 'table5']:
        plot_loss_comparison_5(results, 
                             os.path.join(args.output_dir, 'table5_conv_robust.png'))
    
    # 生成汇总报告
    generate_summary_report(results, 
                          os.path.join(args.output_dir, 'summary_report.json'))


if __name__ == "__main__":
    main() 