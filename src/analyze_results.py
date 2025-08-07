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


def plot_algorithm_comparison(results: Dict[str, Any], save_path: str = None):
    """
    绘制算法对比图
    
    Args:
        results: 实验结果字典
        save_path: 保存路径
    """
    # 按算法分组
    algorithms = {}
    for exp_id, data in results.items():
        mode = data['args']['mode']
        if mode not in algorithms:
            algorithms[mode] = []
        algorithms[mode].append(data)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (algo, data_list) in enumerate(algorithms.items()):
        color = colors[i % len(colors)]
        
        # 收集数据
        accuracies = [d['final_accuracy'] for d in data_list]
        losses = [d['final_loss'] for d in data_list]
        times = [d['avg_time_per_epoch'] for d in data_list]
        
        # 准确率对比
        axes[0, 0].bar(f"{algo}\n(n={len(data_list)})", np.mean(accuracies), 
                       yerr=np.std(accuracies), color=color, alpha=0.7, capsize=5)
        
        # 损失对比
        axes[0, 1].bar(f"{algo}\n(n={len(data_list)})", np.mean(losses), 
                       yerr=np.std(losses), color=color, alpha=0.7, capsize=5)
        
        # 时间对比
        axes[1, 0].bar(f"{algo}\n(n={len(data_list)})", np.mean(times), 
                       yerr=np.std(times), color=color, alpha=0.7, capsize=5)
        
        # 准确率vs时间散点图
        axes[1, 1].scatter(np.mean(times), np.mean(accuracies), 
                           s=100, color=color, alpha=0.7, label=algo)
    
    axes[0, 0].set_title('最终准确率对比')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].set_title('最终损失对比')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 0].set_title('平均训练时间对比')
    axes[1, 0].set_ylabel('时间 (秒)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].set_title('准确率 vs 训练时间')
    axes[1, 1].set_xlabel('平均训练时间 (秒)')
    axes[1, 1].set_ylabel('最终准确率')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"算法对比图已保存到: {save_path}")
    else:
        plt.show()


def plot_training_curves(results: Dict[str, Any], save_path: str = None):
    """
    绘制训练曲线
    
    Args:
        results: 实验结果字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (exp_id, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        training_log = data['training_log']
        
        epochs = training_log['epoch']
        accuracies = training_log['accuracy']
        losses = training_log['loss']
        times = training_log['time_per_epoch']
        
        # 准确率曲线
        axes[0, 0].plot(epochs, accuracies, color=color, alpha=0.7, 
                        label=f"{exp_id} (最终: {accuracies[-1]:.3f})")
        
        # 损失曲线
        axes[0, 1].plot(epochs, losses, color=color, alpha=0.7, 
                        label=f"{exp_id} (最终: {losses[-1]:.3f})")
        
        # 训练时间
        axes[1, 0].plot(epochs, times, color=color, alpha=0.7, 
                        label=f"{exp_id} (平均: {np.mean(times):.2f}s)")
        
        # 累积时间
        cumulative_time = np.cumsum(times)
        axes[1, 1].plot(epochs, cumulative_time, color=color, alpha=0.7, 
                        label=f"{exp_id} (总计: {cumulative_time[-1]:.2f}s)")
    
    axes[0, 0].set_title('训练准确率曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('训练损失曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('每轮训练时间')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('时间 (秒)')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('累积训练时间')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('时间 (秒)')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线图已保存到: {save_path}")
    else:
        plt.show()


def plot_byzantine_impact(results: Dict[str, Any], save_path: str = None):
    """
    绘制拜占庭节点数量影响
    
    Args:
        results: 实验结果字典
        save_path: 保存路径
    """
    # 按算法和拜占庭节点数分组
    impact_data = {}
    
    for exp_id, data in results.items():
        mode = data['args']['mode']
        f = data['args']['f']
        
        if mode not in impact_data:
            impact_data[mode] = {}
        
        impact_data[mode][f] = {
            'accuracy': data['final_accuracy'],
            'loss': data['final_loss'],
            'time': data['avg_time_per_epoch']
        }
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (algo, f_data) in enumerate(impact_data.items()):
        color = colors[i % len(colors)]
        
        f_values = sorted(f_data.keys())
        accuracies = [f_data[f]['accuracy'] for f in f_values]
        losses = [f_data[f]['loss'] for f in f_values]
        times = [f_data[f]['time'] for f in f_values]
        
        # 准确率 vs 拜占庭节点数
        axes[0, 0].plot(f_values, accuracies, 'o-', color=color, 
                        linewidth=2, markersize=8, label=algo)
        
        # 损失 vs 拜占庭节点数
        axes[0, 1].plot(f_values, losses, 's-', color=color, 
                        linewidth=2, markersize=8, label=algo)
        
        # 时间 vs 拜占庭节点数
        axes[1, 0].plot(f_values, times, '^-', color=color, 
                        linewidth=2, markersize=8, label=algo)
        
        # 准确率 vs 时间散点图
        axes[1, 1].scatter(times, accuracies, color=color, s=100, 
                           alpha=0.7, label=algo)
    
    axes[0, 0].set_title('拜占庭节点数对准确率的影响')
    axes[0, 0].set_xlabel('拜占庭节点数 (f)')
    axes[0, 0].set_ylabel('最终准确率')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('拜占庭节点数对损失的影响')
    axes[0, 1].set_xlabel('拜占庭节点数 (f)')
    axes[0, 1].set_ylabel('最终损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('拜占庭节点数对训练时间的影响')
    axes[1, 0].set_xlabel('拜占庭节点数 (f)')
    axes[1, 0].set_ylabel('平均训练时间 (秒)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('准确率 vs 训练时间')
    axes[1, 1].set_xlabel('平均训练时间 (秒)')
    axes[1, 1].set_ylabel('最终准确率')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"拜占庭影响图已保存到: {save_path}")
    else:
        plt.show()


def generate_summary_report(results: Dict[str, Any], save_path: str = None):
    """
    生成汇总报告
    
    Args:
        results: 实验结果字典
        save_path: 保存路径
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
    parser.add_argument('--plot-type', choices=['all', 'comparison', 'curves', 'byzantine'], 
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
    if args.plot_type in ['all', 'comparison']:
        plot_algorithm_comparison(results, 
                                os.path.join(args.output_dir, 'algorithm_comparison.png'))
    
    if args.plot_type in ['all', 'curves']:
        plot_training_curves(results, 
                           os.path.join(args.output_dir, 'training_curves.png'))
    
    if args.plot_type in ['all', 'byzantine']:
        plot_byzantine_impact(results, 
                            os.path.join(args.output_dir, 'byzantine_impact.png'))
    
    # 生成汇总报告
    generate_summary_report(results, 
                          os.path.join(args.output_dir, 'summary_report.json'))


if __name__ == "__main__":
    main() 