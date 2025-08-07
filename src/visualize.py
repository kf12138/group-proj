# src/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Any


def plot_training_curves(logs: Dict[str, Any], save_path: str = None):
    """
    绘制训练曲线
    
    Args:
        logs: 训练日志字典，包含 'epoch', 'accuracy', 'loss', 'time_per_epoch' 等键
        save_path: 保存图片的路径，如果为None则显示图片
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = logs['epoch']
    
    # 准确率曲线
    axes[0, 0].plot(epochs, logs['accuracy'], 'b-', linewidth=2)
    axes[0, 0].set_title('测试准确率')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].grid(True)
    
    # 损失曲线
    axes[0, 1].plot(epochs, logs['loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('测试损失')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].grid(True)
    
    # 每轮训练时间
    axes[1, 0].plot(epochs, logs['time_per_epoch'], 'g-', linewidth=2)
    axes[1, 0].set_title('每轮训练时间')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('时间 (秒)')
    axes[1, 0].grid(True)
    
    # 累积训练时间
    cumulative_time = np.cumsum(logs['time_per_epoch'])
    axes[1, 1].plot(epochs, cumulative_time, 'm-', linewidth=2)
    axes[1, 1].set_title('累积训练时间')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('时间 (秒)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()


def compare_algorithms(results: Dict[str, Dict], save_path: str = None):
    """
    比较不同聚合算法的性能
    
    Args:
        results: 字典，键为算法名称，值为训练日志
        save_path: 保存图片的路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (algo_name, log) in enumerate(results.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        epochs = log['epoch']
        
        # 准确率比较
        axes[0, 0].plot(epochs, log['accuracy'], color=color, marker=marker, 
                        linewidth=2, label=algo_name, markersize=6)
        
        # 损失比较
        axes[0, 1].plot(epochs, log['loss'], color=color, marker=marker, 
                        linewidth=2, label=algo_name, markersize=6)
        
        # 最终准确率
        final_acc = log['accuracy'][-1]
        axes[1, 0].bar(algo_name, final_acc, color=color, alpha=0.7)
        
        # 平均训练时间
        avg_time = np.mean(log['time_per_epoch'])
        axes[1, 1].bar(algo_name, avg_time, color=color, alpha=0.7)
    
    axes[0, 0].set_title('准确率比较')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('损失比较')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('最终准确率')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].set_title('平均训练时间')
    axes[1, 1].set_ylabel('时间 (秒)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较图已保存到: {save_path}")
    else:
        plt.show()


def plot_byzantine_impact(results: Dict[str, Dict], save_path: str = None):
    """
    绘制拜占庭节点数量对性能的影响
    
    Args:
        results: 字典，键为拜占庭节点数量，值为训练日志
        save_path: 保存图片的路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    byzantine_counts = sorted(results.keys())
    final_accuracies = [results[f]['accuracy'][-1] for f in byzantine_counts]
    final_losses = [results[f]['loss'][-1] for f in byzantine_counts]
    
    # 准确率 vs 拜占庭节点数
    axes[0].plot(byzantine_counts, final_accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('拜占庭节点数对准确率的影响')
    axes[0].set_xlabel('拜占庭节点数')
    axes[0].set_ylabel('最终准确率')
    axes[0].grid(True)
    
    # 损失 vs 拜占庭节点数
    axes[1].plot(byzantine_counts, final_losses, 'ro-', linewidth=2, markersize=8)
    axes[1].set_title('拜占庭节点数对损失的影响')
    axes[1].set_xlabel('拜占庭节点数')
    axes[1].set_ylabel('最终损失')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"拜占庭影响图已保存到: {save_path}")
    else:
        plt.show()


def save_results(results: Dict[str, Any], filepath: str):
    """
    保存训练结果到JSON文件
    
    Args:
        results: 训练结果字典
        filepath: 保存路径
    """
    # 将numpy数组转换为列表以便JSON序列化
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载训练结果
    
    Args:
        filepath: 文件路径
    
    Returns:
        训练结果字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 将列表转换回numpy数组
    for key, value in results.items():
        if isinstance(value, list):
            results[key] = np.array(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list):
                    value[k] = np.array(v)
    
    return results


if __name__ == "__main__":
    # 示例用法
    print("可视化工具已加载，可以导入使用以下函数：")
    print("- plot_training_curves(): 绘制训练曲线")
    print("- compare_algorithms(): 比较不同算法")
    print("- plot_byzantine_impact(): 分析拜占庭影响")
    print("- save_results() / load_results(): 保存/加载结果") 