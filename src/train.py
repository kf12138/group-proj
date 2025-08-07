# src/train.py

import os
import sys
import argparse
import random
import time
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
import copy

# 将 src 目录加入导入路径
sys.path.append(os.path.dirname(__file__))

from worker import Worker
from krum import get_aggregator
from model import get_model


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss


def apply_byzantine_attack(grad_lists, bad_idxs, attack_type='reverse'):
    """
    应用不同类型的拜占庭攻击
    
    Args:
        grad_lists: 所有Worker的梯度列表
        bad_idxs: 拜占庭节点的索引列表
        attack_type: 攻击类型 ('reverse', 'zero', 'random', 'sign_flip')
    """
    for i in bad_idxs:
        if attack_type == 'reverse':
            # 反向攻击：梯度反向
            grad_lists[i] = [-5.0 * g for g in grad_lists[i]]
        elif attack_type == 'zero':
            # 零梯度攻击：发送零梯度
            grad_lists[i] = [torch.zeros_like(g) for g in grad_lists[i]]
        elif attack_type == 'random':
            # 随机攻击：发送随机梯度
            grad_lists[i] = [torch.randn_like(g) * 10.0 for g in grad_lists[i]]
        elif attack_type == 'sign_flip':
            # 符号翻转攻击：梯度符号翻转
            grad_lists[i] = [-g for g in grad_lists[i]]
        else:
            raise ValueError(f"未知的攻击类型: {attack_type}")


def save_training_log(training_log, args, save_dir="results/logs"):
    """
    保存训练日志到文件
    
    Args:
        training_log: 训练日志字典
        args: 命令行参数
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    filename = f"{args.model}_{args.mode}_f{args.f}_{args.attack_type}_n{args.n_workers}_e{args.epochs}.json"
    filepath = os.path.join(save_dir, filename)
    
    # 准备保存的数据
    save_data = {
        'args': vars(args),
        'training_log': training_log,
        'final_accuracy': training_log['accuracy'][-1],
        'final_loss': training_log['loss'][-1],
        'total_time': sum(training_log['time_per_epoch']),
        'avg_time_per_epoch': sum(training_log['time_per_epoch']) / len(training_log['time_per_epoch'])
    }
    
    # 保存到JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"训练日志已保存到: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'conv'], default='mlp',
                        help="模型结构：mlp 或 conv")
    parser.add_argument('--mode', choices=['average', 'krum', 'multikrum', 'median', 'trimmed_mean', 'bulyan'], default='krum',
                        help="聚合方式")
    parser.add_argument('--f', type=int, default=2, help="最大 Byzantine 节点数")
    parser.add_argument('--m', type=int, default=None,
                        help="Multi-Krum 中选取的梯度数 m，默认 n-f")
    parser.add_argument('--epochs', type=int, default=10, help="训练 Epoch 数")
    parser.add_argument('--batch-size', type=int, default=64, help="每个 Worker 的 batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="学习率")
    parser.add_argument('--n-workers', type=int, default=10, help="Worker 数量")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help="运行设备")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--attack-type', choices=['reverse', 'zero', 'random', 'sign_flip'], 
                        default='reverse', help="拜占庭攻击类型")
    parser.add_argument('--log-interval', type=int, default=1, help="日志记录间隔（epoch）")
    parser.add_argument('--save-log', action='store_true', help="保存训练日志到文件")
    args = parser.parse_args()

    # 参数验证
    if args.f >= args.n_workers // 2:
        raise ValueError(f"f={args.f} 太大，必须 < n_workers/2={args.n_workers//2}")

    # 固定随机种子，保证可复现
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    # 训练日志
    training_log = {
        'epoch': [],
        'accuracy': [],
        'loss': [],
        'time_per_epoch': []
    }

    # —— 数据准备 —— #
    transform = transforms.ToTensor()
    full_train = datasets.MNIST('data/raw', train=True, download=True, transform=transform)
    full_test  = datasets.MNIST('data/raw', train=False, download=True, transform=transform)

    # 划分给 n-workers
    total = len(full_train)
    per = total // args.n_workers
    splits = [per] * args.n_workers
    splits[-1] = total - per * (args.n_workers - 1)
    subsets = torch.utils.data.random_split(full_train, splits)
    workers = [Worker(sub, batch_size=args.batch_size, shuffle=True)
               for sub in subsets]

    # 全局测试集 Loader
    test_loader = DataLoader(full_test, batch_size=1000, shuffle=False)

    # —— 模型初始化 —— #
    ModelClass = get_model(args.model)
    model = ModelClass().to(device)
    loss_fn = nn.CrossEntropyLoss()
    lr = args.lr

    it_per_epoch = len(subsets[0]) // args.batch_size

    print(f"开始训练 - 模型: {args.model}, 聚合: {args.mode}, 攻击: {args.attack_type}, "
          f"拜占庭节点: {args.f}/{args.n_workers}")
    print(f"设备: {device}, 学习率: {lr}, 批次大小: {args.batch_size}")

    # —— 训练循环 —— #
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        
        # 记录每轮梯度聚合的统计信息
        grad_norms = []
        
        for it in range(it_per_epoch):
            # 并行收集每个 Worker 的梯度
            def worker_grad(w):
                local_model = copy.deepcopy(model)#采用深拷贝，避免多线程间梯度混乱
                return w.compute_gradient(local_model, loss_fn, device)
            with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                grad_lists = list(executor.map(worker_grad, workers))

            # 模拟 Byzantine：随机选 f 个索引，应用指定类型的攻击
            if args.f > 0:
                bad_idxs = random.sample(range(args.n_workers), args.f)
                apply_byzantine_attack(grad_lists, bad_idxs, args.attack_type)

            # 聚合
            if args.mode == 'average':
                agg = []
                for layer_grads in zip(*grad_lists):
                    stacked = torch.stack(layer_grads, dim=0)
                    agg.append(stacked.mean(dim=0))
            else:
                # 使用拜占庭容错聚合算法
                aggregator = get_aggregator(args.mode)
                if args.mode == 'multikrum':
                    agg = aggregator(grad_lists, args.f, args.m)
                else:
                    agg = aggregator(grad_lists, args.f)

            # 记录聚合梯度的范数
            agg_norm = sum(g.pow(2).sum().item() for g in agg) ** 0.5
            grad_norms.append(agg_norm)

            # 4) 更新模型参数
            with torch.no_grad():
                for p, g in zip(model.parameters(), agg):
                    p -= lr * g

        epoch_time = time.time() - epoch_start_time
        
        # —— 评估 —— #
        acc, loss = evaluate(model, device, test_loader)
        
        # 记录训练日志
        training_log['epoch'].append(epoch)
        training_log['accuracy'].append(acc)
        training_log['loss'].append(loss)
        training_log['time_per_epoch'].append(epoch_time)
        
        # 打印训练信息
        attack_str = args.attack_type if args.f > 0 else 'null'
        if epoch % args.log_interval == 0:
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            print(f"Epoch {epoch:>2}/{args.epochs} | "
                  f"Model: {args.model:>4} | Mode: {args.mode:>9} | "
                  f"Attack: {attack_str:>8} | Byzantine f={args.f:<2} | "
                  f"Test Acc: {acc*100:5.2f}% | Loss: {loss:6.4f} | "
                  f"Grad Norm: {avg_grad_norm:8.2f} | Time: {epoch_time:5.2f}s")

    # 打印最终结果
    final_acc = training_log['accuracy'][-1]
    final_loss = training_log['loss'][-1]
    total_time = sum(training_log['time_per_epoch'])
    
    print(f"\n训练完成！")
    print(f"最终测试准确率: {final_acc*100:.2f}%")
    print(f"最终测试损失: {final_loss:.4f}")
    print(f"总训练时间: {total_time:.2f}秒")
    print(f"平均每轮时间: {total_time/args.epochs:.2f}秒")

    # 保存训练日志
    if args.save_log:
        save_training_log(training_log, args)

    return training_log


if __name__ == "__main__":
    main()
