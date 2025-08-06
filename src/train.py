# src/train.py

import os
import sys
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 将 src 目录加入导入路径
sys.path.append(os.path.dirname(__file__))

from worker import Worker
from krum import krum, multi_krum
from model import get_model


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'conv'], default='mlp',
                        help="模型结构：mlp 或 conv")
    parser.add_argument('--mode', choices=['average', 'krum', 'multikrum'], default='krum',
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
    args = parser.parse_args()

    # 固定随机种子，保证可复现
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

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

    # —— 训练循环 —— #
    for epoch in range(1, args.epochs + 1):
        model.train()
        for it in range(it_per_epoch):
            # 1) 收集每个 Worker 的梯度
            grad_lists = [w.compute_gradient(model, loss_fn, device) for w in workers]

            # 2) 模拟 Byzantine：随机选 f 个索引，把它们的梯度反向放大
            if args.f > 0:
                bad_idxs = random.sample(range(args.n_workers), args.f)
                for i in bad_idxs:
                    grad_lists[i] = [-5.0 * g for g in grad_lists[i]]

            # 3) 聚合
            if args.mode == 'average':
                agg = []
                for layer_grads in zip(*grad_lists):
                    stacked = torch.stack(layer_grads, dim=0)
                    agg.append(stacked.mean(dim=0))
            elif args.mode == 'krum':
                agg = krum(grad_lists, args.f)
            else:  # multikrum
                agg = multi_krum(grad_lists, args.f, args.m)

            # 4) 更新模型参数
            with torch.no_grad():
                for p, g in zip(model.parameters(), agg):
                    p -= lr * g

        # —— 评估 —— #
        acc = evaluate(model, device, test_loader)
        print(f"Epoch {epoch:>2}/{args.epochs} | Model: {args.model:>4} | "
              f"Mode: {args.mode:>9} | Byzantine f={args.f:<2} | "
              f"Test Acc: {acc*100:5.2f}%")

    print("训练完毕！")


if __name__ == "__main__":
    main()
