# src/krum.py

import torch

def krum(grad_lists, f):
    """原始 KRUM 聚合"""
    n = len(grad_lists)
    flats = [torch.cat([g.view(-1) for g in gl]) for gl in grad_lists]
    scores = []
    for i in range(n):
        dists = [(flats[i] - flats[j]).pow(2).sum().item()
                 for j in range(n) if j != i]
        dists.sort()
        scores.append(sum(dists[: n - f - 2]))
    i_star = int(torch.argmin(torch.tensor(scores)))
    return _unflatten(grad_lists, flats[i_star])


def multi_krum(grad_lists, f, m=None):
    """原始 Multi‐Krum 聚合"""
    n = len(grad_lists)
    if m is None:
        m = n - f
    flats = [torch.cat([g.view(-1) for g in gl]) for gl in grad_lists]
    scores = []
    for i in range(n):
        dists = [(flats[i] - flats[j]).pow(2).sum().item()
                 for j in range(n) if j != i]
        dists.sort()
        scores.append(sum(dists[: n - f - 2]))
    idxs = sorted(range(n), key=lambda i: scores[i])[:m]
    # 平均这 m 个 grad_lists
    return [torch.mean(torch.stack([gl[l] for l in idxs], dim=0), dim=0)
            for gl in zip(*grad_lists)]


def coordinate_median(grad_lists, **kwargs):
    """
    Coordinate‐wise Median
    对每个参数坐标单独取中位数
    """
    # 对每层分别计算
    agg = []
    for layer_grads in zip(*grad_lists):
        stacked = torch.stack(layer_grads, dim=0)  # [n, *shape]
        median = torch.median(stacked, dim=0).values
        agg.append(median)
    return agg


def trimmed_mean(grad_lists, f, **kwargs):
    """
    Coordinate‐wise Trimmed Mean
    去掉每个坐标上 最大 f 个 和 最小 f 个 后再平均
    """
    agg = []
    for layer_grads in zip(*grad_lists):
        stacked = torch.stack(layer_grads, dim=0)  # [n, *shape]
        # sort 每个坐标
        sorted_grads, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_grads[f: len(grad_lists)-f]  # 去两端 f 个
        agg.append(torch.mean(trimmed, dim=0))
    return agg


def bulyan(grad_lists, f, m=None):
    """
    Bulyan = (Multi‐)Krum 内部多轮筛 + Coordinate‐wise Trimmed Mean
    1) 用 Multi-Krum 多轮筛选出 m = n - 2f 最可信梯度
    2) 对这 m 个梯度使用坐标截尾均值（Trimmed Mean），去掉两端 f 个后平均
    """
    n = len(grad_lists)
    if m is None:
        m = n - 2*f
    # 第一步：Multi‐Krum 选出 m 个
    from functools import partial
    # 复用 multi_krum 但不做平均，而拿到 idxs
    flats = [torch.cat([g.view(-1) for g in gl]) for gl in grad_lists]
    scores = []
    for i in range(n):
        dists = [(flats[i] - flats[j]).pow(2).sum().item()
                 for j in range(n) if j != i]
        dists.sort()
        scores.append(sum(dists[: n - f - 2]))
    idxs = sorted(range(n), key=lambda i: scores[i])[:m]
    selected = [grad_lists[i] for i in idxs]
    # 第二步：对 selected 用 trimmed_mean，去掉 f 个
    return trimmed_mean(selected, f)


def _unflatten(grad_lists, flat_tensor):
    """辅助：把一个扁平向量拆回 grad_lists[0] 的形状列表"""
    agg = []
    idx = 0
    for g in grad_lists[0]:
        numel = g.numel()
        agg.append(flat_tensor[idx: idx+numel].view_as(g))
        idx += numel
    return agg
