import torch

def krum(grad_lists, f):
    """
    KRUM 聚合：从 grad_lists 中选出最“可信”的一个梯度列表。
    grad_lists: List[List[Tensor]]，外层长度为 n（worker 数量），内层长度为参数层数。
    f: 最大 Byzantine 节点数（不可超过 n//2 - 1）。
    返回: List[Tensor]，与 model.parameters() 对应的聚合后梯度。
    """
    n = len(grad_lists)
    # 扁平化每个列表为一个向量
    flats = [torch.cat([g.view(-1) for g in gl]) for gl in grad_lists]
    scores = []
    for i in range(n):
        # 计算第 i 个向量到其他所有向量的距离平方
        dists = [(flats[i] - flats[j]).pow(2).sum().item()
                 for j in range(n) if j != i]
        dists.sort()
        # 累加最接近的 n-f-2 个距离
        scores.append(sum(dists[: n - f - 2]))
    # 选得分最小的索引
    i_star = int(torch.argmin(torch.tensor(scores)))
    # 把该向量拆回各层梯度形状
    chosen = flats[i_star]
    agg = []
    idx = 0
    for g in grad_lists[0]:
        numel = g.numel()
        agg.append(chosen[idx: idx + numel].view_as(g))
        idx += numel
    return agg


def multi_krum(grad_lists, f, m=None):
    """
    Multi-Krum 聚合：选出 m 个“最可信”梯度列表后再平均。
    grad_lists: List[List[Tensor]]
    f: 最大 Byzantine 节点数
    m: 从中选取的梯度数量，默认 m = n - f
    返回: List[Tensor]，聚合后梯度
    """
    n = len(grad_lists)
    if m is None:
        m = n - f
    # 扁平化
    flats = [torch.cat([g.view(-1) for g in gl]) for gl in grad_lists]
    scores = []
    for i in range(n):
        dists = [(flats[i] - flats[j]).pow(2).sum().item()
                 for j in range(n) if j != i]
        dists.sort()
        scores.append(sum(dists[: n - f - 2]))
    # 选出得分最小的 m 个索引
    idxs = sorted(range(n), key=lambda i: scores[i])[:m]
    # 对这些索引对应的梯度列表按层平均
    agg = []
    num_layers = len(grad_lists[0])
    for layer in range(num_layers):
        stacked = torch.stack([grad_lists[i][layer] for i in idxs], dim=0)
        agg.append(stacked.mean(dim=0))
    return agg
