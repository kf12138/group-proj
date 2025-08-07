import torch

def krum(grad_lists, f):
    """
    KRUM 聚合：从 grad_lists 中选出最"可信"的一个梯度列表。
    grad_lists: List[List[Tensor]]，外层长度为 n（worker 数量），内层长度为参数层数。
    f: 最大 Byzantine 节点数（不可超过 n//2 - 1）。
    返回: List[Tensor]，与 model.parameters() 对应的聚合后梯度。
    """
    n = len(grad_lists)
    
    # 参数验证
    if n == 0:
        raise ValueError("grad_lists 不能为空")
    if f < 0:
        raise ValueError("f 必须 >= 0")
    if f >= n // 2:
        raise ValueError(f"f={f} 太大，必须 < n/2={n//2}")
    if n - f - 2 < 0:
        raise ValueError(f"n-f-2={n-f-2} < 0，无法计算可信度")
    
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
    Multi-Krum 聚合：选出 m 个"最可信"梯度列表后再平均。
    grad_lists: List[List[Tensor]]
    f: 最大 Byzantine 节点数
    m: 从中选取的梯度数量，默认 m = n - f
    返回: List[Tensor]，聚合后梯度
    """
    n = len(grad_lists)
    
    # 参数验证
    if n == 0:
        raise ValueError("grad_lists 不能为空")
    if f < 0:
        raise ValueError("f 必须 >= 0")
    if f >= n // 2:
        raise ValueError(f"f={f} 太大，必须 < n/2={n//2}")
    
    if m is None:
        m = n - f
    elif m <= 0 or m > n:
        raise ValueError(f"m={m} 必须在 (0, n] 范围内")
    
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


def median(grad_lists, f):
    """
    Median 聚合：对每个参数层，选择所有Worker梯度的中位数。
    对拜占庭攻击有较强的鲁棒性。
    
    Args:
        grad_lists: List[List[Tensor]]，所有Worker的梯度列表
        f: 最大拜占庭节点数（用于参数验证）
    
    Returns:
        List[Tensor]，聚合后的梯度
    """
    n = len(grad_lists)
    
    # 参数验证
    if n == 0:
        raise ValueError("grad_lists 不能为空")
    if f < 0:
        raise ValueError("f 必须 >= 0")
    if f >= n // 2:
        raise ValueError(f"f={f} 太大，必须 < n/2={n//2}")
    
    agg = []
    num_layers = len(grad_lists[0])
    
    for layer in range(num_layers):
        # 收集该层所有Worker的梯度
        layer_grads = torch.stack([grad_lists[i][layer] for i in range(n)], dim=0)
        # 计算中位数
        median_grad = torch.median(layer_grads, dim=0)[0]
        agg.append(median_grad)
    
    return agg

def trimmed_mean(grad_lists, f):
    """
    Trimmed Mean 聚合：对每个参数层，去除最高和最低的f个值后求平均。
    
    Args:
        grad_lists: List[List[Tensor]]，所有Worker的梯度列表
        f: 最大拜占庭节点数，也是要trim的数量
    
    Returns:
        List[Tensor]，聚合后的梯度
    """
    n = len(grad_lists)
    
    # 参数验证
    if n == 0:
        raise ValueError("grad_lists 不能为空")
    if f < 0:
        raise ValueError("f 必须 >= 0")
    if f >= n // 2:
        raise ValueError(f"f={f} 太大，必须 < n/2={n//2}")
    
    agg = []
    num_layers = len(grad_lists[0])
    
    for layer in range(num_layers):
        # 收集该层所有Worker的梯度
        layer_grads = torch.stack([grad_lists[i][layer] for i in range(n)], dim=0)  # [n, ...]
        flat_grads = layer_grads.view(n, -1)  # [n, num_params]
        # 排序
        sorted_grads, _ = torch.sort(flat_grads, dim=0)
        # 去除前f和后f
        trimmed = sorted_grads[f:n-f, :]
        # 求平均
        trimmed_mean_flat = trimmed.mean(dim=0)
        trimmed_grad = trimmed_mean_flat.view_as(layer_grads[0])
        agg.append(trimmed_grad)
    
    return agg


def bulyan(grad_lists, f):
    """
    Bulyan 聚合：结合Krum和Trimmed Mean的算法。
    先用Krum选择n-2f个最可信的梯度，再用Trimmed Mean聚合。
    
    Args:
        grad_lists: List[List[Tensor]]，所有Worker的梯度列表
        f: 最大拜占庭节点数
    
    Returns:
        List[Tensor]，聚合后的梯度
    """
    n = len(grad_lists)
    
    # 参数验证
    if n == 0:
        raise ValueError("grad_lists 不能为空")
    if f < 0:
        raise ValueError("f 必须 >= 0")
    if f >= n // 2:
        raise ValueError(f"f={f} 太大，必须 < n/2={n//2}")
    if n - 2*f <= 0:
        raise ValueError(f"n-2f={n-2*f} <= 0，无法执行Bulyan")
    
    # 用Krum选择n-2f个最可信的梯度
    flats = [torch.cat([g.view(-1) for g in gl]) for gl in grad_lists]
    scores = []
    for i in range(n):
        dists = [(flats[i] - flats[j]).pow(2).sum().item()
                 for j in range(n) if j != i]
        dists.sort()
        scores.append(sum(dists[: n - f - 2]))
    
    # 选择得分最小的n-2f个索引
    selected_idxs = sorted(range(n), key=lambda i: scores[i])[:n-2*f]
    
    # 对选中的梯度用trimmed mean聚合，调整f值
    selected_grads = [grad_lists[i] for i in selected_idxs]
    # 对于选中的n-2f个梯度，使用更小的f值
    num_selected = len(selected_grads)
    adjusted_f = min(f, (num_selected // 2) - 1)
    adjusted_f = max(0, adjusted_f)
    return trimmed_mean(selected_grads, adjusted_f)


# 方便外部 import
_aggregators = {
    'krum': krum,
    'multikrum': multi_krum,
    'median': median,
    'trimmed_mean': trimmed_mean,
    'bulyan': bulyan
}

def get_aggregator(name: str):
    """获取聚合算法函数"""
    if name not in _aggregators:
        raise ValueError(f"未知的聚合算法: {name}，可用选项: {list(_aggregators.keys())}")
    return _aggregators[name]
