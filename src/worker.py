import torch
from torch.utils.data import DataLoader

class Worker:
    """
    模拟分布式系统中的一个节点（Worker）。
    每个 Worker 持有自己的一份 DataLoader，在给定的全局模型上计算一个 mini-batch 的梯度。
    """
    def __init__(self, dataset_split, batch_size=64, shuffle=True):
        """
        :param dataset_split: torch.utils.data.Dataset，Worker 自己的数据子集
        :param batch_size: int，该 Worker 每次计算梯度时使用的 batch 大小
        :param shuffle: bool，每轮是否打乱子集
        """
        self.loader = DataLoader(dataset_split, batch_size=batch_size, shuffle=shuffle)
        # 预先创建一个迭代器
        self._iterator = iter(self.loader)

    def compute_gradient(self, model, loss_fn, device):
        """
        在当前模型和损失函数下，计算一个 mini-batch 的梯度，不更新模型参数。
        :param model: torch.nn.Module，全局模型
        :param loss_fn: 损失函数实例（如 nn.CrossEntropyLoss()）
        :param device: 'cpu' 或 'cuda'
        :return: List[Tensor]，与 model.parameters() 对应的梯度列表
        """
        try:
            data, target = next(self._iterator)
        except StopIteration:
            # 如果本轮已经遍历完一次 DataLoader，就重新创建
            self._iterator = iter(self.loader)
            data, target = next(self._iterator)

        data, target = data.to(device), target.to(device)

        model.zero_grad()            # 清空模型已有梯度
        output = model(data)         # 前向
        loss = loss_fn(output, target)
        loss.backward()              # 反向，计算梯度

        # 收集所有参数的梯度，并拷贝一份返回
        grads = [param.grad.detach().clone() for param in model.parameters()]
        return grads
