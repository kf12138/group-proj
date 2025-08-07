# 联邦学习拜占庭容错实验

本实验用于对比联邦学习中的多种拜占庭容错聚合算法，支持多模型、多攻击类型、多节点配置的自动化实验与可视化分析。

## 支持内容
- **模型**：MLP（多层感知机）、ConvNet（卷积神经网络）
- **聚合算法**：average, krum, median, trimmed_mean, bulyan
- **攻击类型**：reverse, zero, random, sign_flip
- **拜占庭节点数**：f=0（无攻击）、f=1

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 一键运行全部实验
```bash
bash run_experiments.sh
```

- 自动遍历所有模型、算法、攻击类型、拜占庭节点数组合
- 日志保存在 `results/logs/`，可视化图表和分析报告在 `results/plots/`

### 3. 查看结果
- 日志文件：`results/logs/*.json`（每次实验详细结果）
- 可视化图表：`results/plots/*.png`（算法对比、训练曲线、拜占庭影响等）
- 汇总报告：`results/plots/summary_report.json`

## 主要参数说明
| 参数 | 说明 | 示例                                  |
|------|------|-------------------------------------|
| --model | 模型类型 | mlp / conv                          |
| --mode | 聚合算法 | average / krum / median / bulyan    |
| --f | 拜占庭节点数 | 0 / 1 / 2                           |
| --attack-type | 攻击类型 | reverse / zero / random / sign_flip |
| --epochs | 训练轮数 | 5（默认脚本）                             |
| --n-workers | Worker数量 | 5（默认脚本）                             |
| --batch-size | 批次大小 | 32（默认脚本）                            |
| --seed | 随机种子 | 42                                  |
| --save-log | 保存日志 | 必须加                                 |

## 目录结构
```
group-proj/
├── data/                # MNIST数据集
├── requirements.txt     # 依赖包
├── run_experiments.sh   # 一键实验脚本
├── results/
│   ├── logs/            # 日志文件
│   └── plots/           # 可视化输出
└── src/
    ├── model.py         # 模型定义
    ├── worker.py        # Worker节点
    ├── krum.py          # 聚合算法
    ├── train.py         # 主训练流程
    └── analyze_results.py # 结果分析与可视化
```

## 命令
- 单次实验：
  ```bash
  python3 src/train.py --model conv --mode krum --f 1 --attack-type reverse --epochs 5 --n-workers 5 --batch-size 32 --save-log
  ```
- 结果分析：
  ```bash
  python3 src/analyze_results.py --logs-dir results/logs --output-dir results/plots
  ```

## 相关结果
- **average** 仅在无攻击（f=0）下有效，遇攻击性能骤降
- **krum/median/trimmed_mean/bulyan** 在有攻击时表现鲁棒性差异
- **ConvNet** 模型下鲁棒性对比更明显
- 可通过 `results/plots/` 下的图表直观对比各算法表现

