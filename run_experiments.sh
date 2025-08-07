#!/bin/bash

# 完备的联邦学习拜占庭容错实验脚本
# 使用方法: bash run_experiments.sh

set -e

echo "开始联邦学习拜占庭容错实验..."

# 创建结果目录
mkdir -p results
mkdir -p results/logs
mkdir -p results/plots

EPOCHS=10
N_WORKERS=10
BATCH_SIZE=32
SEED=42


# reverse攻击类型下（epoch轮次小时也能有较明显效果），各聚合算法在conv和mlp模型训练时的鲁棒性表现（f=0/1/2）
for model in mlp conv; do
  for mode in average krum median trimmed_mean bulyan; do
    for f in 0 1 2 3; do
      for attack in reverse; do # 其他可选攻击类型有：zero random sign_flip
        echo "=== $model $mode f=$f attack=$attack ==="
        python3 src/train.py --model $model --mode $mode --f $f --epochs $EPOCHS --n-workers $N_WORKERS --batch-size $BATCH_SIZE --attack-type $attack --seed $SEED --save-log
      done
    done
  done
done



# 生成结果分析
echo "=== 生成结果分析 ==="
python3 src/analyze_results.py --logs-dir results/logs --output-dir results/plots

echo "实验和分析完成！结果保存在 results/ 目录中"

