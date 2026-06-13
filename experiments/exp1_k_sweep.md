# Exp1 K Sweep: SIFT-BoVW + Linear SVM

## 实验目的

比较不同视觉词典大小 K 对 SIFT-BoVW 图像分类性能的影响。

## 实验设置

- Method: SIFT-BoVW + SVM
- Kernel: linear
- K values: 50, 100, 200, 400, 800
- Train path: dataset/train
- Test path: dataset/test

## 运行命令

```bash
python CodeFiles/BoW.py --train_path dataset/train --test_path dataset/test --no_clusters 50 --kernel_type linear
python CodeFiles/BoW.py --train_path dataset/train --test_path dataset/test --no_clusters 100 --kernel_type linear
python CodeFiles/BoW.py --train_path dataset/train --test_path dataset/test --no_clusters 200 --kernel_type linear
python CodeFiles/BoW.py --train_path dataset/train --test_path dataset/test --no_clusters 400 --kernel_type linear
python CodeFiles/BoW.py --train_path dataset/train --test_path dataset/test --no_clusters 800 --kernel_type linear
```

# 可复现 K 值实验结果

| K | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Train Time/s | Test Time/s | Total Time/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.607656 | 0.612004 | 0.607882 | 0.606137 | 0.605790 | 108.844361 | 5.998237 | 114.842597 |
| 100 | 0.559809 | 0.567913 | 0.559770 | 0.544102 | 0.543703 | 129.401166 | 3.389270 | 132.790436 |
| 200 | 0.574163 | 0.588258 | 0.574220 | 0.561503 | 0.561000 | 206.242138 | 4.004148 | 210.246286 |
| 400 | 0.598086 | 0.624985 | 0.597537 | 0.582039 | 0.582144 | 397.014748 | 3.213588 | 400.228336 |
| 800 | 0.679426 | 0.722068 | 0.679639 | 0.661117 | 0.660489 | 958.055946 | 11.550493 | 969.606439 |

本轮实验在固定随机种子 RANDOM_STATE=42 的条件下进行。结果表明，K=800 取得最高 Accuracy 和 Macro-F1，说明更大的视觉词典能够提供更细粒度的局部模式表达。但同时 K=800 的训练时间显著增加，体现出 BoVW 方法在词典规模增大时的计算开销问题。

K 值过小时，视觉词典表达能力不足，不同局部结构可能被合并到同一个视觉词中；K 值增大后，局部模式表达更细致，分类性能可能提升。但 K 值过大时，BoVW 直方图维度升高、特征变稀疏，KMeans 聚类和 SVM 训练时间也会增加。
