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

## 实验结果

|   K | Accuracy | 备注                 |
| --: | -------: | -------------------- |
|  50 |     待填 | 视觉词较少           |
| 100 |    0.665 | baseline             |
| 200 |     待填 | 词典更细             |
| 400 |     待填 | 表达能力更强但更慢   |
| 800 |     待填 | 可能更稀疏，耗时更高 |

K 值过小时，视觉词典表达能力不足，不同局部结构可能被合并到同一个视觉词中；K 值增大后，局部模式表达更细致，分类性能可能提升。但 K 值过大时，BoVW 直方图维度升高、特征变稀疏，KMeans 聚类和 SVM 训练时间也会增加。
