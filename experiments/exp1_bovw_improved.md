# Exp1 Improved BoVW: RootSIFT + TF-IDF

## 实验目的

在第一轮可复现 K sweep 的 SIFT-BoVW 基线基础上，测试两个常见的传统特征改进策略是否能提升图像分类性能：

- RootSIFT：改进局部 SIFT descriptor 的度量性质。
- TF-IDF BoVW：降低高频视觉词的权重，突出更有区分度的视觉词。

本实验不覆盖原始 `CodeFiles/BoW.py`，改进流程放在独立脚本 `CodeFiles/BoW_improved.py` 中，便于和 baseline 结果对照。

## RootSIFT 原理

RootSIFT 仍然先使用 SIFT 提取原始 128 维 descriptor，然后对每个 descriptor 做两步变换：

1. L1 normalization：让每个 descriptor 的元素和归一化到 1。
2. sqrt 变换：对归一化后的每个维度取平方根。

该变换近似将欧氏距离用于 Hellinger kernel 空间，常用于提升 SIFT matching 和 BoVW 表达质量。若某张图像没有 SIFT descriptors，则脚本会安全跳过该图像，并同步跳过对应 label。

## TF-IDF BoVW 原理

普通 BoVW 使用 visual word 的 count histogram 表示图像。TF-IDF BoVW 在 count histogram 上进一步加权：

- TF：图像内部某个 visual word 的出现频率。
- IDF：该 visual word 在整个训练集中的稀有程度。

训练阶段：

```text
train count BoVW -> TfidfTransformer.fit_transform
```

测试阶段：

```text
test count BoVW -> TfidfTransformer.transform
```

测试集严禁重新 `fit`，否则会引入测试集统计信息，造成信息泄漏。

## SVM 参数搜索设置

分类器使用 `SVC(class_weight="balanced")`，并通过 `GridSearchCV(scoring="f1_macro", cv=5)` 搜索参数。

Linear kernel：

```text
C = [0.01, 0.1, 1, 10, 100]
```

RBF kernel：

```text
C = [0.1, 1, 10, 100]
gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1]
```

全局随机种子固定为：

```text
RANDOM_STATE = 42
```

## 运行命令

RootSIFT + TF-IDF + linear SVM，K=800：

```bash
python CodeFiles/BoW_improved.py --train_path dataset/train --test_path dataset/test --no_clusters 800 --kernel_type linear --use_rootsift --use_tfidf --output_dir outputs/bovw_improved
```

RootSIFT + TF-IDF + RBF SVM，K=800：

```bash
python CodeFiles/BoW_improved.py --train_path dataset/train --test_path dataset/test --no_clusters 800 --kernel_type rbf --use_rootsift --use_tfidf --output_dir outputs/bovw_improved
```

只使用 RootSIFT，不使用 TF-IDF：

```bash
python CodeFiles/BoW_improved.py --train_path dataset/train --test_path dataset/test --no_clusters 800 --kernel_type linear --use_rootsift --output_dir outputs/bovw_improved
```

只使用 TF-IDF，不使用 RootSIFT：

```bash
python CodeFiles/BoW_improved.py --train_path dataset/train --test_path dataset/test --no_clusters 800 --kernel_type linear --use_tfidf --output_dir outputs/bovw_improved
```

## 输出文件

默认输出目录：

```text
outputs/bovw_improved
```

每次运行追加汇总：

```text
outputs/bovw_improved/summary_improved.csv
```

单次运行还会保存：

```text
outputs/bovw_improved/<method>_result.txt
outputs/bovw_improved/<method>_confusion_matrix.png
```

其中 `<method>` 会包含 K 值、kernel、是否使用 RootSIFT、是否使用 TF-IDF。

## 待填写结果表格

| K | Kernel | RootSIFT | TF-IDF | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Best Params | Train Time/s | Test Time/s | Total Time/s |
|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 800 | linear | Yes | Yes |  |  |  |  |  |  |  |  |  |
| 800 | rbf | Yes | Yes |  |  |  |  |  |  |  |  |  |
| 800 | linear | Yes | No |  |  |  |  |  |  |  |  |  |
| 800 | linear | No | Yes |  |  |  |  |  |  |  |  |  |

## 结果检查

```bash
cat outputs/bovw_improved/summary_improved.csv
```
