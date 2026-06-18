# 实验一：统一 SVC + GridSearchCV 的传统特征对比实验

## 实验目的

在 **统一 SVC + GridSearchCV 训练协议**下横向对比 Color Histogram、HOG 和 SIFT-BoVW-K800 三种传统图像分类特征，使实验结论比单纯 LinearSVC 对比更严谨、更具可重复性。

## 为什么需要统一 SVC + GridSearchCV

之前的 `compare_traditional_features.py` 使用 `LinearSVC(class_weight='balanced')` 作为分类器，其正则化参数 C 使用默认值（1.0）。这存在两个问题：

1. **不同特征对 C 的敏感度不同。** Color Histogram（512 维 L2 归一化向量）、HOG（高维梯度直方图）和 SIFT-BoVW（800 维计数直方图）的特征分布差异很大，固定 C=1.0 可能使某些特征处于欠拟合或过拟合状态。
2. **无法公平比较特征表达能力。** 如果一种特征在默认 C 下表现不佳，可能只是超参不合适，不一定代表特征表达能力弱。

因此本实验改用 `SVC(kernel='linear', class_weight='balanced')` + `GridSearchCV(param_grid={'C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='f1_macro')`，每种特征独立搜索最优 C，在各自最优超参数下比较分类性能。

## 为什么每种特征要独立搜索 C

- **不能跨特征复用 C。** 如果 SIFT-BoVW 搜索出的 best C=10.0，不能直接用于 Color Histogram 或 HOG。不同特征的正则化需求不同。
- **信息泄漏风险。** 如果使用某特征在 GridSearchCV 中选出的 C 直接套用到另一个特征，相当于间接利用了测试集信息（因为 GridSearchCV 的 C 选择依赖于交叉验证结果）。
- **独立搜索保证公平。** 每种特征都在同等条件下（相同的 param_grid、cv、scoring）独立完成超参数优化，对比的是"各特征在自身最优配置下的表现"。

## 与 LinearSVC 对比实验的区别

| 维度 | LinearSVC 对比实验 | SVC + GridSearchCV 对比实验 |
|------|-------------------|---------------------------|
| 脚本 | `compare_traditional_features.py` | `compare_traditional_features_svc.py` |
| 输出目录 | `outputs/feature_compare_k800/` | `outputs/feature_compare_svc_k800/` |
| 分类器 | `LinearSVC(C=1.0)` | `SVC + GridSearchCV(C=[0.01,...,100])` |
| 控制变量 | 特征表达（控制分类器一致） | 特征+最优超参（控制搜索协议一致） |
| 结论粒度 | 特征表达能力对比 | 特征+超参联合优化后的对比 |

## 运行命令

```bash
python CodeFiles/compare_traditional_features_svc.py \
    --train_path dataset/train \
    --test_path dataset/test \
    --image_size 150 \
    --bovw_clusters 800 \
    --output_dir outputs/feature_compare_svc_k800 \
    --cv 5
```

## 结果表格

见 `outputs/feature_compare_svc_k800/summary_feature_compare_svc.csv`。

| method | accuracy | macro_f1 | best_C | total_time |
|--------|----------|----------|--------|------------|
| Color Histogram (SVC) | 0.690476 | 0.688484 | 100 | 45.9s |
| HOG (SVC) | 0.666667 | 0.646149 | 0.1 | 2891.7s |
| SIFT-BoVW K=800 (SVC) | 0.600000 | 0.583553 | 0.01 | 2984.6s |

（完整数据见 `outputs/feature_compare_svc_k800/summary_feature_compare_svc.csv`。）

## 图表

| 文件 | 说明 |
|------|------|
| `outputs/feature_compare_svc_k800/svc_grid_metrics_bar.png` | 三种方法的 accuracy 和 macro_f1 柱状图 |
| `outputs/feature_compare_svc_k800/svc_grid_time_bar.png` | 三种方法的 total_time 柱状图 |
| `outputs/feature_compare_svc_k800/svc_grid_best_c_bar.png` | 三种方法 GridSearchCV 选出的 best C（对数坐标） |

## 可写入报告的分析模板

### 实验设置

> 为消除分类器超参数对特征对比的干扰，我们在统一 SVC + GridSearchCV 协议下重新评估了三种传统特征。每种特征独立搜索最优正则化参数 C ∈ {0.01, 0.1, 1, 10, 100}，使用 5 折交叉验证和 macro_f1 作为评分标准。

### 结果分析

> 在统一 SVC + GridSearchCV 协议下，Color Histogram 以 accuracy=69.05%（macro_f1=68.85%）表现最优，HOG 为 66.67%（64.61%），SIFT-BoVW-K800 为 60.00%（58.36%）。各特征的 best C 值差异巨大：Color Histogram C=100，HOG C=0.1，SIFT-BoVW C=0.01。同一参数 C 跨越四个数量级（0.01 → 100），说明不同特征对正则化强度的最优需求确实存在显著差异，有力验证了独立搜索 C 的必要性。

### 与 LinearSVC 对比的讨论

> 相比 LinearSVC 固定 C=1.0 的结果，SVC + GridSearchCV 协议下各特征表现差异明显：
> - Color Histogram: 65.7% → **69.0%（+3.3%）** —— C=100 远大于默认 1.0，说明该特征需要弱正则化；
> - HOG: 68.1% → **66.7%（-1.4%）** —— C=0.1 小于默认 1.0，GridSearch 反而略降，说明默认 C 对 HOG 尚可；
> - SIFT-BoVW: 56.9% → **60.0%（+3.1%）** —— C=0.01 远小于默认 1.0，说明需要极强正则化抑制稀疏直方图的过拟合。
>
> 这一结果说明：(1) 统一 GridSearchCV 协议确实能发现更优的超参数；(2) 不同特征的最优 C 跨四个数量级，验证了独立搜索的必要性；(3) Color Histogram 在统一协议下成为最优方法，挑战了 HOG 在 LinearSVC 下的领先结论。

---

## 生成脚本

- `CodeFiles/compare_traditional_features_svc.py`
