# 实验一：最终实验结果汇总

> 生成时间：2026-06-18 20:20:06
> 选择指标：macro_f1（平局时比较 accuracy）

## 结果组织方式

实验一包含多条子实验线，各自输出中间 CSV 结果，最终由 `CodeFiles/build_final_exp1_tables.py` 读取并生成以下对比表：

1. **同分类器横向对比表**（表 1）：
   Color Histogram、HOG、SIFT-BoVW-K800 三者统一使用 LinearSVC，
   用于控制分类器一致的横向对比。
2. **较优配置综合对比表**（表 2）：
   各传统特征在实验中的较优配置 — Color Histogram 和 HOG 来自
   feature_compare 脚本，SIFT-BoVW 使用 K sweep 中的最优结果。
3. **BoVW 改进尝试对比表**（表 3）：
   对比原始 SIFT-BoVW 与 RootSIFT+TF-IDF 改进版本的性能差异。
4. **统一 SVC + GridSearchCV 横向对比表**（表 4）：
   在统一 SVC + GridSearchCV 协议下重新对比三种传统特征，
   每种特征独立搜索最优 C，比表 1 的 LinearSVC 对比更公平。

## SIFT-BoVW 最优结果选取说明

SIFT-BoVW 的最优结果来自 **K sweep 实验**（`outputs/summary_results.csv`），
按 `macro_f1` 选择 K=800, kernel=linear 的配置，accuracy=0.679426, macro_f1=0.661117。

**为什么不直接用 feature_compare 中的 `sift_bovw_k800`？**

feature_compare 中的 `sift_bovw_k800` 固定使用 **LinearSVC**（无 GridSearchCV），其目的是与 Color Histogram、HOG **统一分类器**进行横向对比。
而 K sweep 中的 SIFT-BoVW 使用 **GridSearchCV 搜索 SVC 最优超参数**，因此 K sweep 中的线性核 SVC 结果（经过 C 参数搜索）可能优于 feature_compare 中的 LinearSVC 结果。
为保证表 1 仅比较特征表达，三方法统一用 LinearSVC；为展示 BoVW 在实验一中的最优性能，表 2 和表 3 采用 K sweep 的最优结果。

## 表 1：同分类器横向对比

Color Histogram、HOG、SIFT-BoVW-K800 均来自 `compare_traditional_features.py`，统一使用 **LinearSVC**（`class_weight='balanced'`）。
该表用于控制分类器一致的横向对比。

| method | setting | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | total_time | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Color Histogram | 3D HSV (8×8×8) + LinearSVC | 0.657143 | 0.657716 | 0.657143 | 0.647602 | 0.647602 | 19.51 | 来自 compare_traditional_features.py；使用 LinearSVC，仅作为基线参考。 |
| HOG | HOG (orient=9, ppc=8×8, cpb=2×2) + LinearSVC | 0.680952 | 0.714059 | 0.680952 | 0.660885 | 0.660885 | 409.64 | 来自 compare_traditional_features.py；使用 LinearSVC，仅作为纹理特征基线。 |
| SIFT-BoVW-K800 | SIFT-BoVW K=800 + LinearSVC | 0.569378 | 0.563223 | 0.569622 | 0.547713 | 0.547510 | 1994.88 | 来自 compare_traditional_features.py；使用 LinearSVC，与 Color Histogram/HOG 统一分类器。 |

## 表 2：较优配置综合对比

**注意：本表不是严格控制同一分类器的实验。**它表示各传统特征在当前实验中的较优配置。
SIFT-BoVW 使用 K sweep 中的最优结果（K=800, kernel=linear, SVC pipeline），而 Color Histogram 和 HOG 来自 feature_compare 脚本（LinearSVC）。
因此本表仅用于概括性对比，不可直接归因于单一变量。

| method | setting | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | total_time | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Color Histogram | 3D HSV (8×8×8) + LinearSVC | 0.657143 | 0.657716 | 0.657143 | 0.647602 | 0.647602 | 19.51 | 来自 compare_traditional_features.py（LinearSVC）；本表不是严格控制同一分类器。 |
| HOG | HOG (orient=9, ppc=8×8, cpb=2×2) + LinearSVC | 0.680952 | 0.714059 | 0.680952 | 0.660885 | 0.660885 | 409.64 | 来自 compare_traditional_features.py（LinearSVC）；本表不是严格控制同一分类器。 |
| SIFT-BoVW (最优 K) | K=800, kernel=linear, SVC pipeline | 0.679426 | 0.722068 | 0.679639 | 0.661117 | 0.660489 | 969.61 | 来自 K sweep 最优结果（select_by=macro_f1），不是 feature_compare 中的 LinearSVC 结果；使用 GridSearchCV 搜索 SVC 最优超参数。 |

## 表 3：BoVW 改进尝试对比

对比原始 SIFT-BoVW（无 RootSIFT/TF-IDF）与 RootSIFT + TF-IDF 改进版本的性能。
改进实验由 `CodeFiles/BoW_improved.py` 完成。

| method | setting | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | total_time | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original SIFT-BoVW (最优 K) | K=800, kernel=linear, SVC pipeline | 0.679426 | 0.722068 | 0.679639 | 0.661117 | 0.660489 | 969.61 | 原始 SIFT-BoVW（无 RootSIFT/TF-IDF）；来自 K sweep 最优结果。 |
| Improved BoVW (linear + RootSIFT + TF-IDF) | K=800, kernel=linear, RootSIFT + TF-IDF, SVC pipeline | 0.626794 | 0.709823 | 0.627094 | 0.612165 | 0.611550 | 1685.91 | 来自 BoW_improved.py；RootSIFT + TF-IDF 旨在改善特征表达和视觉词权重。 |
| Improved BoVW (rbf + RootSIFT + TF-IDF) | K=800, kernel=rbf, RootSIFT + TF-IDF, SVC pipeline | 0.650718 | 0.720710 | 0.650739 | 0.632897 | 0.632495 | 1681.48 | 来自 BoW_improved.py；RBF kernel 可捕捉非线性决策边界。 |

## 表 4：统一 SVC + GridSearchCV 横向对比

Color Histogram、HOG、SIFT-BoVW-K800 均来自 `compare_traditional_features_svc.py`，三者统一使用 **SVC(kernel='linear', class_weight='balanced') + GridSearchCV(param_grid={'C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='f1_macro')**。

**与表 1（LinearSVC 对比）的区别：**

- 表 1 使用 `LinearSVC(C=1.0, class_weight='balanced')`，所有方法共享同一个固定超参数；
- 表 4 每种特征**独立运行 GridSearchCV**，在相同的 param_grid 和 cv 设定下搜索各自的最优 C；
- 表 1 控制的是「分类器完全一致」，表 4 控制的是「超参数搜索协议一致」；
- 表 4 比表 1 对超参数的选择更鲁棒，因此更能反映特征表达能力的上限。

**为什么三种特征需要独立搜索 C？**

不同特征的正则化需求差异巨大。实验结果显示：Color Histogram 的最优 C=100（弱正则化，512 维 L2 归一化向量），HOG 的最优 C=0.1（较强正则化，10,404 维高维特征），SIFT-BoVW 的最优 C=0.01（极强正则化，800 维稀疏计数直方图）。同一 C 值跨四个数量级，若强行统一 C 则无法公平比较。

**与表 2（较优配置综合对比）的关系：**

表 2 混合了不同分类器协议（Color Histogram/HOG 用 LinearSVC，BoVW 用 SVC），不是严格控制变量实验；表 4 三种特征使用**完全相同的训练协议**（SVC + GridSearchCV），是实验一中最接近「统一分类器调参协议」的公平比较。

| method | setting | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | best_params | total_time | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Color Histogram | 3D HSV (8×8×8) + SVC + GridSearchCV | 0.690476 | 0.720083 | 0.690476 | 0.688484 | 0.688484 | C=100 | 45.86 | 来自 compare_traditional_features_svc.py；独立 GridSearchCV，best C=100；比 LinearSVC 固定 C=1.0 更公平。 |
| HOG | HOG (orient=9, ppc=8×8, cpb=2×2) + SVC + GridSearchCV | 0.666667 | 0.688015 | 0.666667 | 0.646149 | 0.646149 | C=0.1 | 2891.69 | 来自 compare_traditional_features_svc.py；独立 GridSearchCV，best C=0.1；高维特征需要强正则化。 |
| SIFT-BoVW-K800 | SIFT-BoVW K=800 + SVC + GridSearchCV | 0.600000 | 0.639335 | 0.600000 | 0.583553 | 0.583553 | C=0.01 | 2984.61 | 来自 compare_traditional_features_svc.py；独立 GridSearchCV，best C=0.01；稀疏直方图需要极强正则化。 |
