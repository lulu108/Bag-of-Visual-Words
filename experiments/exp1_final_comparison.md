# 实验一：最终实验结果汇总

> 生成时间：2026-06-17 12:04:45
> 选择指标：macro_f1（平局时比较 accuracy）

## 结果组织方式

实验一包含三条子实验线，各自输出中间 CSV 结果，最终由 `CodeFiles/build_final_exp1_tables.py` 读取并生成以下三张对比表：

1. **同分类器横向对比表**（表 1）：
   Color Histogram、HOG、SIFT-BoVW-K800 三者统一使用 LinearSVC，
   用于控制分类器一致的横向对比。
2. **较优配置综合对比表**（表 2）：
   各传统特征在实验中的较优配置 — Color Histogram 和 HOG 来自
   feature_compare 脚本，SIFT-BoVW 使用 K sweep 中的最优结果。
3. **BoVW 改进尝试对比表**（表 3）：
   对比原始 SIFT-BoVW 与 RootSIFT+TF-IDF 改进版本的性能差异。

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
| Color Histogram | 3D HSV (8×8×8) + LinearSVC | 0.657143 | 0.657716 | 0.657143 | 0.647602 | 0.647602 | 18.72 | 来自 compare_traditional_features.py；使用 LinearSVC，仅作为基线参考。 |
| HOG | HOG (orient=9, ppc=8×8, cpb=2×2) + LinearSVC | 0.680952 | 0.714059 | 0.680952 | 0.660885 | 0.660885 | 216.01 | 来自 compare_traditional_features.py；使用 LinearSVC，仅作为纹理特征基线。 |
| SIFT-BoVW-K800 | SIFT-BoVW K=800 + LinearSVC | 0.598086 | 0.611642 | 0.597865 | 0.572688 | 0.572788 | 997.20 | 来自 compare_traditional_features.py；使用 LinearSVC，与 Color Histogram/HOG 统一分类器。 |

## 表 2：较优配置综合对比

**注意：本表不是严格控制同一分类器的实验。**它表示各传统特征在当前实验中的较优配置。
SIFT-BoVW 使用 K sweep 中的最优结果（K=800, kernel=linear, SVC pipeline），而 Color Histogram 和 HOG 来自 feature_compare 脚本（LinearSVC）。
因此本表仅用于概括性对比，不可直接归因于单一变量。

| method | setting | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | total_time | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Color Histogram | 3D HSV (8×8×8) + LinearSVC | 0.657143 | 0.657716 | 0.657143 | 0.647602 | 0.647602 | 18.72 | 来自 compare_traditional_features.py（LinearSVC）；本表不是严格控制同一分类器。 |
| HOG | HOG (orient=9, ppc=8×8, cpb=2×2) + LinearSVC | 0.680952 | 0.714059 | 0.680952 | 0.660885 | 0.660885 | 216.01 | 来自 compare_traditional_features.py（LinearSVC）；本表不是严格控制同一分类器。 |
| SIFT-BoVW (最优 K) | K=800, kernel=linear, SVC pipeline | 0.679426 | 0.722068 | 0.679639 | 0.661117 | 0.660489 | 969.61 | 来自 K sweep 最优结果（select_by=macro_f1），不是 feature_compare 中的 LinearSVC 结果；使用 GridSearchCV 搜索 SVC 最优超参数。 |

## 表 3：BoVW 改进尝试对比

对比原始 SIFT-BoVW（无 RootSIFT/TF-IDF）与 RootSIFT + TF-IDF 改进版本的性能。
改进实验由 `CodeFiles/BoW_improved.py` 完成。

| method | setting | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | total_time | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original SIFT-BoVW (最优 K) | K=800, kernel=linear, SVC pipeline | 0.679426 | 0.722068 | 0.679639 | 0.661117 | 0.660489 | 969.61 | 原始 SIFT-BoVW（无 RootSIFT/TF-IDF）；来自 K sweep 最优结果。 |
| Improved BoVW (linear + RootSIFT + TF-IDF) | K=800, kernel=linear, RootSIFT + TF-IDF, SVC pipeline | 0.626794 | 0.709823 | 0.627094 | 0.612165 | 0.611550 | 1060.21 | 来自 BoW_improved.py；RootSIFT + TF-IDF 旨在改善特征表达和视觉词权重。 |
| Improved BoVW (rbf + RootSIFT + TF-IDF) | K=800, kernel=rbf, RootSIFT + TF-IDF, SVC pipeline | 0.650718 | 0.720710 | 0.650739 | 0.632897 | 0.632495 | 1681.48 | 来自 BoW_improved.py；RBF kernel 可捕捉非线性决策边界。 |
