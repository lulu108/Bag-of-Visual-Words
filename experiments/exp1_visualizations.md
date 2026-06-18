# 实验一：可视化图表说明

> 生成脚本：`CodeFiles/build_exp1_visualizations.py`
> 输出目录：`outputs/final_visualizations/`

---

## 图表清单

| 编号 | 文件名 | 用途 |
|------|--------|------|
| A | `k_sweep_accuracy_macro_f1.png` | K sweep 性能曲线 |
| B | `k_sweep_time.png` | K sweep 时间曲线 |
| C | `k_sweep_accuracy_time_tradeoff.png` | K sweep 性能-时间权衡散点图 |
| D | `same_classifier_metrics_bar.png` | 同分类器横向对比柱状图 |
| E | `same_classifier_time_bar.png` | 同分类器时间柱状图 |
| F | `best_config_metrics_bar.png` | 较优配置综合对比柱状图 |
| G | `best_config_time_bar.png` | 较优配置时间柱状图 |
| H | `bovw_improvement_metrics_bar.png` | BoVW 改进尝试指标对比 |
| I | `bovw_improvement_time_bar.png` | BoVW 改进尝试时间对比 |
| J | `bovw_pipeline_diagram.png` | SIFT-BoVW 流程图 |
| K | `feature_pipeline_comparison.png` | 三种特征流程对比图 |

---

## 图表详解

### A. K Sweep 性能曲线

**文件：** `outputs/final_visualizations/k_sweep_accuracy_macro_f1.png`

**报告章节建议：** 3.1 K 值选择实验 / BoVW 参数实验

**简短解释：**
随视觉词典大小 K 从 50 增大到 800，SIFT-BoVW 的分类精度总体呈上升趋势。K=50 时 accuracy 约 60.8%，K=800 达到最优 67.9%。K=100 和 K=200 出现轻微下降，可能与视觉词粒度不足导致表征不稳定有关。总体趋势表明更大的视觉词典能提供更细粒度的局部模式表达。

---

### B. K Sweep 时间曲线

**文件：** `outputs/final_visualizations/k_sweep_time.png`

**报告章节建议：** 3.1 K 值选择实验 / 计算成本分析

**简短解释：**
视觉词典大小 K 与计算时间近似呈线性增长关系。K=50 仅需约 115 秒，K=800 需要约 970 秒。时间开销主要来自 KMeans 聚类的复杂度 O(N·K·d) 和 SIFT 描述子提取。该图说明 BoVW 方法的可扩展性瓶颈在于词典规模。

---

### C. K Sweep 性能-时间权衡散点图

**文件：** `outputs/final_visualizations/k_sweep_accuracy_time_tradeoff.png`

**报告章节建议：** 3.1 K 值选择实验 / 权衡分析

**简短解释：**
散点图将各 K 值的 macro_f1 与 total_time 直接对比。K=800 位于右上角，性能最高但时间成本也最大。K=400 在性能（58.2%）与时间（400s）之间提供了折中选择。该图为实际应用中根据计算预算选择 K 值提供了参考依据。

---

### D. 同分类器横向对比柱状图

**文件：** `outputs/final_visualizations/same_classifier_metrics_bar.png`

**报告章节建议：** 3.2 传统特征对比实验 / 控制分类器变量的横向对比

**简短解释：**
在统一使用 LinearSVC 的条件下，HOG 特征以 68.1% accuracy 和 66.1% macro_f1 表现最优，Color Histogram 以 65.7% accuracy 紧随其后，SIFT-BoVW-K800 仅 56.9%。这表明在缺少超参数搜索的情况下，SIFT-BoVW 的 LinearSVC 分类性能不如 HOG 全局纹理特征和 HSV 颜色直方图。该图严格控制分类器变量，仅比较特征表达的差异。

---

### E. 同分类器时间柱状图

**文件：** `outputs/final_visualizations/same_classifier_time_bar.png`

**报告章节建议：** 3.2 传统特征对比实验 / 计算效率比较

**简短解释：**
Color Histogram 仅需约 20 秒即可完成全流程，HOG 需要约 410 秒，SIFT-BoVW-K800 需要约 1995 秒（约 33 分钟）。SIFT-BoVW 的时间开销是 HOG 的约 5 倍、Color Histogram 的约 100 倍。该图直观展示了不同特征提取方法在计算效率上的巨大差异。

---

### F. 较优配置综合对比柱状图

**文件：** `outputs/final_visualizations/best_config_metrics_bar.png`

**报告章节建议：** 3.3 综合对比与讨论

**简短解释：**
本图展示了各传统特征在当前实验中的较优配置。SIFT-BoVW 使用了 K sweep 中由 GridSearchCV SVC 选出的最优结果（K=800, accuracy=67.9%），HOG 和 Color Histogram 使用 LinearSVC。HOG（68.1%）与 SIFT-BoVW（67.9%）性能接近，但需注意两者使用了不同的分类器和超参数搜索策略，不可直接归因于特征差异。

---

### G. 较优配置时间柱状图

**文件：** `outputs/final_visualizations/best_config_time_bar.png`

**报告章节建议：** 3.3 综合对比与讨论 / 效率分析

**简短解释：**
SIFT-BoVW 最优配置（K=800 + GridSearchCV SVC）需要约 970 秒，HOG 约 410 秒，Color Histogram 约 20 秒。尽管 SIFT-BoVW 在性能上与 HOG 接近，其计算成本是 HOG 的 2.4 倍。实际部署中，若对实时性有要求，Color Histogram 和 HOG 提供了更具吸引力的效率-性能平衡。

---

### H. BoVW 改进尝试指标对比

**文件：** `outputs/final_visualizations/bovw_improvement_metrics_bar.png`

**报告章节建议：** 3.4 BoVW 改进实验

**简短解释：**
对比原始 SIFT-BoVW（67.9%）、RootSIFT+TF-IDF linear（62.7%）和 RootSIFT+TF-IDF rbf（65.1%）三种配置。RootSIFT + TF-IDF 的引入并未带来预期提升，原始版本性能最佳。可能原因包括：数据集体量较小（7 类共 210 张图像），RootSIFT 的平方根变换和 TF-IDF 加权在此条件下未能有效改善特征表达，反而可能引入了额外的稀疏性问题。

---

### I. BoVW 改进尝试时间对比

**文件：** `outputs/final_visualizations/bovw_improvement_time_bar.png`

**报告章节建议：** 3.4 BoVW 改进实验 / 效率分析

**简短解释：**
RootSIFT + TF-IDF 改进版本的计算时间（约 1680 秒）显著高于原始版本（约 970 秒），额外开销主要来自 TF-IDF 变换和 RootSIFT 的 L1 归一化加平方根运算。在性能未提升的情况下，额外的计算成本使这些改进策略在此数据集上不具备实际优势。

---

### J. SIFT-BoVW 流程图

**文件：** `outputs/final_visualizations/bovw_pipeline_diagram.png`

**报告章节建议：** 2. 方法论 / 2.1 SIFT-BoVW 方法

**简短解释：**
该流程图展示了 SIFT-BoVW 的完整处理流程：输入图像 → SIFT 关键点检测与描述子提取 → KMeans 构建视觉词汇词典 → 对每张图像编码 BoVW 频率直方图 → StandardScaler 标准化 → SVM 分类器 → 预测类别。该图用于方法论章节，帮助读者快速理解 BoVW 的处理步骤。

---

### K. 三种特征流程对比图

**文件：** `outputs/final_visualizations/feature_pipeline_comparison.png`

**报告章节建议：** 2. 方法论 / 2.2 传统特征方法概述

**简短解释：**
横向三栏对比图展示了 Color Histogram、HOG 和 SIFT-BoVW 三者的处理流程差异。Color Histogram 最简单（HSV 色彩空间统计），HOG 记录梯度方向分布（纹理/边缘），SIFT-BoVW 最复杂（局部关键点 → 视觉词典 → 频率直方图）。该图用于方法论开头，直观说明三种方法在特征表达上的本质区别。

---

## 使用命令

```bash
python CodeFiles/build_exp1_visualizations.py \
    --k_sweep_csv outputs/summary_results.csv \
    --same_classifier_csv outputs/final_tables/final_same_classifier_comparison.csv \
    --best_config_csv outputs/final_tables/final_best_config_comparison.csv \
    --improvement_csv outputs/final_tables/final_bovw_improvement_comparison.csv \
    --output_dir outputs/final_visualizations
```
