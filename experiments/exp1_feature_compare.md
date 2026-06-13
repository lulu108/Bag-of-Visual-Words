# Exp1 Feature Compare: Color Histogram vs HOG vs SIFT-BoVW

## 实验目的

在同一训练集、测试集和分类器设置下，对比三类传统图像特征的分类效果与运行开销：

- Color Histogram
- HOG
- SIFT-BoVW

本实验用于回答：在当前七类场景/物体图像分类任务中，颜色分布、梯度纹理和局部关键点词袋特征各自能提供多少判别能力，以及它们在特征提取和模型训练时间上的差异。

## 三种特征方法说明

### Color Histogram

- 输入图像 resize 到 `image_size x image_size`。
- 将 BGR 图像转换到 HSV 空间。
- 提取 3D HSV histogram，默认 bins 为 `(8, 8, 8)`。
- 将 histogram 展平并做 L2 归一化，作为整张图像的颜色分布特征。

### HOG

- 输入图像 resize 到 `image_size x image_size`。
- 将图像转换为灰度图。
- 使用 `skimage.feature.hog` 提取梯度方向直方图特征。
- 参数固定为：
  - `orientations=9`
  - `pixels_per_cell=(8,8)`
  - `cells_per_block=(2,2)`
  - `block_norm="L2-Hys"`

### SIFT-BoVW

- 输入图像 resize 到 `image_size x image_size`。
- 使用 SIFT 提取局部 descriptors。
- 默认视觉词典大小 `K=400`。
- 使用 `KMeans(random_state=42, n_init=10)` 在训练集 descriptors 上构建视觉词典。
- 将每张图编码为 BoVW histogram。
- 使用训练集统计量 `StandardScaler` 标准化训练集和测试集特征。

三种特征均使用同一种分类器：

- Linear SVM
- `class_weight="balanced"`
- `random_state=42`

## 运行命令

```bash
python CodeFiles/compare_traditional_features.py --train_path dataset/train --test_path dataset/test --image_size 150 --output_dir outputs/feature_compare
```

也可以使用默认参数直接运行：

```bash
python CodeFiles/compare_traditional_features.py
```

## 输出文件

所有输出默认保存到：

```text
outputs/feature_compare
```

汇总结果：

```text
outputs/feature_compare/summary_feature_compare.csv
```

单方法结果：

```text
outputs/feature_compare/color_histogram_result.txt
outputs/feature_compare/hog_result.txt
outputs/feature_compare/sift_bovw_k400_result.txt
```

混淆矩阵：

```text
outputs/feature_compare/color_histogram_confusion_matrix.png
outputs/feature_compare/hog_confusion_matrix.png
outputs/feature_compare/sift_bovw_k400_confusion_matrix.png
```

## 待填写结果表格

| Method | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Feature Extract Time/s | Train Time/s | Test Time/s | Total Time/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Color Histogram |  |  |  |  |  |  |  |  |  |
| HOG |  |  |  |  |  |  |  |  |  |
| SIFT-BoVW K=400 |  |  |  |  |  |  |  |  |  |

## 结果检查

运行后可以用以下命令查看汇总结果：

```bash
cat outputs/feature_compare/summary_feature_compare.csv
```
