#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 SVC + GridSearchCV 的传统特征对比实验。

目标：
  在同一 SVC + GridSearchCV 训练协议下比较 Color Histogram、HOG、SIFT-BoVW-K800
  三种传统特征，使实验比单纯 LinearSVC 对比更严谨。

关键设计：
  - 每种特征独立运行 GridSearchCV（不能跨特征复用搜索到的 C）；
  - KMeans / StandardScaler / TF-IDF 等只在训练集 fit，测试集只 transform；
  - 无 SIFT descriptors 的图像用全零 BoVW histogram 表示，避免样本数量不一致；
  - 数据读取使用 sorted() 保证标签顺序稳定。

输出：
  - CSV 汇总表
  - 每种方法的 result txt、confusion matrix png、normalized confusion matrix png
  - 三张 svc_grid 对比图表
  - 实验文档 experiments/exp1_feature_compare_svc.md
"""

import argparse
import csv
import os
import sys
import time
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
IMAGE_SIZE = 150
CLASSIFIER_MAX_ITER = 10000
COLOR_HIST_BINS = (8, 8, 8)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
SVM_PARAM_GRID = {"C": [0.01, 0.1, 1, 10, 100]}
METRIC_FIELDS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
SUMMARY_FIELDS = [
    "method",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "best_C",
    "feature_extract_time",
    "train_time",
    "test_time",
    "total_time",
]
DPI = 300

# ---------------------------------------------------------------------------
# 中文字体
# ---------------------------------------------------------------------------
def _setup_cn():
    import matplotlib.font_manager as fm
    for name in ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None
FONT_NAME = _setup_cn()


# ===================================================================
# 数据读取
# ===================================================================
def get_image_paths_and_labels(root_path):
    """按类别和文件名排序，返回稳定的路径、整数标签和类别名。"""
    image_paths = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ])
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(root_path, class_name)
        for filename in sorted(os.listdir(class_dir)):
            image_paths.append(os.path.join(class_dir, filename))
            labels.append(class_index)
    return image_paths, labels, class_names


# ===================================================================
# 图像读取
# ===================================================================
def read_color_image(image_path, image_size=IMAGE_SIZE):
    """读取彩色图像并 resize。"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.resize(image, (image_size, image_size))


def read_gray_image(image_path, image_size=IMAGE_SIZE):
    """读取灰度图像并 resize。"""
    image = cv2.imread(image_path, 0)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.resize(image, (image_size, image_size))


# ===================================================================
# 特征提取
# ===================================================================
def l2_normalize(feature, eps=1e-10):
    """L2 归一化。"""
    feature = feature.astype(np.float32, copy=False)
    norm = np.linalg.norm(feature)
    if norm < eps:
        return feature
    return feature / norm


def extract_color_histogram(image):
    """3D HSV Color Histogram + L2 归一化。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, COLOR_HIST_BINS,
                        [0, 180, 0, 256, 0, 256])
    return l2_normalize(hist.flatten())


def extract_hog_feature(gray_image):
    """HOG 纹理/边缘特征。"""
    feature = hog(
        gray_image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )
    return feature.astype(np.float32)


def extract_feature_matrix(image_paths, extractor, image_size=IMAGE_SIZE):
    """遍历图像路径列表，提取特征矩阵 [N, D] 并计时。"""
    start = time.time()
    features = []
    for p in image_paths:
        if extractor.__name__ == "extract_color_histogram":
            img = read_color_image(p, image_size)
        else:
            img = read_gray_image(p, image_size)
        features.append(extractor(img))
    elapsed = time.time() - start
    return np.asarray(features, dtype=np.float32), elapsed


def extract_sift_descriptors_per_image(image_paths, image_size=IMAGE_SIZE):
    """
    返回 list of ndarray or None。
    None 表示该图像无有效 SIFT descriptors。
    """
    sift = cv2.SIFT_create()
    descriptors_per_image = []
    for p in image_paths:
        gray = read_gray_image(p, image_size)
        _, des = sift.detectAndCompute(gray, None)
        descriptors_per_image.append(des if des is not None else None)
    return descriptors_per_image


def build_vocabulary(descriptors_per_image, no_clusters):
    """在有效 SIFT descriptors 上训练 KMeans 视觉词典。"""
    valid_descriptors = [d for d in descriptors_per_image if d is not None]
    if not valid_descriptors:
        raise ValueError("No valid SIFT descriptors in training set.")
    stacked = np.vstack(valid_descriptors)
    kmeans = KMeans(n_clusters=no_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(stacked)
    return kmeans


def encode_bovw_histograms(kmeans, descriptors_per_image, no_clusters):
    """
    将每张图编码为 BoVW count histogram。
    descriptors 为 None 的图像用全零向量表示，保持样本数量一致。
    """
    n_images = len(descriptors_per_image)
    features = np.zeros((n_images, no_clusters), dtype=np.float32)
    for i, des in enumerate(descriptors_per_image):
        if des is not None and len(des) > 0:
            words = kmeans.predict(des)
            features[i], _ = np.histogram(words, bins=np.arange(no_clusters + 1))
    return features


# ===================================================================
# 训练协议：SVC + GridSearchCV
# ===================================================================
def train_with_gridsearchcv(train_features, train_labels, cv_folds):
    """
    统一训练协议：SVC(kernel=linear, class_weight=balanced) + GridSearchCV.
    返回 best_estimator、best_params 和 train_time。
    """
    svm = SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=CLASSIFIER_MAX_ITER,
    )
    grid = GridSearchCV(
        svm,
        SVM_PARAM_GRID,
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1,
    )
    t0 = time.time()
    grid.fit(train_features, train_labels)
    train_time = time.time() - t0
    return grid.best_estimator_, grid.best_params_, train_time


# ===================================================================
# 指标计算
# ===================================================================
def calculate_metrics(true_labels, predictions):
    """计算所有分类指标。"""
    acc = accuracy_score(true_labels, predictions)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )
    wf1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "weighted_f1": wf1,
    }


# ===================================================================
# 混淆矩阵与结果输出
# ===================================================================
def plot_confusion_matrix(cm, class_names, save_path, title, normalize=False):
    """
    绘制混淆矩阵。normalize=True 时沿行归一化。
    """
    if normalize:
        cm_display = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    n = len(class_names)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm_display.max() / 2.0 if cm_display.size and cm_display.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm_display[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_display[i, j] > threshold else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_result_txt(save_path, method_name, metrics, best_params, cm):
    """保存单个方法的详细结果 txt。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"method: {method_name}\n")
        f.write(f"best_params: {best_params}\n")
        for field in METRIC_FIELDS:
            f.write(f"{field}: {metrics[field]:.6f}\n")
        f.write(f"feature_extract_time: {metrics['feature_extract_time']:.6f}\n")
        f.write(f"train_time: {metrics['train_time']:.6f}\n")
        f.write(f"test_time: {metrics['test_time']:.6f}\n")
        f.write(f"total_time: {metrics['total_time']:.6f}\n")
        f.write("confusion_matrix:\n")
        f.write(str(cm))
        f.write("\n")


def append_summary(output_dir, row):
    """追加一行到汇总 CSV。"""
    summary_path = os.path.join(output_dir, "summary_feature_compare_svc.csv")
    os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        # 格式化数值字段
        row_out = {"method": row["method"], "best_C": row["best_C"]}
        for field in SUMMARY_FIELDS:
            if field in ("method", "best_C"):
                continue
            row_out[field] = f"{row[field]:.6f}" if isinstance(row[field], (int, float)) else row[field]
        writer.writerow(row_out)


# ===================================================================
# 单方法完整实验
# ===================================================================
def run_single_method(method_name, train_features, train_labels,
                      test_features, test_labels, class_names,
                      output_dir, feature_extract_time, cv_folds):
    """对一个特征方法运行 SVC+GridSearchCV 并保存全部输出。"""
    # 训练
    model, best_params, train_time = train_with_gridsearchcv(
        train_features, train_labels, cv_folds
    )

    # 测试
    t0 = time.time()
    predictions = model.predict(test_features)
    test_time = time.time() - t0

    # 指标
    metrics = calculate_metrics(test_labels, predictions)
    total_time = feature_extract_time + train_time + test_time

    full_metrics = {
        **metrics,
        "feature_extract_time": feature_extract_time,
        "train_time": train_time,
        "test_time": test_time,
        "total_time": total_time,
    }

    # 混淆矩阵
    cm = confusion_matrix(test_labels, predictions, labels=np.arange(len(class_names)))

    safe_name = method_name.replace(" ", "_").replace("-", "_").lower()
    cm_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix.png")
    cm_norm_path = os.path.join(output_dir, f"{safe_name}_confusion_matrix_normalized.png")
    result_path = os.path.join(output_dir, f"{safe_name}_result.txt")

    plot_confusion_matrix(cm, class_names, cm_path,
                          title=f"{method_name} confusion matrix")
    plot_confusion_matrix(cm, class_names, cm_norm_path,
                          title=f"{method_name} normalized confusion matrix",
                          normalize=True)
    write_result_txt(result_path, method_name, full_metrics, best_params, cm)

    # 汇总行
    best_c = best_params.get("C", "?")
    append_summary(output_dir, {
        "method": method_name,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "best_C": str(best_c),
        "feature_extract_time": feature_extract_time,
        "train_time": train_time,
        "test_time": test_time,
        "total_time": total_time,
    })

    return full_metrics, best_params, cm


# ===================================================================
# SIFT-BoVW 特征提取（含 KMeans + StandardScaler）
# ===================================================================
def extract_sift_bovw_features(train_paths, test_paths, no_clusters, image_size=IMAGE_SIZE):
    """
    完整的 SIFT-BoVW 特征提取流水线：
    SIFT → KMeans(训练集 only) → BoVW histogram → StandardScaler(训练集 only)
    返回 train/test 特征矩阵、标签和总特征提取时间。
    """
    t0 = time.time()

    # 提取 SIFT descriptors
    train_descriptors_list = extract_sift_descriptors_per_image(train_paths, image_size)
    test_descriptors_list = extract_sift_descriptors_per_image(test_paths, image_size)

    # 训练 KMeans（仅训练集）
    kmeans = build_vocabulary(train_descriptors_list, no_clusters)

    # 编码 BoVW histogram
    train_counts = encode_bovw_histograms(kmeans, train_descriptors_list, no_clusters)
    test_counts = encode_bovw_histograms(kmeans, test_descriptors_list, no_clusters)

    # StandardScaler（仅训练集 fit）
    scaler = StandardScaler().fit(train_counts)
    train_features = scaler.transform(train_counts)
    test_features = scaler.transform(test_counts)

    feature_time = time.time() - t0
    return train_features, test_features, feature_time


# ===================================================================
# 可视化图表
# ===================================================================
METRIC_COLORS = {"accuracy": "#1f77b4", "macro_f1": "#ff7f0e"}
BAR_WIDTH = 0.30

def _add_bar_labels(ax, rects, fmt="{:.3f}"):
    for r in rects:
        h = r.get_height()
        if h > 0:
            ax.annotate(fmt.format(h), xy=(r.get_x() + r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)


def plot_metrics_bar(rows, output_dir):
    """三种方法的 accuracy + macro_f1 柱状图。"""
    methods = [r["method"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    b1 = ax.bar(x - BAR_WIDTH/2, acc, BAR_WIDTH, color=METRIC_COLORS["accuracy"],
                label="Accuracy", edgecolor="white")
    b2 = ax.bar(x + BAR_WIDTH/2, f1, BAR_WIDTH, color=METRIC_COLORS["macro_f1"],
                label="Macro-F1", edgecolor="white")
    _add_bar_labels(ax, b1)
    _add_bar_labels(ax, b2)

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("SVC + GridSearchCV: Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(output_dir, "svc_grid_metrics_bar.png")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] 已保存: {path}")
    return path


def plot_time_bar(rows, output_dir):
    """三种方法的 total_time 柱状图。"""
    methods = [r["method"] for r in rows]
    times = [r["total_time"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    colors = ["#1f77b4", "#1f77b4", "#d62728"]
    b = ax.bar(x, times, BAR_WIDTH * 1.5, color=colors[:len(methods)], edgecolor="white")
    _add_bar_labels(ax, b, fmt="{:.0f}s")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("SVC + GridSearchCV: Computational Cost", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(output_dir, "svc_grid_time_bar.png")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] 已保存: {path}")
    return path


def plot_best_c_bar(rows, output_dir):
    """三种方法 GridSearchCV 选出的 best C 柱状图（对数坐标）。"""
    methods = [r["method"] for r in rows]
    best_c_vals = [float(r["best_C"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    b = ax.bar(x, best_c_vals, BAR_WIDTH * 1.5, color=["#1f77b4", "#ff7f0e", "#2ca02c"],
               edgecolor="white")
    _add_bar_labels(ax, b, fmt="{:.4f}")

    ax.set_ylabel("Best C Value (log scale)", fontsize=12)
    ax.set_title("SVC + GridSearchCV: Best C Selected for Each Feature", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(output_dir, "svc_grid_best_c_bar.png")
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] 已保存: {path}")
    return path


# ===================================================================
# 主流程
# ===================================================================
def run_experiment(train_path, test_path, image_size, bovw_clusters, output_dir, cv_folds):
    np.random.seed(RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据（三种方法共享同样路径和标签）
    train_paths, train_labels, class_names = get_image_paths_and_labels(train_path)
    test_paths, test_labels, test_class_names = get_image_paths_and_labels(test_path)
    if class_names != test_class_names:
        raise ValueError("Train and test class folders do not match.")
    print(f"[数据] 训练集: {len(train_paths)} 张, 测试集: {len(test_paths)} 张")
    print(f"[数据] 类别: {class_names}")

    summary_rows = []

    # ================================================================
    # 1. Color Histogram
    # ================================================================
    print("\n" + "=" * 50)
    print("[1/3] Color Histogram")
    print("=" * 50)
    ch_train, ch_feat_time_train = extract_feature_matrix(
        train_paths, extract_color_histogram, image_size
    )
    ch_test, ch_feat_time_test = extract_feature_matrix(
        test_paths, extract_color_histogram, image_size
    )
    ch_feat_time = ch_feat_time_train + ch_feat_time_test
    ch_metrics, ch_best_params, _ = run_single_method(
        "Color Histogram (SVC)",
        ch_train, train_labels,
        ch_test, test_labels,
        class_names, output_dir, ch_feat_time, cv_folds,
    )
    print(f"  best_params: {ch_best_params}")
    print(f"  accuracy: {ch_metrics['accuracy']:.6f}")
    print(f"  macro_f1: {ch_metrics['macro_f1']:.6f}")
    print(f"  total_time: {ch_metrics['total_time']:.2f}s")
    summary_rows.append({
        "method": "Color Histogram (SVC)",
        "accuracy": ch_metrics["accuracy"],
        "macro_precision": ch_metrics["macro_precision"],
        "macro_recall": ch_metrics["macro_recall"],
        "macro_f1": ch_metrics["macro_f1"],
        "weighted_f1": ch_metrics["weighted_f1"],
        "best_C": str(ch_best_params.get("C", "?")),
        "feature_extract_time": ch_metrics["feature_extract_time"],
        "train_time": ch_metrics["train_time"],
        "test_time": ch_metrics["test_time"],
        "total_time": ch_metrics["total_time"],
    })

    # ================================================================
    # 2. HOG
    # ================================================================
    print("\n" + "=" * 50)
    print("[2/3] HOG")
    print("=" * 50)
    hog_train, hog_feat_time_train = extract_feature_matrix(
        train_paths, extract_hog_feature, image_size
    )
    hog_test, hog_feat_time_test = extract_feature_matrix(
        test_paths, extract_hog_feature, image_size
    )
    hog_feat_time = hog_feat_time_train + hog_feat_time_test
    hog_metrics, hog_best_params, _ = run_single_method(
        "HOG (SVC)",
        hog_train, train_labels,
        hog_test, test_labels,
        class_names, output_dir, hog_feat_time, cv_folds,
    )
    print(f"  best_params: {hog_best_params}")
    print(f"  accuracy: {hog_metrics['accuracy']:.6f}")
    print(f"  macro_f1: {hog_metrics['macro_f1']:.6f}")
    print(f"  total_time: {hog_metrics['total_time']:.2f}s")
    summary_rows.append({
        "method": "HOG (SVC)",
        "accuracy": hog_metrics["accuracy"],
        "macro_precision": hog_metrics["macro_precision"],
        "macro_recall": hog_metrics["macro_recall"],
        "macro_f1": hog_metrics["macro_f1"],
        "weighted_f1": hog_metrics["weighted_f1"],
        "best_C": str(hog_best_params.get("C", "?")),
        "feature_extract_time": hog_metrics["feature_extract_time"],
        "train_time": hog_metrics["train_time"],
        "test_time": hog_metrics["test_time"],
        "total_time": hog_metrics["total_time"],
    })

    # ================================================================
    # 3. SIFT-BoVW-K800
    # ================================================================
    print("\n" + "=" * 50)
    print(f"[3/3] SIFT-BoVW K={bovw_clusters}")
    print("=" * 50)
    bovw_train, bovw_test, bovw_feat_time = extract_sift_bovw_features(
        train_paths, test_paths, bovw_clusters, image_size
    )
    print(f"  BoVW train shape: {bovw_train.shape}")
    print(f"  BoVW test shape:  {bovw_test.shape}")

    bovw_metrics, bovw_best_params, _ = run_single_method(
        f"SIFT-BoVW K={bovw_clusters} (SVC)",
        bovw_train, train_labels,
        bovw_test, test_labels,
        class_names, output_dir, bovw_feat_time, cv_folds,
    )
    print(f"  best_params: {bovw_best_params}")
    print(f"  accuracy: {bovw_metrics['accuracy']:.6f}")
    print(f"  macro_f1: {bovw_metrics['macro_f1']:.6f}")
    print(f"  total_time: {bovw_metrics['total_time']:.2f}s")
    summary_rows.append({
        "method": f"SIFT-BoVW K={bovw_clusters} (SVC)",
        "accuracy": bovw_metrics["accuracy"],
        "macro_precision": bovw_metrics["macro_precision"],
        "macro_recall": bovw_metrics["macro_recall"],
        "macro_f1": bovw_metrics["macro_f1"],
        "weighted_f1": bovw_metrics["weighted_f1"],
        "best_C": str(bovw_best_params.get("C", "?")),
        "feature_extract_time": bovw_metrics["feature_extract_time"],
        "train_time": bovw_metrics["train_time"],
        "test_time": bovw_metrics["test_time"],
        "total_time": bovw_metrics["total_time"],
    })

    # ================================================================
    # 生成图表
    # ================================================================
    print("\n" + "=" * 50)
    print("[图表] 生成 SVC Grid 对比图表")
    print("=" * 50)
    chart_paths = []
    chart_paths.append(plot_metrics_bar(summary_rows, output_dir))
    chart_paths.append(plot_time_bar(summary_rows, output_dir))
    chart_paths.append(plot_best_c_bar(summary_rows, output_dir))

    # ================================================================
    # 终端总结
    # ================================================================
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    for row in summary_rows:
        print(f"  {row['method']}:")
        print(f"    best_C     = {row['best_C']}")
        print(f"    accuracy   = {row['accuracy']:.6f}")
        print(f"    macro_f1   = {row['macro_f1']:.6f}")
        print(f"    total_time = {row['total_time']:.2f}s")
    print(f"\n  CSV: {os.path.join(output_dir, 'summary_feature_compare_svc.csv')}")
    for p in chart_paths:
        print(f"  图表: {p}")
    print("=" * 60)

    return summary_rows, chart_paths


# ===================================================================
# CLI
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="统一 SVC+GridSearchCV 的传统特征对比实验。"
    )
    parser.add_argument("--train_path", default="dataset/train")
    parser.add_argument("--test_path", default="dataset/test")
    parser.add_argument("--image_size", default=150, type=int)
    parser.add_argument("--bovw_clusters", default=800, type=int)
    parser.add_argument("--output_dir",
                        default=os.path.join("outputs", "feature_compare_svc_k800"))
    parser.add_argument("--cv", default=5, type=int, dest="cv_folds",
                        help="GridSearchCV 交叉验证折数（默认 5）。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        args.train_path,
        args.test_path,
        args.image_size,
        args.bovw_clusters,
        args.output_dir,
        args.cv_folds,
    )
