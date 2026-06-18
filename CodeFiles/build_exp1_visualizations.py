#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一可视化脚本。

功能：
  读取已有 CSV 结果和混淆矩阵文件，生成 11 张实验报告用图表。
  所有图表保存为 PNG，dpi=300，白底，适合直接放入实验报告。

使用方式：
  python CodeFiles/build_exp1_visualizations.py \
      --k_sweep_csv outputs/summary_results.csv \
      --same_classifier_csv outputs/final_tables/final_same_classifier_comparison.csv \
      --best_config_csv outputs/final_tables/final_best_config_comparison.csv \
      --improvement_csv outputs/final_tables/final_bovw_improvement_comparison.csv \
      --output_dir outputs/final_visualizations
"""

import argparse
import csv
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 中文字体配置
# ---------------------------------------------------------------------------
def setup_chinese_font():
    """尝试配置 matplotlib 中文字体，使图表中的中文标签正常显示。"""
    # Windows 常见中文字体优先级列表
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "DejaVu Sans",
    ]
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}

    selected = None
    for font_name in candidate_fonts:
        if font_name in available:
            selected = font_name
            break

    if selected is None:
        # 回退：列出可用中文字体
        print("[WARN] 未找到常见中文字体，使用默认字体（中文可能显示为方框）。")
        print("       可用字体列表（部分）：", ", ".join(sorted(available)[:20]))
    else:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    return selected


FONT_NAME = setup_chinese_font()


# ---------------------------------------------------------------------------
# 通用绘图辅助
# ---------------------------------------------------------------------------
BAR_WIDTH = 0.30
METRIC_COLORS = {
    "accuracy": "#1f77b4",    # matplotlib 默认蓝色
    "macro_f1": "#ff7f0e",    # matplotlib 默认橙色
}
DPI = 300


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_dataframe(csv_path):
    """读取 CSV，不存在时报错退出。"""
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 文件不存在: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[ERROR] CSV 文件为空: {csv_path}", file=sys.stderr)
        sys.exit(1)
    return df


def save_figure(fig, output_path, close=True):
    """保存图片并打印路径。"""
    ensure_output_dir(os.path.dirname(output_path))
    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"[OK] 已保存: {output_path}")
    if close:
        plt.close(fig)


def add_value_labels(ax, rects, fmt="{:.2f}"):
    """在柱状图顶部添加数值标签。"""
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(
                fmt.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def add_metric_labels_to_grouped_bars(ax, bars_grouped, fmt="{:.3f}"):
    """为分组柱状图的每一组添加数值标签。"""
    for group in bars_grouped:
        add_value_labels(ax, group, fmt=fmt)


# ---------------------------------------------------------------------------
# 图表 A: K sweep 性能曲线
# ---------------------------------------------------------------------------
def plot_k_sweep_accuracy_macro_f1(df_ks, output_dir):
    """横轴 K，纵轴 score，绘制 accuracy 和 macro_f1 两条曲线。"""
    df = df_ks.sort_values("no_clusters")
    k_values = df["no_clusters"].values
    accuracy = df["accuracy"].values
    macro_f1 = df["macro_f1"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_values, accuracy, "o-", color=METRIC_COLORS["accuracy"],
            linewidth=2, markersize=7, label="Accuracy")
    ax.plot(k_values, macro_f1, "s--", color=METRIC_COLORS["macro_f1"],
            linewidth=2, markersize=7, label="Macro-F1")

    # 标注最优点 K=800
    best_k = 800
    best_idx = list(k_values).index(best_k) if best_k in k_values else -1
    if best_idx >= 0:
        ax.annotate(
            f"K={best_k}\nAcc={accuracy[best_idx]:.4f}\nF1={macro_f1[best_idx]:.4f}",
            xy=(k_values[best_idx], accuracy[best_idx]),
            xytext=(best_k - 80, accuracy[best_idx] - 0.03),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
        )

    ax.set_xlabel("K (Number of Visual Words)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("K Sweep: Performance vs Visual Vocabulary Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    save_path = os.path.join(output_dir, "k_sweep_accuracy_macro_f1.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 B: K sweep 时间曲线
# ---------------------------------------------------------------------------
def plot_k_sweep_time(df_ks, output_dir):
    """横轴 K，纵轴 total_time，展示时间开销随 K 增长。"""
    df = df_ks.sort_values("no_clusters")
    k_values = df["no_clusters"].values
    times = df["total_time"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(k_values, 0, times, alpha=0.2, color="#1f77b4")
    ax.plot(k_values, times, "o-", color="#1f77b4", linewidth=2, markersize=7)

    # 标注每个点的具体时间
    for k, t in zip(k_values, times):
        ax.annotate(
            f"{t:.0f}s",
            xy=(k, t),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("K (Number of Visual Words)", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("K Sweep: Computational Cost vs Visual Vocabulary Size", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    save_path = os.path.join(output_dir, "k_sweep_time.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 C: K sweep 性能-时间权衡散点图
# ---------------------------------------------------------------------------
def plot_k_sweep_tradeoff(df_ks, output_dir):
    """横轴 total_time，纵轴 macro_f1，每个点标注 K 值。"""
    df = df_ks.sort_values("no_clusters")
    times = df["total_time"].values
    macro_f1 = df["macro_f1"].values
    k_labels = df["no_clusters"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(times, macro_f1, s=100, c="#2ca02c", edgecolors="black", linewidth=0.5, zorder=5)

    for k, t, f1 in zip(k_labels, times, macro_f1):
        offset_x, offset_y = 15, 0.005
        if k == 800:
            offset_x = -80
            offset_y = -0.008
        ax.annotate(
            f"K={k}",
            xy=(t, f1),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if k == 800 else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9) if k == 800 else None,
        )

    ax.set_xlabel("Total Time (seconds)", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("K Sweep: Performance-Time Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, "k_sweep_accuracy_time_tradeoff.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 D: 同分类器横向对比柱状图（accuracy + macro_f1）
# ---------------------------------------------------------------------------
def plot_same_classifier_metrics_bar(df_sc, output_dir):
    """Color Histogram / HOG / SIFT-BoVW-K800 的 accuracy 和 macro_f1 柱状图。"""
    methods = df_sc["method"].tolist()
    accuracy = df_sc["accuracy"].astype(float).values
    macro_f1 = df_sc["macro_f1"].astype(float).values

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))

    bars_acc = ax.bar(x - BAR_WIDTH/2, accuracy, BAR_WIDTH,
                      color=METRIC_COLORS["accuracy"], label="Accuracy", edgecolor="white")
    bars_f1 = ax.bar(x + BAR_WIDTH/2, macro_f1, BAR_WIDTH,
                     color=METRIC_COLORS["macro_f1"], label="Macro-F1", edgecolor="white")

    add_value_labels(ax, bars_acc, fmt="{:.3f}")
    add_value_labels(ax, bars_f1, fmt="{:.3f}")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Same Classifier Comparison (LinearSVC): Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(output_dir, "same_classifier_metrics_bar.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 E: 同分类器时间柱状图
# ---------------------------------------------------------------------------
def plot_same_classifier_time_bar(df_sc, output_dir):
    """Color Histogram / HOG / SIFT-BoVW-K800 的 total_time 柱状图。"""
    methods = df_sc["method"].tolist()
    times = df_sc["total_time"].astype(float).values

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))

    # 根据值大小使用不同颜色突出 SIFT-BoVW 的高时间成本
    bar_colors = ["#1f77b4", "#1f77b4", "#d62728"]
    bars = ax.bar(x, times, BAR_WIDTH * 1.5, color=bar_colors, edgecolor="white")

    add_value_labels(ax, bars, fmt="{:.0f}s")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("Same Classifier Comparison (LinearSVC): Computational Cost", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(output_dir, "same_classifier_time_bar.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 F: 较优配置综合对比柱状图（accuracy + macro_f1）
# ---------------------------------------------------------------------------
def plot_best_config_metrics_bar(df_bc, output_dir):
    """Color Histogram / HOG / SIFT-BoVW 最优 K 的 accuracy + macro_f1。"""
    methods_short = ["Color Histogram", "HOG", "SIFT-BoVW\n(best K)"]
    accuracy = df_bc["accuracy"].astype(float).values
    macro_f1 = df_bc["macro_f1"].astype(float).values

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods_short))

    bars_acc = ax.bar(x - BAR_WIDTH/2, accuracy, BAR_WIDTH,
                      color=METRIC_COLORS["accuracy"], label="Accuracy", edgecolor="white")
    bars_f1 = ax.bar(x + BAR_WIDTH/2, macro_f1, BAR_WIDTH,
                     color=METRIC_COLORS["macro_f1"], label="Macro-F1", edgecolor="white")

    add_value_labels(ax, bars_acc, fmt="{:.3f}")
    add_value_labels(ax, bars_f1, fmt="{:.3f}")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Best Configuration Comparison: Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_short, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(output_dir, "best_config_metrics_bar.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 G: 较优配置时间柱状图
# ---------------------------------------------------------------------------
def plot_best_config_time_bar(df_bc, output_dir):
    """Color Histogram / HOG / SIFT-BoVW 最优 K 的 total_time 柱状图。"""
    methods_short = ["Color Histogram", "HOG", "SIFT-BoVW\n(best K)"]
    times = df_bc["total_time"].astype(float).values

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods_short))

    bar_colors = ["#1f77b4", "#1f77b4", "#d62728"]
    bars = ax.bar(x, times, BAR_WIDTH * 1.5, color=bar_colors, edgecolor="white")

    add_value_labels(ax, bars, fmt="{:.0f}s")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("Best Configuration Comparison: Computational Cost", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_short, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(output_dir, "best_config_time_bar.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 H: BoVW 改进尝试指标对比
# ---------------------------------------------------------------------------
def plot_bovw_improvement_metrics_bar(df_im, output_dir):
    """Original / Improved linear / Improved rbf 的 accuracy + macro_f1。"""
    methods_short = ["Original\nSIFT-BoVW", "Improved\nlinear+RootSIFT\n+TF-IDF", "Improved\nrbf+RootSIFT\n+TF-IDF"]
    accuracy = df_im["accuracy"].astype(float).values
    macro_f1 = df_im["macro_f1"].astype(float).values

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(methods_short))

    bars_acc = ax.bar(x - BAR_WIDTH/2, accuracy, BAR_WIDTH,
                      color=METRIC_COLORS["accuracy"], label="Accuracy", edgecolor="white")
    bars_f1 = ax.bar(x + BAR_WIDTH/2, macro_f1, BAR_WIDTH,
                     color=METRIC_COLORS["macro_f1"], label="Macro-F1", edgecolor="white")

    add_value_labels(ax, bars_acc, fmt="{:.3f}")
    add_value_labels(ax, bars_f1, fmt="{:.3f}")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("BoVW Improvement Attempts: Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_short, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(output_dir, "bovw_improvement_metrics_bar.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 I: BoVW 改进尝试时间对比
# ---------------------------------------------------------------------------
def plot_bovw_improvement_time_bar(df_im, output_dir):
    """Original / Improved linear / Improved rbf 的 total_time 柱状图。"""
    methods_short = ["Original\nSIFT-BoVW", "Improved\nlinear+RootSIFT\n+TF-IDF", "Improved\nrbf+RootSIFT\n+TF-IDF"]
    times = df_im["total_time"].astype(float).values

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(methods_short))

    bars = ax.bar(x, times, BAR_WIDTH * 1.5, color=["#1f77b4", "#d62728", "#d62728"], edgecolor="white")

    add_value_labels(ax, bars, fmt="{:.0f}s")

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("BoVW Improvement Attempts: Computational Cost", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_short, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    save_path = os.path.join(output_dir, "bovw_improvement_time_bar.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 J: SIFT-BoVW 流程图（纯 matplotlib）
# ---------------------------------------------------------------------------
def plot_bovw_pipeline_diagram(output_dir):
    """
    SIFT-BoVW pipeline 流程图。
    使用 matplotlib 的 FancyBboxPatch 和 FancyArrowPatch 绘制。
    """
    steps = [
        "Input\nImage",
        "SIFT\nKeypoints &\nDescriptors",
        "KMeans\nVisual\nVocabulary",
        "BoVW\nHistogram",
        "Standard-\nization",
        "SVM\nClassifier",
        "Predicted\nClass",
    ]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)

    n = len(steps)
    box_width = 1.6
    box_height = 1.6
    y_center = 1.5

    for i, step in enumerate(steps):
        x_center = 1.0 + i * 2.0

        # 画圆角矩形框
        rect = mpatches.FancyBboxPatch(
            (x_center - box_width / 2, y_center - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor="#d9e6f2" if i < n - 1 else "#c6e6c6",
            edgecolor="#2c3e50",
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # 文字
        ax.text(
            x_center, y_center, step,
            ha="center", va="center", fontsize=9, fontweight="bold",
            linespacing=1.3,
        )

        # 画箭头（除最后一个）
        if i < n - 1:
            arrow = mpatches.FancyArrowPatch(
                (x_center + box_width / 2 + 0.05, y_center),
                (x_center + 2.0 - box_width / 2 - 0.05, y_center),
                arrowstyle="->", mutation_scale=20, linewidth=1.5, color="#2c3e50",
            )
            ax.add_patch(arrow)

    ax.axis("off")
    ax.set_title("SIFT-BoVW Pipeline", fontsize=14, fontweight="bold", pad=10)

    save_path = os.path.join(output_dir, "bovw_pipeline_diagram.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 图表 K: 三种特征流程对比图（横向三栏）
# ---------------------------------------------------------------------------
def plot_feature_pipeline_comparison(output_dir):
    """
    横向三栏对比：Color Histogram / HOG / SIFT-BoVW 各自的处理流程。
    """
    pipelines = {
        "Color Histogram": [
            "Input\nImage",
            "BGR → HSV\nColor Space",
            "3D HSV\nHistogram\n(8×8×8)",
            "L2\nNormalize",
            "LinearSVC\nClassifier",
        ],
        "HOG": [
            "Input\nImage",
            "BGR → Gray\nConvert",
            "Gradient\nOrientation\nHistogram",
            "HOG\nDescriptor\nVector",
            "LinearSVC\nClassifier",
        ],
        "SIFT-BoVW": [
            "Input\nImage",
            "SIFT Keypoints\n& Descriptors",
            "KMeans\nVisual Words\n(K=800)",
            "BoVW\nFrequency\nHistogram",
            "SVM\nClassifier",
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Feature Pipeline Comparison", fontsize=15, fontweight="bold", y=1.02)

    box_width = 2.2
    box_height = 1.2
    y_positions = [3.5, 2.7, 1.9, 1.1, 0.3]

    for col_idx, (pipe_name, steps) in enumerate(pipelines.items()):
        ax = axes[col_idx]
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4.5)

        # 标题背景
        title_rect = mpatches.FancyBboxPatch(
            (0.3, 3.9), 3.4, 0.5,
            boxstyle="round,pad=0.05",
            facecolor="#2c3e50",
            edgecolor="none",
        )
        ax.add_patch(title_rect)
        ax.text(2, 4.15, pipe_name, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

        # 步骤框
        color_map = {"Color Histogram": "#aec7e8", "HOG": "#98df8a", "SIFT-BoVW": "#ffbb78"}
        base_color = color_map.get(pipe_name, "#d9e6f2")

        for i, step in enumerate(steps):
            y = y_positions[i]
            rect = mpatches.FancyBboxPatch(
                (1.0, y), box_width, box_height,
                boxstyle="round,pad=0.08",
                facecolor=base_color if i < len(steps) - 1 else "#c6e6c6",
                edgecolor="#2c3e50",
                linewidth=1.2,
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.text(2.1, y + box_height / 2, step, ha="center", va="center",
                    fontsize=8.5, linespacing=1.2)

            # 下箭头
            if i < len(steps) - 1:
                next_y = y_positions[i + 1] + box_height
                arrow = mpatches.FancyArrowPatch(
                    (2.1, y),
                    (2.1, next_y + 0.05),
                    arrowstyle="->", mutation_scale=15, linewidth=1.2,
                    color="#2c3e50",
                )
                ax.add_patch(arrow)

        ax.axis("off")

    save_path = os.path.join(output_dir, "feature_pipeline_comparison.png")
    save_figure(fig, save_path)
    return save_path


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="实验一可视化脚本（仅读取已有 CSV，不重新训练模型）。"
    )
    parser.add_argument(
        "--k_sweep_csv",
        default=os.path.join("outputs", "summary_results.csv"),
        help="K sweep 汇总 CSV 路径。",
    )
    parser.add_argument(
        "--same_classifier_csv",
        default=os.path.join("outputs", "final_tables", "final_same_classifier_comparison.csv"),
        help="同分类器横向对比 CSV 路径。",
    )
    parser.add_argument(
        "--best_config_csv",
        default=os.path.join("outputs", "final_tables", "final_best_config_comparison.csv"),
        help="较优配置综合对比 CSV 路径。",
    )
    parser.add_argument(
        "--improvement_csv",
        default=os.path.join("outputs", "final_tables", "final_bovw_improvement_comparison.csv"),
        help="BoVW 改进对比 CSV 路径。",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("outputs", "final_visualizations"),
        help="可视化图片输出目录。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    print("=" * 60)
    print("[实验一] 可视化图表生成")
    print("=" * 60)

    generated_paths = []

    # ---- 读取数据 ----
    print(f"\n[读取] K sweep CSV: {args.k_sweep_csv}")
    df_ks = read_csv_dataframe(args.k_sweep_csv)
    print(f"        {len(df_ks)} 行。")

    print(f"[读取] 同分类器对比 CSV: {args.same_classifier_csv}")
    df_sc = read_csv_dataframe(args.same_classifier_csv)
    print(f"        {len(df_sc)} 行。")

    print(f"[读取] 较优配置对比 CSV: {args.best_config_csv}")
    df_bc = read_csv_dataframe(args.best_config_csv)
    print(f"        {len(df_bc)} 行。")

    print(f"[读取] BoVW 改进对比 CSV: {args.improvement_csv}")
    df_im = read_csv_dataframe(args.improvement_csv)
    print(f"        {len(df_im)} 行。")

    # ---- A: K sweep 性能曲线 ----
    print("\n--- K Sweep 性能图表 ---")
    p = plot_k_sweep_accuracy_macro_f1(df_ks, args.output_dir)
    generated_paths.append(p)

    # ---- B: K sweep 时间曲线 ----
    p = plot_k_sweep_time(df_ks, args.output_dir)
    generated_paths.append(p)

    # ---- C: K sweep 权衡散点图 ----
    p = plot_k_sweep_tradeoff(df_ks, args.output_dir)
    generated_paths.append(p)

    # ---- D: 同分类器指标柱状图 ----
    print("\n--- 同分类器横向对比图表 ---")
    p = plot_same_classifier_metrics_bar(df_sc, args.output_dir)
    generated_paths.append(p)

    # ---- E: 同分类器时间柱状图 ----
    p = plot_same_classifier_time_bar(df_sc, args.output_dir)
    generated_paths.append(p)

    # ---- F: 较优配置指标柱状图 ----
    print("\n--- 较优配置综合对比图表 ---")
    p = plot_best_config_metrics_bar(df_bc, args.output_dir)
    generated_paths.append(p)

    # ---- G: 较优配置时间柱状图 ----
    p = plot_best_config_time_bar(df_bc, args.output_dir)
    generated_paths.append(p)

    # ---- H: BoVW 改进指标对比 ----
    print("\n--- BoVW 改进尝试图表 ---")
    p = plot_bovw_improvement_metrics_bar(df_im, args.output_dir)
    generated_paths.append(p)

    # ---- I: BoVW 改进时间对比 ----
    p = plot_bovw_improvement_time_bar(df_im, args.output_dir)
    generated_paths.append(p)

    # ---- J: SIFT-BoVW 流程图 ----
    print("\n--- 流程图 ---")
    p = plot_bovw_pipeline_diagram(args.output_dir)
    generated_paths.append(p)

    # ---- K: 三种特征流程对比图 ----
    p = plot_feature_pipeline_comparison(args.output_dir)
    generated_paths.append(p)

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("可视化图表生成完毕！")
    print("=" * 60)
    print(f"共生成 {len(generated_paths)} 张图片，保存在: {args.output_dir}")
    print("")
    for i, p in enumerate(generated_paths, 1):
        print(f"  [{i:2d}] {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
