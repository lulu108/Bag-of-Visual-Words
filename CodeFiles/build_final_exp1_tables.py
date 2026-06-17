#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一最终对比表格构建脚本。

功能：
  1. 从 K sweep CSV 中按指定指标筛选 BoVW 最优结果。
  2. 从特征对比 CSV 中读取 Color Histogram / HOG / SIFT-BoVW-K800 结果。
  3. 生成三张最终对比 CSV 表格和一份实验报告 Markdown 文档。

使用方式：
  python CodeFiles/build_final_exp1_tables.py \
      --k_sweep_csv outputs/summary_results.csv \
      --feature_compare_csv outputs/feature_compare_k800/summary_feature_compare.csv \
      --improved_csv outputs/bovw_improved/summary_improved.csv \
      --output_dir outputs/final_tables \
      --select_by macro_f1
"""

import argparse
import csv
import os
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
K_SWEEP_METHOD_FILTER = "SIFT-BoVW-SVM"

# feature_compare CSV 中三行的 method 名称
FC_COLOR_METHOD = "color_histogram"
FC_HOG_METHOD = "hog"
FC_BOVW_PREFIX = "sift_bovw_k"

# improved CSV 中期望的 method 名称
IMPROVED_LINEAR_METHOD = "improved_bovw_k800_linear_rootsift_tfidf"
IMPROVED_RBF_METHOD = "improved_bovw_k800_rbf_rootsift_tfidf"

# 表 1 输出列
TABLE_COMPARISON_FIELDS = [
    "method",
    "setting",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "total_time",
    "notes",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def read_csv_rows(csv_path):
    """读取 CSV 文件，返回 list[dict]。对不存在的文件直接报错退出。"""
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 文件不存在: {csv_path}", file=sys.stderr)
        sys.exit(1)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        print(f"[ERROR] CSV 文件为空: {csv_path}", file=sys.stderr)
        sys.exit(1)
    return rows


def safe_float(val, default=0.0):
    """将字符串转为 float，失败时返回 default。"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# 步骤 3：从 K sweep 中选出 BoVW 最优结果
# ---------------------------------------------------------------------------
def select_best_bovw(rows, select_by="macro_f1"):
    """
    从 k_sweep_csv 中筛选 method == SIFT-BoVW-SVM 的行，
    按 select_by 降序选择最优行；平局时比较 accuracy。
    """
    bovw_rows = [r for r in rows if r.get("method", "").strip() == K_SWEEP_METHOD_FILTER]
    if not bovw_rows:
        print(f"[ERROR] k_sweep_csv 中未找到 method='{K_SWEEP_METHOD_FILTER}' 的行。", file=sys.stderr)
        sys.exit(1)

    def sort_key(r):
        primary = safe_float(r.get(select_by, 0))
        secondary = safe_float(r.get("accuracy", 0))
        return (primary, secondary)

    best = max(bovw_rows, key=sort_key)
    return best


# ---------------------------------------------------------------------------
# 步骤 4：读取 feature_compare CSV，生成表 1
# ---------------------------------------------------------------------------
def build_same_classifier_table(fc_rows, output_dir):
    """
    表 1：同分类器横向对比表。
    从 feature_compare CSV 中提取 color_histogram / hog / sift_bovw_k* 三行。
    """
    # 查找目标行
    color_row = None
    hog_row = None
    bovw_row = None
    for r in fc_rows:
        method = r.get("method", "").strip()
        if method == FC_COLOR_METHOD:
            color_row = r
        elif method == FC_HOG_METHOD:
            hog_row = r
        elif method.startswith(FC_BOVW_PREFIX):
            bovw_row = r

    missing = []
    if color_row is None:
        missing.append(FC_COLOR_METHOD)
    if hog_row is None:
        missing.append(FC_HOG_METHOD)
    if bovw_row is None:
        missing.append(f"{FC_BOVW_PREFIX}*")
    if missing:
        print(f"[ERROR] feature_compare_csv 中缺少以下方法行: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    rows_out = []

    # Color Histogram
    rows_out.append({
        "method": "Color Histogram",
        "setting": "3D HSV (8×8×8) + LinearSVC",
        "accuracy": f"{safe_float(color_row.get('accuracy')):.6f}",
        "macro_precision": f"{safe_float(color_row.get('macro_precision')):.6f}",
        "macro_recall": f"{safe_float(color_row.get('macro_recall')):.6f}",
        "macro_f1": f"{safe_float(color_row.get('macro_f1')):.6f}",
        "weighted_f1": f"{safe_float(color_row.get('weighted_f1')):.6f}",
        "total_time": f"{safe_float(color_row.get('total_time')):.2f}",
        "notes": "来自 compare_traditional_features.py；使用 LinearSVC，仅作为基线参考。",
    })

    # HOG
    rows_out.append({
        "method": "HOG",
        "setting": "HOG (orient=9, ppc=8×8, cpb=2×2) + LinearSVC",
        "accuracy": f"{safe_float(hog_row.get('accuracy')):.6f}",
        "macro_precision": f"{safe_float(hog_row.get('macro_precision')):.6f}",
        "macro_recall": f"{safe_float(hog_row.get('macro_recall')):.6f}",
        "macro_f1": f"{safe_float(hog_row.get('macro_f1')):.6f}",
        "weighted_f1": f"{safe_float(hog_row.get('weighted_f1')):.6f}",
        "total_time": f"{safe_float(hog_row.get('total_time')):.2f}",
        "notes": "来自 compare_traditional_features.py；使用 LinearSVC，仅作为纹理特征基线。",
    })

    # SIFT-BoVW-K800 (from feature_compare)
    bovw_method_name = bovw_row.get("method", "").strip()
    rows_out.append({
        "method": "SIFT-BoVW-K800",
        "setting": f"SIFT-BoVW K=800 + LinearSVC",
        "accuracy": f"{safe_float(bovw_row.get('accuracy')):.6f}",
        "macro_precision": f"{safe_float(bovw_row.get('macro_precision')):.6f}",
        "macro_recall": f"{safe_float(bovw_row.get('macro_recall')):.6f}",
        "macro_f1": f"{safe_float(bovw_row.get('macro_f1')):.6f}",
        "weighted_f1": f"{safe_float(bovw_row.get('weighted_f1')):.6f}",
        "total_time": f"{safe_float(bovw_row.get('total_time')):.2f}",
        "notes": "来自 compare_traditional_features.py；使用 LinearSVC，与 Color Histogram/HOG 统一分类器。",
    })

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "final_same_classifier_comparison.csv")
    write_output_csv(csv_path, TABLE_COMPARISON_FIELDS, rows_out)
    return rows_out, csv_path


# ---------------------------------------------------------------------------
# 步骤 5：生成表 2 —— 较优配置综合对比
# ---------------------------------------------------------------------------
def build_best_config_table(fc_rows, best_bovw, output_dir):
    """
    表 2：较优配置综合对比表。
    包含 Color Histogram、HOG（均来自 feature_compare）和 K sweep 最优 BoVW。
    """
    color_row = None
    hog_row = None
    for r in fc_rows:
        method = r.get("method", "").strip()
        if method == FC_COLOR_METHOD:
            color_row = r
        elif method == FC_HOG_METHOD:
            hog_row = r

    if color_row is None or hog_row is None:
        print("[ERROR] feature_compare_csv 缺少 Color Histogram 或 HOG 行。", file=sys.stderr)
        sys.exit(1)

    best_k = best_bovw.get("no_clusters", "?").strip()
    best_kernel = best_bovw.get("kernel", "?").strip()

    rows_out = []

    # Color Histogram
    rows_out.append({
        "method": "Color Histogram",
        "setting": "3D HSV (8×8×8) + LinearSVC",
        "accuracy": f"{safe_float(color_row.get('accuracy')):.6f}",
        "macro_precision": f"{safe_float(color_row.get('macro_precision')):.6f}",
        "macro_recall": f"{safe_float(color_row.get('macro_recall')):.6f}",
        "macro_f1": f"{safe_float(color_row.get('macro_f1')):.6f}",
        "weighted_f1": f"{safe_float(color_row.get('weighted_f1')):.6f}",
        "total_time": f"{safe_float(color_row.get('total_time')):.2f}",
        "notes": "来自 compare_traditional_features.py（LinearSVC）；本表不是严格控制同一分类器。",
    })

    # HOG
    rows_out.append({
        "method": "HOG",
        "setting": "HOG (orient=9, ppc=8×8, cpb=2×2) + LinearSVC",
        "accuracy": f"{safe_float(hog_row.get('accuracy')):.6f}",
        "macro_precision": f"{safe_float(hog_row.get('macro_precision')):.6f}",
        "macro_recall": f"{safe_float(hog_row.get('macro_recall')):.6f}",
        "macro_f1": f"{safe_float(hog_row.get('macro_f1')):.6f}",
        "weighted_f1": f"{safe_float(hog_row.get('weighted_f1')):.6f}",
        "total_time": f"{safe_float(hog_row.get('total_time')):.2f}",
        "notes": "来自 compare_traditional_features.py（LinearSVC）；本表不是严格控制同一分类器。",
    })

    # SIFT-BoVW best from K sweep
    rows_out.append({
        "method": "SIFT-BoVW (最优 K)",
        "setting": f"K={best_k}, kernel={best_kernel}, SVC pipeline",
        "accuracy": f"{safe_float(best_bovw.get('accuracy')):.6f}",
        "macro_precision": f"{safe_float(best_bovw.get('macro_precision')):.6f}",
        "macro_recall": f"{safe_float(best_bovw.get('macro_recall')):.6f}",
        "macro_f1": f"{safe_float(best_bovw.get('macro_f1')):.6f}",
        "weighted_f1": f"{safe_float(best_bovw.get('weighted_f1')):.6f}",
        "total_time": f"{safe_float(best_bovw.get('total_time')):.2f}",
        "notes": (
            f"来自 K sweep 最优结果（select_by=macro_f1），"
            f"不是 feature_compare 中的 LinearSVC 结果；"
            f"使用 GridSearchCV 搜索 SVC 最优超参数。"
        ),
    })

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "final_best_config_comparison.csv")
    write_output_csv(csv_path, TABLE_COMPARISON_FIELDS, rows_out)
    return rows_out, csv_path


# ---------------------------------------------------------------------------
# 步骤 6：生成表 3 —— BoVW 改进尝试对比（仅当 improved_csv 存在时）
# ---------------------------------------------------------------------------
def build_improvement_table(improved_csv, best_bovw, output_dir):
    """
    表 3：BoVW 改进尝试对比表。
    如果 improved_csv 不存在则跳过（返回 None）。
    """
    if not os.path.isfile(improved_csv):
        print(f"[INFO] improved_csv 不存在 ({improved_csv})，跳过表 3 生成。")
        return None, None

    improved_rows = read_csv_rows(improved_csv)
    improved_by_method = {r.get("method", "").strip(): r for r in improved_rows}

    rows_out = []

    # Original best BoVW from K sweep
    best_k = best_bovw.get("no_clusters", "?").strip()
    best_kernel = best_bovw.get("kernel", "?").strip()
    rows_out.append({
        "method": "Original SIFT-BoVW (最优 K)",
        "setting": f"K={best_k}, kernel={best_kernel}, SVC pipeline",
        "accuracy": f"{safe_float(best_bovw.get('accuracy')):.6f}",
        "macro_precision": f"{safe_float(best_bovw.get('macro_precision')):.6f}",
        "macro_recall": f"{safe_float(best_bovw.get('macro_recall')):.6f}",
        "macro_f1": f"{safe_float(best_bovw.get('macro_f1')):.6f}",
        "weighted_f1": f"{safe_float(best_bovw.get('weighted_f1')):.6f}",
        "total_time": f"{safe_float(best_bovw.get('total_time')):.2f}",
        "notes": "原始 SIFT-BoVW（无 RootSIFT/TF-IDF）；来自 K sweep 最优结果。",
    })

    # Improved: linear + RootSIFT + TF-IDF
    if IMPROVED_LINEAR_METHOD in improved_by_method:
        r = improved_by_method[IMPROVED_LINEAR_METHOD]
        rows_out.append({
            "method": "Improved BoVW (linear + RootSIFT + TF-IDF)",
            "setting": "K=800, kernel=linear, RootSIFT + TF-IDF, SVC pipeline",
            "accuracy": f"{safe_float(r.get('accuracy')):.6f}",
            "macro_precision": f"{safe_float(r.get('macro_precision')):.6f}",
            "macro_recall": f"{safe_float(r.get('macro_recall')):.6f}",
            "macro_f1": f"{safe_float(r.get('macro_f1')):.6f}",
            "weighted_f1": f"{safe_float(r.get('weighted_f1')):.6f}",
            "total_time": f"{safe_float(r.get('total_time')):.2f}",
            "notes": "来自 BoW_improved.py；RootSIFT + TF-IDF 旨在改善特征表达和视觉词权重。",
        })
    else:
        print(f"[INFO] improved_csv 中未找到 '{IMPROVED_LINEAR_METHOD}'，跳过该行。")

    # Improved: rbf + RootSIFT + TF-IDF
    if IMPROVED_RBF_METHOD in improved_by_method:
        r = improved_by_method[IMPROVED_RBF_METHOD]
        rows_out.append({
            "method": "Improved BoVW (rbf + RootSIFT + TF-IDF)",
            "setting": "K=800, kernel=rbf, RootSIFT + TF-IDF, SVC pipeline",
            "accuracy": f"{safe_float(r.get('accuracy')):.6f}",
            "macro_precision": f"{safe_float(r.get('macro_precision')):.6f}",
            "macro_recall": f"{safe_float(r.get('macro_recall')):.6f}",
            "macro_f1": f"{safe_float(r.get('macro_f1')):.6f}",
            "weighted_f1": f"{safe_float(r.get('weighted_f1')):.6f}",
            "total_time": f"{safe_float(r.get('total_time')):.2f}",
            "notes": "来自 BoW_improved.py；RBF kernel 可捕捉非线性决策边界。",
        })
    else:
        print(f"[INFO] improved_csv 中未找到 '{IMPROVED_RBF_METHOD}'，跳过该行。")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "final_bovw_improvement_comparison.csv")
    write_output_csv(csv_path, TABLE_COMPARISON_FIELDS, rows_out)
    return rows_out, csv_path


# ---------------------------------------------------------------------------
# 输出写入
# ---------------------------------------------------------------------------
def write_output_csv(csv_path, fieldnames, rows):
    """将 rows 按 fieldnames 写入 CSV。"""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[OK] 已保存: {csv_path}")


# ---------------------------------------------------------------------------
# 步骤 7：生成 Markdown 报告
# ---------------------------------------------------------------------------
def generate_markdown_report(
    table1_rows,
    table2_rows,
    table3_rows,
    best_bovw,
    best_k,
    best_accuracy,
    best_macro_f1,
    select_by,
    report_path,
):
    """生成 experiments/exp1_final_comparison.md。"""
    best_kernel = best_bovw.get("kernel", "?").strip()

    def csv_row_to_md(row, fields):
        return "| " + " | ".join(str(row.get(f, "")) for f in fields) + " |"

    header_md = "| " + " | ".join(TABLE_COMPARISON_FIELDS) + " |"
    sep_md = "|" + "|".join("---:" for _ in TABLE_COMPARISON_FIELDS) + "|"

    lines = []
    lines.append("# 实验一：最终实验结果汇总")
    lines.append("")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 选择指标：{select_by}（平局时比较 accuracy）")
    lines.append("")
    lines.append("## 结果组织方式")
    lines.append("")
    lines.append(
        "实验一包含三条子实验线，各自输出中间 CSV 结果，"
        "最终由 `CodeFiles/build_final_exp1_tables.py` 读取并生成以下三张对比表："
    )
    lines.append("")
    lines.append("1. **同分类器横向对比表**（表 1）：")
    lines.append("   Color Histogram、HOG、SIFT-BoVW-K800 三者统一使用 LinearSVC，")
    lines.append("   用于控制分类器一致的横向对比。")
    lines.append("2. **较优配置综合对比表**（表 2）：")
    lines.append("   各传统特征在实验中的较优配置 — Color Histogram 和 HOG 来自")
    lines.append("   feature_compare 脚本，SIFT-BoVW 使用 K sweep 中的最优结果。")
    lines.append("3. **BoVW 改进尝试对比表**（表 3）：")
    lines.append("   对比原始 SIFT-BoVW 与 RootSIFT+TF-IDF 改进版本的性能差异。")
    lines.append("")
    lines.append("## SIFT-BoVW 最优结果选取说明")
    lines.append("")
    lines.append(
        f"SIFT-BoVW 的最优结果来自 **K sweep 实验**（`outputs/summary_results.csv`），"
    )
    lines.append(
        f"按 `{select_by}` 选择 K={best_k}, kernel={best_kernel} 的配置，"
        f"accuracy={best_accuracy}, macro_f1={best_macro_f1}。"
    )
    lines.append("")
    lines.append(
        "**为什么不直接用 feature_compare 中的 `sift_bovw_k800`？**"
    )
    lines.append("")
    lines.append(
        "feature_compare 中的 `sift_bovw_k800` 固定使用 **LinearSVC**（无 GridSearchCV），"
        "其目的是与 Color Histogram、HOG **统一分类器**进行横向对比。"
    )
    lines.append(
        "而 K sweep 中的 SIFT-BoVW 使用 **GridSearchCV 搜索 SVC 最优超参数**，"
        "因此 K sweep 中的线性核 SVC 结果（经过 C 参数搜索）"
        "可能优于 feature_compare 中的 LinearSVC 结果。"
    )
    lines.append(
        "为保证表 1 仅比较特征表达，三方法统一用 LinearSVC；"
        "为展示 BoVW 在实验一中的最优性能，表 2 和表 3 采用 K sweep 的最优结果。"
    )
    lines.append("")
    lines.append("## 表 1：同分类器横向对比")
    lines.append("")
    lines.append(
        "Color Histogram、HOG、SIFT-BoVW-K800 均来自 "
        "`compare_traditional_features.py`，统一使用 **LinearSVC**（`class_weight='balanced'`）。"
    )
    lines.append("该表用于控制分类器一致的横向对比。")
    lines.append("")
    lines.append(header_md)
    lines.append(sep_md)
    for row in table1_rows:
        lines.append(csv_row_to_md(row, TABLE_COMPARISON_FIELDS))
    lines.append("")
    lines.append("## 表 2：较优配置综合对比")
    lines.append("")
    lines.append(
        "**注意：本表不是严格控制同一分类器的实验。**"
        "它表示各传统特征在当前实验中的较优配置。"
    )
    lines.append(
        f"SIFT-BoVW 使用 K sweep 中的最优结果（K={best_k}, kernel={best_kernel}, SVC pipeline），"
        "而 Color Histogram 和 HOG 来自 feature_compare 脚本（LinearSVC）。"
    )
    lines.append("因此本表仅用于概括性对比，不可直接归因于单一变量。")
    lines.append("")
    lines.append(header_md)
    lines.append(sep_md)
    for row in table2_rows:
        lines.append(csv_row_to_md(row, TABLE_COMPARISON_FIELDS))
    lines.append("")

    if table3_rows is not None:
        lines.append("## 表 3：BoVW 改进尝试对比")
        lines.append("")
        lines.append(
            "对比原始 SIFT-BoVW（无 RootSIFT/TF-IDF）与"
            " RootSIFT + TF-IDF 改进版本的性能。"
        )
        lines.append("改进实验由 `CodeFiles/BoW_improved.py` 完成。")
        lines.append("")
        lines.append(header_md)
        lines.append(sep_md)
        for row in table3_rows:
            lines.append(csv_row_to_md(row, TABLE_COMPARISON_FIELDS))
        lines.append("")
    else:
        lines.append("## 表 3：BoVW 改进尝试对比")
        lines.append("")
        lines.append("（improved_csv 不存在，该表暂未生成。）")
        lines.append("")

    # 确保 experiments 目录存在
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] 已保存实验报告: {report_path}")


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="实验一最终对比表格构建脚本（仅读取已有 CSV，不重新训练模型）。"
    )
    parser.add_argument(
        "--k_sweep_csv",
        default=os.path.join("outputs", "summary_results.csv"),
        help="K sweep 汇总 CSV 路径。",
    )
    parser.add_argument(
        "--feature_compare_csv",
        default=os.path.join("outputs", "feature_compare_k800", "summary_feature_compare.csv"),
        help="传统特征对比汇总 CSV 路径。",
    )
    parser.add_argument(
        "--improved_csv",
        default=os.path.join("outputs", "bovw_improved", "summary_improved.csv"),
        help="BoVW 改进实验汇总 CSV 路径。",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("outputs", "final_tables"),
        help="最终对比表格输出目录。",
    )
    parser.add_argument(
        "--select_by",
        default="macro_f1",
        choices=["macro_f1", "accuracy", "weighted_f1"],
        help="选择 BoVW 最优 K 时使用的指标（默认 macro_f1）。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("[实验一] 最终对比表格构建")
    print("=" * 60)

    # ---- 步骤 3：K sweep 最优选择 ----
    print(f"\n[1/5] 读取 K sweep CSV: {args.k_sweep_csv}")
    k_rows = read_csv_rows(args.k_sweep_csv)
    print(f"       共 {len(k_rows)} 行。")

    best = select_best_bovw(k_rows, select_by=args.select_by)
    best_k = best.get("no_clusters", "?").strip()
    best_accuracy = safe_float(best.get("accuracy"))
    best_macro_f1 = safe_float(best.get("macro_f1"))

    print(f"\n[结果] 最优 BoVW 配置:")
    print(f"       K = {best_k}")
    print(f"       kernel = {best.get('kernel', '?').strip()}")
    print(f"       accuracy = {best_accuracy:.6f}")
    print(f"       macro_f1 = {best_macro_f1:.6f}")
    print(f"       weighted_f1 = {safe_float(best.get('weighted_f1')):.6f}")
    print(f"       total_time = {safe_float(best.get('total_time')):.2f} s")

    # ---- 步骤 4：表 1 ----
    print(f"\n[2/5] 读取 feature_compare CSV: {args.feature_compare_csv}")
    fc_rows = read_csv_rows(args.feature_compare_csv)
    print(f"       共 {len(fc_rows)} 行。")

    table1_rows, table1_path = build_same_classifier_table(fc_rows, args.output_dir)

    # ---- 步骤 5：表 2 ----
    print(f"\n[3/5] 生成较优配置综合对比表...")
    table2_rows, table2_path = build_best_config_table(fc_rows, best, args.output_dir)

    # ---- 步骤 6：表 3 (optional) ----
    print(f"\n[4/5] 检查 improved_csv: {args.improved_csv}")
    table3_rows, table3_path = build_improvement_table(
        args.improved_csv, best, args.output_dir
    )

    # ---- 步骤 7：Markdown 报告 ----
    print(f"\n[5/5] 生成实验报告...")
    report_path = os.path.join("experiments", "exp1_final_comparison.md")
    generate_markdown_report(
        table1_rows=table1_rows,
        table2_rows=table2_rows,
        table3_rows=table3_rows,
        best_bovw=best,
        best_k=best_k,
        best_accuracy=f"{best_accuracy:.6f}",
        best_macro_f1=f"{best_macro_f1:.6f}",
        select_by=args.select_by,
        report_path=report_path,
    )

    # ---- 终端总结 ----
    print("\n" + "=" * 60)
    print("构建完成！")
    print("=" * 60)
    print(f"  最优 SIFT-BoVW K 值: {best_k}")
    print(f"  最优 accuracy:        {best_accuracy:.6f}")
    print(f"  最优 macro_f1:        {best_macro_f1:.6f}")
    print(f"  表 1 (同分类器对比):  {table1_path}")
    print(f"  表 2 (较优配置对比):  {table2_path}")
    if table3_path:
        print(f"  表 3 (BoVW 改进对比): {table3_path}")
    else:
        print(f"  表 3 (BoVW 改进对比): 未生成（improved_csv 不存在）")
    print(f"  实验报告:              {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
