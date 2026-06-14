import argparse
import csv
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
IMAGE_SIZE = 150
CLASS_NAMES = ["city", "face", "green", "house_building", "house_indoor", "office", "sea"]
METRIC_FIELDS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
TIME_FIELDS = ["train_time", "test_time", "total_time"]
SUMMARY_FIELDS = [
    "method",
    "no_clusters",
    "kernel",
    "use_rootsift",
    "use_tfidf",
    *METRIC_FIELDS,
    *TIME_FIELDS,
]


np.random.seed(RANDOM_STATE)


def get_image_paths_and_labels(path):
    """
    按类别目录和文件名排序读取图像，返回稳定的路径和整数标签。
    数据读取顺序固定后，KMeans、SVM 参数搜索和输出结果更容易复现。
    """
    image_paths = []
    labels = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        folder_path = os.path.join(path, class_name)
        if not os.path.isdir(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(class_index)
    return image_paths, labels


def read_gray_image(image_path):
    image = cv2.imread(image_path, 0)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))


def apply_rootsift(descriptors, eps=1e-7):
    """
    RootSIFT 先对每个 SIFT descriptor 做 L1 normalization，再进行 sqrt 变换。
    输入输出 shape 均为 [N, 128]，空 descriptor 会在调用前被跳过。
    """
    descriptors = descriptors.astype(np.float32, copy=False)
    l1_norm = np.sum(np.abs(descriptors), axis=1, keepdims=True)
    normalized = descriptors / np.maximum(l1_norm, eps)
    return np.sqrt(normalized).astype(np.float32)


def extract_descriptors(image_paths, labels, use_rootsift):
    sift = cv2.SIFT_create()
    descriptor_list = []
    filtered_labels = []
    filtered_paths = []
    for image_path, label in zip(image_paths, labels):
        image = read_gray_image(image_path)
        _, descriptors = sift.detectAndCompute(image, None)
        # 没有 SIFT descriptors 的图像无法进入 BoVW 编码，必须同步跳过 label。
        if descriptors is None:
            continue
        if use_rootsift:
            descriptors = apply_rootsift(descriptors)
        descriptor_list.append(descriptors)
        filtered_labels.append(label)
        filtered_paths.append(image_path)
    return descriptor_list, filtered_labels, filtered_paths


def stack_descriptors(descriptor_list):
    if not descriptor_list:
        raise ValueError("No valid SIFT descriptors were extracted.")
    return np.vstack(descriptor_list)


def build_visual_vocabulary(descriptor_list, no_clusters):
    descriptors = stack_descriptors(descriptor_list)
    kmeans = KMeans(n_clusters=no_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(descriptors)
    return kmeans


def encode_bovw_counts(kmeans, descriptor_list, no_clusters):
    """
    将每张图像的 descriptors 批量预测为 visual words，再统计 count histogram。
    这里先保留 count，后续可选择 TF-IDF 或 StandardScaler。
    """
    features = np.zeros((len(descriptor_list), no_clusters), dtype=np.float32)
    for index, descriptors in enumerate(descriptor_list):
        visual_words = kmeans.predict(descriptors)
        features[index], _ = np.histogram(visual_words, bins=np.arange(no_clusters + 1))
    return features


def apply_tfidf(train_counts, test_counts, transformer_cls=TfidfTransformer):
    """
    TF-IDF 只在训练集 BoVW count 上 fit，测试集只能 transform，避免测试集信息泄漏。
    """
    transformer = transformer_cls()
    train_features = transformer.fit_transform(train_counts)
    test_features = transformer.transform(test_counts)
    if hasattr(train_features, "toarray"):
        train_features = train_features.toarray()
    if hasattr(test_features, "toarray"):
        test_features = test_features.toarray()
    return train_features.astype(np.float32), test_features.astype(np.float32), transformer


def scale_features(train_features, test_features):
    scaler = StandardScaler().fit(train_features)
    return scaler.transform(train_features), scaler.transform(test_features), scaler


def get_svm_param_grid(kernel_type):
    if kernel_type == "linear":
        return {"C": [0.01, 0.1, 1, 10, 100]}
    if kernel_type == "rbf":
        return {
            "C": [0.1, 1, 10, 100],
            "gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1],
        }
    raise ValueError("kernel_type must be either linear or rbf")


def train_svm(train_features, train_labels, kernel_type):
    svm = SVC(kernel=kernel_type, class_weight="balanced", random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        svm,
        get_svm_param_grid(kernel_type),
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_search.fit(train_features, train_labels)
    return grid_search.best_estimator_, grid_search.best_params_


def calculate_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def plot_confusion_matrix(true_labels, predictions, save_path):
    cm = confusion_matrix(true_labels, predictions, labels=np.arange(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="Improved BoVW confusion matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                format(cm[row, col], "d"),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return cm


def build_method_name(no_clusters, kernel_type, use_rootsift, use_tfidf):
    parts = ["improved_bovw", f"k{no_clusters}", kernel_type]
    if use_rootsift:
        parts.append("rootsift")
    if use_tfidf:
        parts.append("tfidf")
    return "_".join(parts)


def write_result_txt(result_path, method_name, no_clusters, kernel_type, use_rootsift, use_tfidf, best_params, metrics, timings, cm):
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"method: {method_name}\n")
        f.write(f"no_clusters: {no_clusters}\n")
        f.write(f"kernel: {kernel_type}\n")
        f.write(f"use_rootsift: {use_rootsift}\n")
        f.write(f"use_tfidf: {use_tfidf}\n")
        f.write(f"best_params: {best_params}\n")
        for field in METRIC_FIELDS:
            f.write(f"{field}: {metrics[field]:.6f}\n")
        for field in TIME_FIELDS:
            f.write(f"{field}: {timings[field]:.6f}\n")
        f.write("confusion_matrix:\n")
        f.write(str(cm))
        f.write("\n")


def append_summary(output_dir, method_name, no_clusters, kernel_type, use_rootsift, use_tfidf, metrics, timings):
    summary_path = os.path.join(output_dir, "summary_improved.csv")
    os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        row = {
            "method": method_name,
            "no_clusters": no_clusters,
            "kernel": kernel_type,
            "use_rootsift": use_rootsift,
            "use_tfidf": use_tfidf,
        }
        row.update({field: f"{metrics[field]:.6f}" for field in METRIC_FIELDS})
        row.update({field: f"{timings[field]:.6f}" for field in TIME_FIELDS})
        writer.writerow(row)


def train_and_evaluate(train_path, test_path, no_clusters, kernel_type, use_rootsift, use_tfidf, output_dir):
    np.random.seed(RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)
    method_name = build_method_name(no_clusters, kernel_type, use_rootsift, use_tfidf)

    total_start = time.time()
    train_start = time.time()
    train_paths, train_labels = get_image_paths_and_labels(train_path)
    train_descriptors, train_labels, _ = extract_descriptors(train_paths, train_labels, use_rootsift)
    kmeans = build_visual_vocabulary(train_descriptors, no_clusters)
    train_counts = encode_bovw_counts(kmeans, train_descriptors, no_clusters)
    if use_tfidf:
        tfidf = TfidfTransformer()
        train_features = tfidf.fit_transform(train_counts)
        if hasattr(train_features, "toarray"):
            train_features = train_features.toarray()
        train_features = train_features.astype(np.float32)
    else:
        tfidf = None
        train_features = train_counts
    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    svm, best_params = train_svm(train_features, train_labels, kernel_type)
    train_time = time.time() - train_start

    test_start = time.time()
    test_paths, test_labels = get_image_paths_and_labels(test_path)
    test_descriptors, test_labels, _ = extract_descriptors(test_paths, test_labels, use_rootsift)
    test_counts = encode_bovw_counts(kmeans, test_descriptors, no_clusters)
    # 测试集只能复用训练阶段 fit 好的 TF-IDF 和 StandardScaler，避免信息泄漏。
    if tfidf is not None:
        test_features = tfidf.transform(test_counts)
        if hasattr(test_features, "toarray"):
            test_features = test_features.toarray()
        test_features = test_features.astype(np.float32)
    else:
        test_features = test_counts
    test_features = scaler.transform(test_features)
    predictions = svm.predict(test_features)
    test_time = time.time() - test_start
    total_time = time.time() - total_start

    metrics = calculate_metrics(test_labels, predictions)
    timings = {
        "train_time": train_time,
        "test_time": test_time,
        "total_time": total_time,
    }
    cm_path = os.path.join(output_dir, f"{method_name}_confusion_matrix.png")
    cm = plot_confusion_matrix(test_labels, predictions, cm_path)
    result_path = os.path.join(output_dir, f"{method_name}_result.txt")
    write_result_txt(result_path, method_name, no_clusters, kernel_type, use_rootsift, use_tfidf, best_params, metrics, timings, cm)
    append_summary(output_dir, method_name, no_clusters, kernel_type, use_rootsift, use_tfidf, metrics, timings)
    return metrics, timings, best_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--no_clusters", required=True, type=int)
    parser.add_argument("--kernel_type", default="linear", choices=["linear", "rbf"])
    parser.add_argument("--use_rootsift", action="store_true")
    parser.add_argument("--use_tfidf", action="store_true")
    parser.add_argument("--output_dir", default=os.path.join("outputs", "bovw_improved"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        args.train_path,
        args.test_path,
        args.no_clusters,
        args.kernel_type,
        args.use_rootsift,
        args.use_tfidf,
        args.output_dir,
    )
