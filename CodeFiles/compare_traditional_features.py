import argparse
import csv
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
DEFAULT_BOVW_CLUSTERS = 400
CLASSIFIER_MAX_ITER = 10000
COLOR_HIST_BINS = (8, 8, 8)
SUMMARY_FIELDS = [
    "method",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "feature_extract_time",
    "train_time",
    "test_time",
    "total_time",
]


def get_image_paths_and_labels(root_path):
    """
    按类别目录和文件名排序读取图像路径，返回稳定的 paths、整数 labels 和 class_names。
    稳定顺序能保证横向对比实验在不同机器上复现实验输入。
    """
    image_paths = []
    labels = []
    class_names = [
        folder
        for folder in sorted(os.listdir(root_path))
        if os.path.isdir(os.path.join(root_path, folder))
    ]
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(root_path, class_name)
        for filename in sorted(os.listdir(class_dir)):
            image_paths.append(os.path.join(class_dir, filename))
            labels.append(class_index)
    return image_paths, labels, class_names


def read_image(image_path, image_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.resize(image, (image_size, image_size))


def l2_normalize(feature):
    norm = np.linalg.norm(feature)
    if norm == 0:
        return feature.astype(np.float32)
    return (feature / norm).astype(np.float32)


def extract_color_hist(image):
    """
    提取 3D HSV Color Histogram，并做 L2 归一化后作为整图颜色分布特征。
    输入为 BGR 图像，输出为一维向量。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, COLOR_HIST_BINS, [0, 180, 0, 256, 0, 256])
    return l2_normalize(hist.flatten())


def extract_hog_feature(image):
    """
    提取 HOG 纹理/边缘特征。
    输入为 BGR 图像，先转灰度，再按统一参数生成一维 HOG descriptor。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feature.astype(np.float32)


def extract_feature_matrix(image_paths, image_size, extractor):
    start_time = time.time()
    features = []
    for image_path in image_paths:
        image = read_image(image_path, image_size)
        features.append(extractor(image))
    return np.asarray(features, dtype=np.float32), time.time() - start_time


def extract_sift_descriptors(image_paths, labels, image_size):
    sift = cv2.SIFT_create()
    descriptor_list = []
    valid_labels = []
    start_time = time.time()
    for image_path, label in zip(image_paths, labels):
        image = read_image(image_path, image_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        # 无 SIFT descriptors 的图像不能编码 BoVW，需同步跳过 label。
        if descriptors is None:
            continue
        descriptor_list.append(descriptors)
        valid_labels.append(label)
    return descriptor_list, valid_labels, time.time() - start_time


def encode_bovw(kmeans, descriptor_list, no_clusters):
    features = np.zeros((len(descriptor_list), no_clusters), dtype=np.float32)
    for index, descriptors in enumerate(descriptor_list):
        visual_words = kmeans.predict(descriptors)
        features[index], _ = np.histogram(visual_words, bins=np.arange(no_clusters + 1))
    return features


def extract_sift_bovw_features(train_paths, train_labels, test_paths, test_labels, image_size, no_clusters):
    """
    复用 SIFT-BoVW 思路：训练集 descriptors 构建视觉词典，再分别编码训练/测试图像。
    KMeans 和 StandardScaler 都只在训练集上 fit，避免测试集信息泄漏。
    """
    train_descriptors, filtered_train_labels, train_descriptor_time = extract_sift_descriptors(
        train_paths, train_labels, image_size
    )
    test_descriptors, filtered_test_labels, test_descriptor_time = extract_sift_descriptors(
        test_paths, test_labels, image_size
    )
    if not train_descriptors:
        raise ValueError("No valid SIFT descriptors found in training set.")
    if not test_descriptors:
        raise ValueError("No valid SIFT descriptors found in test set.")

    cluster_start = time.time()
    stacked_descriptors = np.vstack(train_descriptors)
    kmeans = KMeans(n_clusters=no_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(stacked_descriptors)
    train_features = encode_bovw(kmeans, train_descriptors, no_clusters)
    test_features = encode_bovw(kmeans, test_descriptors, no_clusters)
    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)
    feature_time = train_descriptor_time + test_descriptor_time + (time.time() - cluster_start)
    return train_features, filtered_train_labels, test_features, filtered_test_labels, feature_time


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


def plot_confusion_matrix(true_labels, predictions, class_names, save_path, title):
    cm = confusion_matrix(true_labels, predictions, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
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


def write_result_txt(save_path, method_name, metrics, cm):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"method: {method_name}\n")
        for field in SUMMARY_FIELDS[1:]:
            f.write(f"{field}: {metrics[field]:.6f}\n")
        f.write("confusion_matrix:\n")
        f.write(str(cm))
        f.write("\n")


def train_and_evaluate(
    method_name,
    train_features,
    train_labels,
    test_features,
    test_labels,
    class_names,
    output_dir,
    feature_extract_time,
):
    """
    所有特征统一使用 Linear SVM + class_weight='balanced'，保证横向对比只比较特征表达。
    返回的 metrics 会同时写入单方法 txt、混淆矩阵 png 和总表 CSV。
    """
    train_start = time.time()
    classifier = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=CLASSIFIER_MAX_ITER,
        dual="auto",
    )
    classifier.fit(train_features, train_labels)
    train_time = time.time() - train_start

    test_start = time.time()
    predictions = classifier.predict(test_features)
    test_time = time.time() - test_start

    metrics = calculate_metrics(test_labels, predictions)
    metrics["feature_extract_time"] = feature_extract_time
    metrics["train_time"] = train_time
    metrics["test_time"] = test_time
    metrics["total_time"] = feature_extract_time + train_time + test_time

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"{method_name}_confusion_matrix.png")
    cm = plot_confusion_matrix(
        test_labels,
        predictions,
        class_names,
        cm_path,
        title=f"{method_name} confusion matrix",
    )
    result_path = os.path.join(output_dir, f"{method_name}_result.txt")
    write_result_txt(result_path, method_name, metrics, cm)
    return metrics


def append_summary(output_dir, rows):
    summary_path = os.path.join(output_dir, "summary_feature_compare.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: row[field] if field == "method" else f"{row[field]:.6f}"
                for field in SUMMARY_FIELDS
            })


def run_experiment(train_path, test_path, image_size, output_dir, bovw_clusters=DEFAULT_BOVW_CLUSTERS):
    np.random.seed(RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)
    train_paths, train_labels, class_names = get_image_paths_and_labels(train_path)
    test_paths, test_labels, test_class_names = get_image_paths_and_labels(test_path)
    if class_names != test_class_names:
        raise ValueError("Train and test class folders do not match.")

    summary_rows = []

    color_train, color_train_time = extract_feature_matrix(train_paths, image_size, extract_color_hist)
    color_test, color_test_time = extract_feature_matrix(test_paths, image_size, extract_color_hist)
    color_metrics = train_and_evaluate(
        "color_histogram",
        color_train,
        train_labels,
        color_test,
        test_labels,
        class_names,
        output_dir,
        color_train_time + color_test_time,
    )
    summary_rows.append({"method": "color_histogram", **color_metrics})

    hog_train, hog_train_time = extract_feature_matrix(train_paths, image_size, extract_hog_feature)
    hog_test, hog_test_time = extract_feature_matrix(test_paths, image_size, extract_hog_feature)
    hog_metrics = train_and_evaluate(
        "hog",
        hog_train,
        train_labels,
        hog_test,
        test_labels,
        class_names,
        output_dir,
        hog_train_time + hog_test_time,
    )
    summary_rows.append({"method": "hog", **hog_metrics})

    bovw_train, bovw_train_labels, bovw_test, bovw_test_labels, bovw_time = extract_sift_bovw_features(
        train_paths,
        train_labels,
        test_paths,
        test_labels,
        image_size,
        bovw_clusters,
    )
    bovw_metrics = train_and_evaluate(
        f"sift_bovw_k{bovw_clusters}",
        bovw_train,
        bovw_train_labels,
        bovw_test,
        bovw_test_labels,
        class_names,
        output_dir,
        bovw_time,
    )
    summary_rows.append({"method": f"sift_bovw_k{bovw_clusters}", **bovw_metrics})

    append_summary(output_dir, summary_rows)
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="dataset/train")
    parser.add_argument("--test_path", default="dataset/test")
    parser.add_argument("--image_size", default=150, type=int)
    parser.add_argument("--output_dir", default=os.path.join("outputs", "feature_compare"))
    parser.add_argument("--bovw_clusters", default=DEFAULT_BOVW_CLUSTERS, type=int,
                        help="Number of visual words (K) for SIFT-BoVW KMeans clustering.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.train_path, args.test_path, args.image_size, args.output_dir,
                   bovw_clusters=args.bovw_clusters)
