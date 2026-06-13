import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BOW_PATH = REPO_ROOT / "CodeFiles" / "BoW.py"

sys.modules.setdefault("cv2", types.SimpleNamespace(SIFT_create=lambda: None))

spec = importlib.util.spec_from_file_location("bow_module", BOW_PATH)
bow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bow)


def test_get_files_sorts_and_shuffles_training_with_fixed_seed(tmp_path):
    for folder in ["b", "a"]:
        folder_path = tmp_path / folder
        folder_path.mkdir()
        for filename in ["2.jpg", "1.jpg"]:
            (folder_path / filename).write_text("x")

    first = bow.getFiles(True, str(tmp_path))
    second = bow.getFiles(True, str(tmp_path))
    expected_sorted = [
        str(tmp_path / "a" / "1.jpg"),
        str(tmp_path / "a" / "2.jpg"),
        str(tmp_path / "b" / "1.jpg"),
        str(tmp_path / "b" / "2.jpg"),
    ]

    assert first == second
    assert sorted(first) == expected_sorted
    assert bow.getFiles(False, str(tmp_path)) == expected_sorted


def test_extract_features_predicts_each_image_in_batch_and_histograms():
    class FakeKMeans:
        def __init__(self):
            self.seen_shapes = []

        def predict(self, descriptors):
            self.seen_shapes.append(descriptors.shape)
            return np.array([0, 2, 2]) if len(self.seen_shapes) == 1 else np.array([1, 1])

    descriptor_list = [
        np.zeros((3, 128), dtype=np.float32),
        np.ones((2, 128), dtype=np.float32),
    ]

    features = bow.extractFeatures(FakeKMeans(), descriptor_list, 2, 4)

    np.testing.assert_array_equal(features[0], np.array([1, 0, 2, 0]))
    np.testing.assert_array_equal(features[1], np.array([0, 2, 0, 0]))


def test_calculate_metrics_returns_requested_scores():
    metrics = bow.calculateMetrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])

    assert set(metrics) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }
    assert metrics["accuracy"] == 0.75
