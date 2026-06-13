import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "CodeFiles" / "compare_traditional_features.py"


class FakeCV2(types.SimpleNamespace):
    COLOR_BGR2HSV = 40
    COLOR_BGR2GRAY = 6
    NORM_MINMAX = 32

    @staticmethod
    def cvtColor(image, code):
        if code == FakeCV2.COLOR_BGR2GRAY:
            return image[:, :, 0]
        return image

    @staticmethod
    def calcHist(images, channels, mask, hist_size, ranges):
        image = images[0]
        hist, _ = np.histogramdd(
            image.reshape(-1, 3),
            bins=hist_size,
            range=[ranges[0:2], ranges[2:4], ranges[4:6]],
        )
        return hist.astype(np.float32)

    @staticmethod
    def normalize(src, dst, alpha=0, beta=1, norm_type=None):
        src_min = float(src.min())
        src_max = float(src.max())
        if src_max == src_min:
            dst[:] = 0
        else:
            dst[:] = (src - src_min) / (src_max - src_min)
        return dst


sys.modules["cv2"] = FakeCV2()
fake_skimage = types.ModuleType("skimage")
fake_skimage_feature = types.ModuleType("skimage.feature")
fake_skimage_feature.hog = lambda image, **kwargs: np.ones(4, dtype=np.float32)
fake_skimage.feature = fake_skimage_feature
sys.modules["skimage"] = fake_skimage
sys.modules["skimage.feature"] = fake_skimage_feature

spec = importlib.util.spec_from_file_location("compare_features", SCRIPT_PATH)
compare_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_features)


def test_get_image_paths_and_labels_are_sorted(tmp_path):
    for folder in ["sea", "city"]:
        folder_path = tmp_path / folder
        folder_path.mkdir()
        for filename in ["2.jpg", "1.jpg"]:
            (folder_path / filename).write_text("x")

    paths, labels, class_names = compare_features.get_image_paths_and_labels(str(tmp_path))

    assert labels == [0, 0, 1, 1]
    assert class_names == ["city", "sea"]
    assert [Path(path).name for path in paths] == ["1.jpg", "2.jpg", "1.jpg", "2.jpg"]


def test_extract_color_hist_returns_normalized_flat_feature():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:8, :, 0] = 90
    image[8:, :, 1] = 128

    feature = compare_features.extract_color_hist(image)

    assert feature.ndim == 1
    assert feature.shape[0] == 8 * 8 * 8
    assert np.isclose(np.linalg.norm(feature), 1.0)


def test_train_and_evaluate_returns_required_metrics(tmp_path):
    train_features = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]],
        dtype=np.float32,
    )
    test_features = np.array([[0.05, 0.0], [1.05, 1.0]], dtype=np.float32)
    train_labels = [0, 0, 1, 1]
    test_labels = [0, 1]

    result = compare_features.train_and_evaluate(
        method_name="unit",
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        class_names=["a", "b"],
        output_dir=str(tmp_path),
        feature_extract_time=0.5,
    )

    for key in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "feature_extract_time",
        "train_time",
        "test_time",
        "total_time",
    ]:
        assert key in result
    assert (tmp_path / "unit_result.txt").exists()
    assert (tmp_path / "unit_confusion_matrix.png").exists()
