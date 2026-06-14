import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "CodeFiles" / "BoW_improved.py"

sys.modules["cv2"] = types.SimpleNamespace(
    COLOR_BGR2GRAY=6,
    SIFT_create=lambda: None,
    cvtColor=lambda image, code: image,
)

spec = importlib.util.spec_from_file_location("bow_improved", SCRIPT_PATH)
bow_improved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bow_improved)


def test_rootsift_l1_normalizes_then_sqrt_transforms():
    descriptors = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    rootsift = bow_improved.apply_rootsift(descriptors)

    expected_first = np.sqrt(np.array([0.25, 0.75, 0.0], dtype=np.float32))
    np.testing.assert_allclose(rootsift[0], expected_first, rtol=1e-6)
    np.testing.assert_allclose(rootsift[1], np.zeros(3), rtol=1e-6)


def test_tfidf_fits_train_once_and_only_transforms_test():
    class FakeTfidfTransformer:
        fit_transform_calls = 0
        transform_calls = 0

        def fit_transform(self, features):
            FakeTfidfTransformer.fit_transform_calls += 1
            return features + 1

        def transform(self, features):
            FakeTfidfTransformer.transform_calls += 1
            return features + 2

    train_counts = np.array([[1.0, 0.0], [0.0, 1.0]])
    test_counts = np.array([[1.0, 1.0]])

    train_features, test_features, transformer = bow_improved.apply_tfidf(
        train_counts,
        test_counts,
        transformer_cls=FakeTfidfTransformer,
    )

    assert isinstance(transformer, FakeTfidfTransformer)
    assert FakeTfidfTransformer.fit_transform_calls == 1
    assert FakeTfidfTransformer.transform_calls == 1
    np.testing.assert_array_equal(train_features, train_counts + 1)
    np.testing.assert_array_equal(test_features, test_counts + 2)


def test_get_svm_param_grid_matches_required_ranges():
    linear_grid = bow_improved.get_svm_param_grid("linear")
    rbf_grid = bow_improved.get_svm_param_grid("rbf")

    assert linear_grid == {"C": [0.01, 0.1, 1, 10, 100]}
    assert rbf_grid == {
        "C": [0.1, 1, 10, 100],
        "gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1],
    }
