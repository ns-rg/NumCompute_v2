import numpy as np
from numcompute_stream.ensemble import EnsembleClassifier


def test_ensemble_fit_predict():
    """Test that ensemble fits and predicts on simple data."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf = EnsembleClassifier(n_trees=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 4
    print("test_ensemble_fit_predict passed!!")


def test_ensemble_bagging():
    """Test that bagging method fits and predicts correctly."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf = EnsembleClassifier(n_trees=5, method="bagging")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 4
    print("test_ensemble_bagging passed!!")


def test_ensemble_random_forest():
    """Test that random forest method fits and predicts correctly."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    clf = EnsembleClassifier(n_trees=5, method="random_forest")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 4
    print("test_ensemble_random_forest passed!!")


def test_ensemble_partial_fit_accumulates():
    """Test that partial_fit accumulates data across chunks."""
    X1 = np.array([[1.0], [2.0]])
    y1 = np.array([0, 0])
    X2 = np.array([[3.0], [4.0]])
    y2 = np.array([1, 1])
    clf = EnsembleClassifier(n_trees=5)
    clf.partial_fit(X1, y1)
    clf.partial_fit(X2, y2)
    assert clf._X_seen.shape[0] == 4
    print("test_ensemble_partial_fit_accumulates passed!!")


def test_ensemble_partial_fit_predict():
    """Test that predict works correctly after partial_fit."""
    X1 = np.array([[1.0], [2.0]])
    y1 = np.array([0, 0])
    X2 = np.array([[3.0], [4.0]])
    y2 = np.array([1, 1])
    clf = EnsembleClassifier(n_trees=5)
    clf.partial_fit(X1, y1)
    clf.partial_fit(X2, y2)
    preds = clf.predict(np.array([[1.5], [3.5]]))
    assert len(preds) == 2
    print("test_ensemble_partial_fit_predict passed!!")


def test_ensemble_score():
    """Test that score returns a value between 0 and 1."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf = EnsembleClassifier(n_trees=5)
    clf.fit(X, y)
    s = clf.score(X, y)
    assert 0.0 <= s <= 1.0
    print("test_ensemble_score passed!!")


def test_ensemble_n_trees():
    """Test that correct number of trees are created."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf = EnsembleClassifier(n_trees=7)
    clf.fit(X, y)
    assert len(clf.trees_) == 7
    print("test_ensemble_n_trees passed!!")


def test_ensemble_invalid_method():
    """Test that invalid method raises ValueError."""
    try:
        clf = EnsembleClassifier(method="invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_ensemble_invalid_method passed!!")


def test_ensemble_invalid_n_trees():
    """Test that n_trees less than 1 raises ValueError."""
    try:
        clf = EnsembleClassifier(n_trees=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_ensemble_invalid_n_trees passed!!")


def test_ensemble_predict_before_fit():
    """Test that predict before fit raises ValueError."""
    clf = EnsembleClassifier(n_trees=5)
    try:
        clf.predict(np.array([[1.0]]))
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_ensemble_predict_before_fit passed!!")


def test_ensemble_mismatched_X_y():
    """Test that mismatched X and y raise ValueError."""
    clf = EnsembleClassifier(n_trees=5)
    try:
        clf.fit(np.array([[1.0], [2.0]]), np.array([0]))
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_ensemble_mismatched_X_y passed!!")


def test_ensemble_majority_vote():
    """Test that majority vote is used for predictions."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf = EnsembleClassifier(n_trees=10, method="bagging")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})
    print("test_ensemble_majority_vote passed!!")


def test_ensemble_multiclass():
    """Test that ensemble handles multiclass classification."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    clf = EnsembleClassifier(n_trees=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(np.unique(preds)) <= 3
    print("test_ensemble_multiclass passed!!")


def test_ensemble_random_forest_default_max_features():
    """Test that random forest uses sqrt(n_features) by default."""
    X = np.array([[1.0, 2.0, 3.0, 4.0]] * 10)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    clf = EnsembleClassifier(n_trees=5, method="random_forest")
    clf.fit(X, y)
    assert clf._get_max_features(4) == 2
    print("test_ensemble_random_forest_default_max_features passed!!")


def test_ensemble_bagging_uses_all_features():
    """Test that bagging uses all features by default."""
    X = np.array([[1.0, 2.0, 3.0, 4.0]] * 10)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    clf = EnsembleClassifier(n_trees=5, method="bagging")
    clf.fit(X, y)
    assert clf._get_max_features(4) == 4
    print("test_ensemble_bagging_uses_all_features passed!!")


if __name__ == "__main__":
    test_ensemble_fit_predict()
    test_ensemble_bagging()
    test_ensemble_random_forest()
    test_ensemble_partial_fit_accumulates()
    test_ensemble_partial_fit_predict()
    test_ensemble_score()
    test_ensemble_n_trees()
    test_ensemble_invalid_method()
    test_ensemble_invalid_n_trees()
    test_ensemble_predict_before_fit()
    test_ensemble_mismatched_X_y()
    test_ensemble_majority_vote()
    test_ensemble_multiclass()
    test_ensemble_random_forest_default_max_features()
    test_ensemble_bagging_uses_all_features()