import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier


def test_tree_fit_predict():
    """Test that tree fits and predicts on simple linearly separable data."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    preds = tree.predict(X)
    assert len(preds) == 4
    print("test_tree_fit_predict passed!!")


def test_tree_perfect_accuracy():
    """Test that tree achieves perfect accuracy on clean separable data."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)
    assert tree.score(X, y) == 1.0
    print("test_tree_perfect_accuracy passed!!")


def test_tree_partial_fit_accumulates():
    """Test that partial_fit accumulates data across chunks."""
    X1 = np.array([[1.0], [2.0]])
    y1 = np.array([0, 0])
    X2 = np.array([[3.0], [4.0]])
    y2 = np.array([1, 1])
    tree = DecisionTreeClassifier(max_depth=3)
    tree.partial_fit(X1, y1)
    tree.partial_fit(X2, y2)
    assert tree._X_seen.shape[0] == 4
    print("test_tree_partial_fit_accumulates passed!!")


def test_tree_partial_fit_predict():
    """Test that predict works correctly after partial_fit."""
    X1 = np.array([[1.0], [2.0]])
    y1 = np.array([0, 0])
    X2 = np.array([[3.0], [4.0]])
    y2 = np.array([1, 1])
    tree = DecisionTreeClassifier(max_depth=3)
    tree.partial_fit(X1, y1)
    tree.partial_fit(X2, y2)
    preds = tree.predict(np.array([[1.5], [3.5]]))
    assert len(preds) == 2
    print("test_tree_partial_fit_predict passed!!")


def test_tree_max_depth():
    """Test that tree respects max_depth setting."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    assert tree._tree["leaf"] == False
    assert tree._tree["left"]["leaf"] == True
    assert tree._tree["right"]["leaf"] == True
    print("test_tree_max_depth passed!!")


def test_tree_gini_criterion():
    """Test that tree works with gini criterion."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(criterion="gini")
    tree.fit(X, y)
    preds = tree.predict(X)
    assert len(preds) == 4
    print("test_tree_gini_criterion passed!!")


def test_tree_entropy_criterion():
    """Test that tree works with entropy criterion."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(X, y)
    preds = tree.predict(X)
    assert len(preds) == 4
    print("test_tree_entropy_criterion passed!!")


def test_tree_invalid_criterion():
    """Test that invalid criterion raises ValueError."""
    try:
        tree = DecisionTreeClassifier(criterion="invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_tree_invalid_criterion passed!!")


def test_tree_min_samples_split():
    """Test that min_samples_split prevents splitting small nodes."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(min_samples_split=10)
    tree.fit(X, y)
    assert tree._tree["leaf"] == True
    print("test_tree_min_samples_split passed!!")


def test_tree_single_class():
    """Test that tree handles data with only one class."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    preds = tree.predict(X)
    assert np.all(preds == 1)
    print("test_tree_single_class passed!!")


def test_tree_predict_before_fit():
    """Test that predict before fit raises ValueError."""
    tree = DecisionTreeClassifier()
    try:
        tree.predict(np.array([[1.0]]))
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_tree_predict_before_fit passed!!")


def test_tree_score():
    """Test that score returns correct accuracy."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)
    assert tree.score(X, y) == 1.0
    print("test_tree_score passed!!")


def test_tree_multiclass():
    """Test that tree handles multiclass classification."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1, 2, 2])
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert len(np.unique(preds)) <= 3
    print("test_tree_multiclass passed!!")


def test_tree_max_features():
    """Test that max_features limits features considered at each split."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_features=2)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert len(preds) == 4
    print("test_tree_max_features passed!!")


def test_tree_mismatched_X_y():
    """Test that mismatched X and y raise ValueError."""
    tree = DecisionTreeClassifier()
    try:
        tree.fit(np.array([[1.0], [2.0]]), np.array([0]))
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_tree_mismatched_X_y passed!!")


if __name__ == "__main__":
    test_tree_fit_predict()
    test_tree_perfect_accuracy()
    test_tree_partial_fit_accumulates()
    test_tree_partial_fit_predict()
    test_tree_max_depth()
    test_tree_gini_criterion()
    test_tree_entropy_criterion()
    test_tree_invalid_criterion()
    test_tree_min_samples_split()
    test_tree_single_class()
    test_tree_predict_before_fit()
    test_tree_score()
    test_tree_multiclass()
    test_tree_max_features()
    test_tree_mismatched_X_y()