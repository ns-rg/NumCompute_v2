import numpy as np
from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    precision,
    recall,
    f1_score,
    mse,
)


def test_accuracy():
    """
    Testing accuracy calculation.
    We will use a simple binary classification example to verify that the accuracy is calculated correctly.
    """
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    result = accuracy(y_true, y_pred)
    assert result == 0.75
    print("test_accuracy passed!!")


def test_confusion_matrix():
    """
    Testing confusion matrix calculation.
    We will use a simple binary classification example to verify that the confusion matrix is calculated correctly.
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = confusion_matrix(y_true, y_pred)
    expected = np.array([[1, 1], [0, 2]])

    assert np.array_equal(result, expected)
    print("test_confusion_matrix passed!!")


def test_precision():
    """
    Testing precision calculation.
    We will use a simple binary classification example to verify that the precision is calculated correctly.
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = precision(y_true, y_pred)
    expected = (1.0 + 2 / 3) / 2

    assert np.isclose(result, expected)
    print("test_precision passed!!")


def test_recall():
    """
    Testing recall calculation.
    We will use a simple binary classification example to verify that the recall is calculated correctly.
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = recall(y_true, y_pred)
    expected = (1 / 2 + 1.0) / 2

    assert np.isclose(result, expected)
    print("test_recall passed!!")


def test_f1_score():
    """
    Testing F1 score calculation.
    We will use a simple binary classification example to verify that the F1 score is calculated correctly.
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = f1_score(y_true, y_pred)

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    expected = 2 * (p * r) / (p + r + 1e-8)

    assert np.isclose(result, expected)
    print("test_f1_score passed!!")


def test_mse():
    """
    Testing Mean Squared Error calculation.
    We will use a simple example to verify that the MSE is calculated correctly.
    """
    y_true = [2, 4, 6]
    y_pred = [3, 5, 4]

    result = mse(y_true, y_pred)

    assert result == 2.0
    print("test_mse passed!!")


def test_invalid_lengths():
    """
    Testing error handling for mismatched input lengths.
    We will provide inputs of different lengths and verify that a ValueError is raised.
    """
    try:
        accuracy([1, 2, 3], [1, 2])
        assert False
    except ValueError:
        assert True
    print("test_invalid_lengths passed!!")


def test_empty_input():
    """
    Testing error handling for empty input.
    We will provide empty inputs and verify that a ValueError is raised.
    """
    try:
        accuracy([], [])
        assert False
    except ValueError:
        assert True
    print("test_empty_input passed!!")


if __name__ == "__main__":
    test_accuracy()
    test_confusion_matrix()
    test_precision()
    test_recall()
    test_f1_score()
    test_mse()
    test_invalid_lengths()
    test_empty_input()
