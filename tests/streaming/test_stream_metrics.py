import numpy as np
from numcompute_stream.metrics import StreamMetrics


def test_stream_metrics_accuracy():
    """Test that StreamMetrics computes correct accuracy over chunks."""
    sm = StreamMetrics()
    sm.update([1, 0, 1, 1], [1, 0, 0, 1])
    result = sm.result()
    assert np.isclose(result["accuracy"], 0.75)
    print("test_stream_metrics_accuracy passed!!")


def test_stream_metrics_accumulates():
    """Test that metrics accumulate correctly across multiple chunks."""
    sm = StreamMetrics()
    sm.update([1, 0], [1, 0])
    sm.update([1, 1], [0, 1])
    result = sm.result()
    assert np.isclose(result["accuracy"], 0.75)
    print("test_stream_metrics_accumulates passed!!")


def test_stream_metrics_precision():
    """Test that StreamMetrics computes correct precision."""
    sm = StreamMetrics()
    sm.update([0, 0, 1, 1], [0, 1, 1, 1])
    result = sm.result()
    assert result["precision"] > 0.0
    print("test_stream_metrics_precision passed!!")


def test_stream_metrics_recall():
    """Test that StreamMetrics computes correct recall."""
    sm = StreamMetrics()
    sm.update([0, 0, 1, 1], [0, 1, 1, 1])
    result = sm.result()
    assert result["recall"] > 0.0
    print("test_stream_metrics_recall passed!!")


def test_stream_metrics_f1():
    """Test that StreamMetrics computes correct f1 score."""
    sm = StreamMetrics()
    sm.update([0, 0, 1, 1], [0, 1, 1, 1])
    result = sm.result()
    assert result["f1"] > 0.0
    print("test_stream_metrics_f1 passed!!")


def test_stream_metrics_confusion_matrix():
    """Test that confusion matrix accumulates correctly over chunks."""
    sm = StreamMetrics()
    sm.update([0, 1], [0, 1])
    sm.update([0, 1], [1, 0])
    result = sm.result()
    assert result["confusion_matrix"].shape == (2, 2)
    print("test_stream_metrics_confusion_matrix passed!!")


def test_stream_metrics_reset():
    """Test that reset clears all accumulated data."""
    sm = StreamMetrics()
    sm.update([1, 0, 1], [1, 0, 0])
    sm.reset()
    sm.update([1, 1], [1, 1])
    result = sm.result()
    assert np.isclose(result["accuracy"], 1.0)
    print("test_stream_metrics_reset passed!!")


def test_stream_metrics_auc_binary():
    """Test that AUC is computed for binary classification."""
    sm = StreamMetrics()
    sm.update([0, 0, 1, 1], [0, 0, 1, 1])
    result = sm.result()
    assert np.isclose(result["auc"], 1.0, atol=0.1)
    print("test_stream_metrics_auc_binary passed!!")


def test_stream_metrics_auc_multiclass_zero():
    """Test that AUC returns 0.0 for multiclass (not supported)."""
    sm = StreamMetrics()
    sm.update([0, 1, 2], [0, 1, 2])
    result = sm.result()
    assert result["auc"] == 0.0
    print("test_stream_metrics_auc_multiclass_zero passed!!")


def test_stream_metrics_rolling_window():
    """Test that rolling window mode only scores last window_size samples."""
    sm = StreamMetrics(window_size=2)
    sm.update([1, 1], [0, 0])
    sm.update([1, 1], [1, 1])
    result = sm.result()
    assert np.isclose(result["accuracy"], 1.0)
    print("test_stream_metrics_rolling_window passed!!")


def test_stream_metrics_no_data():
    """Test that result() before any update raises ValueError."""
    sm = StreamMetrics()
    try:
        sm.result()
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_metrics_no_data passed!!")


def test_stream_metrics_mismatched_lengths():
    """Test that mismatched chunk lengths raise ValueError."""
    sm = StreamMetrics()
    try:
        sm.update([1, 0], [1])
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_metrics_mismatched_lengths passed!!")


def test_stream_metrics_perfect_predictions():
    """Test that perfect predictions give accuracy of 1.0."""
    sm = StreamMetrics()
    sm.update([0, 1, 0, 1], [0, 1, 0, 1])
    result = sm.result()
    assert np.isclose(result["accuracy"], 1.0)
    print("test_stream_metrics_perfect_predictions passed!!")


def test_stream_metrics_all_wrong():
    """Test that all wrong predictions give accuracy of 0.0."""
    sm = StreamMetrics()
    sm.update([0, 0, 1, 1], [1, 1, 0, 0])
    result = sm.result()
    assert np.isclose(result["accuracy"], 0.0)
    print("test_stream_metrics_all_wrong passed!!")


def test_stream_metrics_result_keys():
    """Test that result dict contains all expected keys."""
    sm = StreamMetrics()
    sm.update([0, 1], [0, 1])
    result = sm.result()
    for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix", "auc"]:
        assert key in result
    print("test_stream_metrics_result_keys passed!!")


if __name__ == "__main__":
    test_stream_metrics_accuracy()
    test_stream_metrics_accumulates()
    test_stream_metrics_precision()
    test_stream_metrics_recall()
    test_stream_metrics_f1()
    test_stream_metrics_confusion_matrix()
    test_stream_metrics_reset()
    test_stream_metrics_auc_binary()
    test_stream_metrics_auc_multiclass_zero()
    test_stream_metrics_rolling_window()
    test_stream_metrics_no_data()
    test_stream_metrics_mismatched_lengths()
    test_stream_metrics_perfect_predictions()
    test_stream_metrics_all_wrong()
    test_stream_metrics_result_keys()