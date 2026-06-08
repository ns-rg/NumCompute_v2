import numpy as np
from numcompute_stream.stats import StreamStats


def test_stream_stats_mean():
    """Test that StreamStats computes correct mean over two chunks."""
    ss = StreamStats(n_features=2)
    ss.update_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
    ss.update_stats(np.array([[5.0, 6.0]]))
    result = ss.result()
    assert np.isclose(result["mean"][0], 3.0)
    assert np.isclose(result["mean"][1], 4.0)
    print("test_stream_stats_mean passed!!")


def test_stream_stats_variance():
    """Test that StreamStats computes correct variance over chunks."""
    ss = StreamStats(n_features=1)
    ss.update_stats(np.array([[2.0], [4.0], [4.0], [4.0], [5.0], [5.0], [7.0], [9.0]]))
    result = ss.result()
    assert np.isclose(result["variance"][0], 4.571428, atol=1e-4)
    print("test_stream_stats_variance passed!!")


def test_stream_stats_min_max():
    """Test that StreamStats tracks correct min and max across chunks."""
    ss = StreamStats(n_features=1)
    ss.update_stats(np.array([[3.0], [7.0]]))
    ss.update_stats(np.array([[1.0], [9.0]]))
    result = ss.result()
    assert result["min"][0] == 1.0
    assert result["max"][0] == 9.0
    print("test_stream_stats_min_max passed!!")


def test_stream_stats_nan_ignored():
    """Test that NaN values are ignored in streaming updates."""
    ss = StreamStats(n_features=1)
    ss.update_stats(np.array([[1.0], [np.nan], [3.0]]))
    result = ss.result()
    assert np.isclose(result["mean"][0], 2.0)
    print("test_stream_stats_nan_ignored passed!!")


def test_stream_stats_quantiles():
    """Test that StreamStats computes reasonable quantile estimates."""
    ss = StreamStats(n_features=1)
    ss.update_stats(np.array([[float(i)] for i in range(1, 11)]))
    result = ss.result()
    assert result["quantiles"][0][1] == 5.5
    print("test_stream_stats_quantiles passed!!")


def test_stream_stats_histogram():
    """Test that StreamStats returns histogram counts and edges."""
    ss = StreamStats(n_features=1, bins=5)
    ss.update_stats(np.array([[float(i)] for i in range(10)]))
    result = ss.result()
    counts, edges = result["histograms"][0]
    assert len(counts) == 5
    assert len(edges) == 6
    print("test_stream_stats_histogram passed!!")


def test_stream_stats_reset():
    """Test that reset clears all accumulated state."""
    ss = StreamStats(n_features=1)
    ss.update_stats(np.array([[1.0], [2.0]]))
    ss.reset()
    ss.update_stats(np.array([[10.0], [20.0]]))
    result = ss.result()
    assert np.isclose(result["mean"][0], 15.0)
    print("test_stream_stats_reset passed!!")


def test_stream_stats_sliding_window():
    """Test that sliding window mode only keeps last window_size samples."""
    ss = StreamStats(n_features=1, window_size=3)
    ss.update_stats(np.array([[1.0], [2.0], [3.0]]))
    ss.update_stats(np.array([[100.0], [200.0], [300.0]]))
    result = ss.result()
    assert result["mean"][0] > 50.0
    print("test_stream_stats_sliding_window passed!!")


def test_stream_stats_single_feature_1d():
    """Test that 1D input is handled correctly as single feature."""
    ss = StreamStats(n_features=1)
    ss.update_stats(np.array([1.0, 2.0, 3.0]))
    result = ss.result()
    assert np.isclose(result["mean"][0], 2.0)
    print("test_stream_stats_single_feature_1d passed!!")


def test_stream_stats_wrong_features():
    """Test that mismatched feature count raises ValueError."""
    ss = StreamStats(n_features=2)
    ss.update_stats(np.array([[1.0, 2.0]]))
    try:
        ss.update_stats(np.array([[1.0, 2.0, 3.0]]))
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_stats_wrong_features passed!!")


def test_stream_stats_empty_chunk():
    """Test that empty chunk raises ValueError."""
    ss = StreamStats(n_features=2)
    try:
        ss.update_stats(np.array([]).reshape(0, 2))
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_stats_empty_chunk passed!!")


def test_stream_stats_no_data_result():
    """Test that calling result() before any update raises ValueError."""
    ss = StreamStats(n_features=2)
    try:
        ss.result()
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_stats_no_data_result passed!!")


def test_stream_stats_std():
    """Test that std is the square root of variance."""
    ss = StreamStats(n_features=1)
    ss.update_stats(np.array([[2.0], [4.0], [6.0]]))
    result = ss.result()
    assert np.isclose(result["std"][0], np.sqrt(result["variance"][0]))
    print("test_stream_stats_std passed!!")


def test_stream_stats_multiple_features():
    """Test StreamStats correctly handles multiple features independently."""
    ss = StreamStats(n_features=3)
    ss.update_stats(np.array([[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]]))
    result = ss.result()
    assert np.isclose(result["mean"][0], 2.0)
    assert np.isclose(result["mean"][1], 20.0)
    assert np.isclose(result["mean"][2], 200.0)
    print("test_stream_stats_multiple_features passed!!")


def test_stream_stats_all_nan_column():
    """Test that a column of all NaNs does not update stats for that feature."""
    ss = StreamStats(n_features=2)
    ss.update_stats(np.array([[np.nan, 5.0], [np.nan, 7.0]]))
    result = ss.result()
    assert np.isclose(result["mean"][1], 6.0)
    print("test_stream_stats_all_nan_column passed!!")


if __name__ == "__main__":
    test_stream_stats_mean()
    test_stream_stats_variance()
    test_stream_stats_min_max()
    test_stream_stats_nan_ignored()
    test_stream_stats_quantiles()
    test_stream_stats_histogram()
    test_stream_stats_reset()
    test_stream_stats_sliding_window()
    test_stream_stats_single_feature_1d()
    test_stream_stats_wrong_features()
    test_stream_stats_empty_chunk()
    test_stream_stats_no_data_result()
    test_stream_stats_std()
    test_stream_stats_multiple_features()
    test_stream_stats_all_nan_column()