import numpy as np
from numcompute.stats import mean, std, min, max, histogram, quantile

def test_mean():
    """Test the mean function with a simple list of numbers."""
    X = [1, 2, np.nan, 4]
    assert mean(X) == np.nanmean(X)
    print("test_mean passed!!")


def test_std():
    """Test the std function with a simple list of numbers."""
    X = [1, 2, np.nan, 4]
    assert np.isclose(std(X), np.nanstd(X))
    print("test_std passed!!")


def test_min():
    """Test the min function with a simple list of numbers."""
    X = [1, 2, np.nan, 4]
    assert min(X) == 1
    print("test_min passed!!")


def test_max():
    """Test the max function with a simple list of numbers."""
    X = [1, 2, np.nan, 4]
    assert max(X) == 4
    print("test_max passed!!")


def test_mean_axis():
    """Test the mean function with a 2D array and axis parameter."""
    X = np.array([[1, 2], [3, np.nan], [5, 6]])

    assert np.array_equal(mean(X, axis=0), np.nanmean(X, axis=0))
    print("test_mean_axis passed!!")


def test_histogram():
    """Test the histogram function with a simple list of numbers."""
    X = [1, 2, 3, 4, np.nan]
    counts, edges = histogram(X, bins=2)

    expected_counts, expected_edges = np.histogram([1, 2, 3, 4], bins=2)

    assert np.array_equal(counts, expected_counts)
    assert np.array_equal(edges, expected_edges)
    print("test_histogram passed!!")


def test_quantile_scalar():
    """Test the quantile function with a simple list of numbers and a single quantile."""
    X = [10, 20, 30, 40]
    result = quantile(X, 0.5)

    assert result == 25.0
    print("test_quantile_scalar passed!!")


def test_quantile_multiple():
    """Test the quantile function with a simple list of numbers and multiple quantiles."""
    X = [10, 20, 30, 40]
    result = quantile(X, [0.25, 0.5, 0.75])

    expected = np.array([17.5, 25.0, 32.5])

    assert np.allclose(result, expected)
    print("test_quantile_multiple passed!!")


def test_quantile_with_nan():
    """Test the quantile function with a list containing NaN values."""
    X = [10, 20, np.nan, 30]
    result = quantile(X, 0.5)

    assert result == 20.0
    print("test_quantile_with_nan passed!!")


def test_empty_histogram():
    """Test the histogram function with an array containing only NaN values."""
    X = [np.nan, np.nan]
    counts, edges = histogram(X)

    assert counts.sum() == 0
    print("test_empty_histogram passed!!")


if __name__ == "__main__":
    test_mean()
    test_std()
    test_min()
    test_max()
    test_mean_axis()
    test_histogram()
    test_quantile_scalar()
    test_quantile_multiple()
    test_quantile_with_nan()
    test_empty_histogram()