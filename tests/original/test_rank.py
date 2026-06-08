import numpy as np
from numcompute.rank import rank, percentile

def test_rank_average():
    """Test the rank function with the "average" method."""
    data = [10, 20, 20, 30]
    result = rank(data, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0])

    assert np.array_equal(result, expected)
    print("test_rank_average passed!!")


def test_rank_dense():
    """Test the rank function with the "dense" method."""
    data = [10, 20, 20, 30]
    result = rank(data, method="dense")
    expected = np.array([1.0, 2.0, 2.0, 3.0])

    assert np.array_equal(result, expected)
    print("test_rank_dense passed!!")


def test_rank_ordinal():
    """Test the rank function with the "ordinal" method."""
    data = [10, 20, 20, 30]
    result = rank(data, method="ordinal")
    expected = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.array_equal(result, expected)
    print("test_rank_ordinal passed!!")


def test_rank_unsorted_data():
    """Test the rank function with unsorted data."""
    data = [30, 10, 20]
    result = rank(data, method="average")
    expected = np.array([3.0, 1.0, 2.0])

    assert np.array_equal(result, expected)
    print("test_rank_unsorted_data passed!!")


def test_rank_invalid_method():
    """Test the rank function with an invalid method."""
    try:
        rank([1, 2, 3], method="wrong")
        assert False
    except ValueError:
        assert True
    print("test_rank_invalid_method passed!!")


def test_percentile_linear():
    """Test the percentile function with linear interpolation."""
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="linear")

    assert result == 25.0
    print("test_percentile_linear passed!!")


def test_percentile_lower():
    """Test the percentile function with lower interpolation."""
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="lower")

    assert result == 20
    print("test_percentile_lower passed!!")


def test_percentile_higher():
    """Test the percentile function with higher interpolation."""
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="higher")

    assert result == 30
    print("test_percentile_higher passed!!")


def test_percentile_midpoint():
    """Test the percentile function with midpoint interpolation."""
    data = [10, 20, 30, 40]
    result = percentile(data, 50, interpolation="midpoint")

    assert result == 25.0
    print("test_percentile_midpoint passed!!")


def test_percentile_with_nan():
    """Test the percentile function with NaN values in the data."""
    data = [10, 20, np.nan, 30]
    result = percentile(data, 50)

    assert result == 20.0
    print("test_percentile_with_nan passed!!")


def test_percentile_invalid_q():
    """Test the percentile function with an invalid q value."""
    try:
        percentile([1, 2, 3], 120)
        assert False
    except ValueError:
        assert True
    print("test_percentile_invalid_q passed!!")


def test_percentile_empty_data():
    """Test the percentile function with an empty data list."""
    try:
        percentile([], 50)
        assert False
    except ValueError:
        assert True
    print("test_percentile_empty_data passed!!")


def test_percentile_invalid_interpolation():
    """Test the percentile function with an invalid interpolation method."""
    try:
        percentile([1, 2, 3], 50, interpolation="wrong")
        assert False
    except ValueError:
        assert True
    print("test_percentile_invalid_interpolation passed!!")


if __name__ == "__main__":
    test_rank_average()
    test_rank_dense()
    test_rank_ordinal()
    test_rank_unsorted_data()
    test_rank_invalid_method()

    test_percentile_linear()
    test_percentile_lower()
    test_percentile_higher()
    test_percentile_midpoint()
    test_percentile_with_nan()
    test_percentile_invalid_q()
    test_percentile_empty_data()
    test_percentile_invalid_interpolation()