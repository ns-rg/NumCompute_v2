import numpy as np
from numcompute.io import load_csv

def test_load_csv_basic():
    """
    Test the basic functionality of the load_csv function.
    We will create a simple CSV file with known content and verify that it is loaded correctly.
    """
    
    data = load_csv("test.csv", skip_header=1)
    assert isinstance(data, np.ndarray), "Output is not a NumPy array"
    assert data.shape == (3, 3), f"Unexpected shape: {data.shape}"

    print("test_load_csv_basic passed!!")


def test_missing_values():
    """
    Test the load_csv function's ability to handle missing values.
    We will create a CSV file with some missing values and verify that they are loaded as NaN.
    """
    
    data = load_csv("test.csv", skip_header=1)

    value = data[1][1]
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise AssertionError(f"Value is not convertible to float: {value}")

    assert np.isnan(value), f"Expected NaN, got {value}"

    print("test_missing_values passed!!")


if __name__ == "__main__":
    test_load_csv_basic()
    test_missing_values()