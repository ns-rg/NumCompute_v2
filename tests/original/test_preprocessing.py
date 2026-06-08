from numcompute.preprocessing import Imputer
import numpy as np

from tests.test_io import test_load_csv_basic, test_missing_values
from numcompute.preprocessing import StandardScaler
from numcompute.preprocessing import OneHotEncoder


def test_imputer():
    """
    Test the Imputer class with a simple example.
    We will create a small dataset with missing values, apply the imputer, and verify that the missing values are filled in."""
    X = np.array(
        [
            ["Alice", 25, 50000],
            ["Bob", np.nan, 60000],
            ["Alice", 30, np.nan],
        ],
        dtype=object,
    )

    imputer = Imputer(strategy="mean")
    X_out = imputer.fit_transform(X)

    assert not np.isnan(float(X_out[1][1]))
    assert not np.isnan(float(X_out[2][2]))

    print("test_imputer passed!!")


def test_standard_scaler():
    """
    Test the StandardScaler class with a simple example.
    We will create a small dataset with numeric and categorical values, apply the scaler, and verify that the numeric values are scaled while the categorical values remain unchanged.
    """
    X = np.array(
        [
            ["A", 10, 100],
            ["B", 20, 200],
            ["C", 30, 300],
        ],
        dtype=object,
    )

    scaler = StandardScaler()
    X_out = scaler.fit_transform(X)

    assert abs(float(X_out[0][1])) < 2
    assert abs(float(X_out[1][1])) < 2

    assert X_out[0][0] == "A"

    print("test_standard_scaler passed!!")


def test_one_hot_encoder():
    """
    Test the OneHotEncoder class with a simple example.
    We will create a small dataset with categorical and numeric values, apply the encoder, and verify that the categorical values are one-hot encoded while the numeric values remain unchanged.
    """
    X = np.array(
        [
            ["Red", 10],
            ["Blue", 20],
            ["Red", 30],
        ],
        dtype=object,
    )

    encoder = OneHotEncoder()
    X_out = encoder.fit_transform(X)

    assert X_out.shape == (3, 3)

    assert (X_out[0][:2] == [1, 0]).all() or (X_out[0][:2] == [0, 1]).all()

    print("test_one_hot_encoder passed!!")


if __name__ == "__main__":
    test_load_csv_basic()
    test_missing_values()
    test_imputer()
    test_standard_scaler()
    test_one_hot_encoder()
