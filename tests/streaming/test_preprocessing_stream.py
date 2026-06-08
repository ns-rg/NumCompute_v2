import numpy as np
from numcompute_stream.preprocessing import Imputer, StandardScaler, OneHotEncoder


def test_standard_scaler_partial_fit_mean():
    """Test that StandardScaler partial_fit updates mean correctly over chunks."""
    scaler = StandardScaler()
    scaler.fit(np.array([["1.0", "2.0"], ["3.0", "4.0"]], dtype=object))
    scaler.partial_fit(np.array([["5.0", "6.0"]], dtype=object))
    assert scaler.mean_[0] is not None
    assert scaler.mean_[1] is not None
    print("test_standard_scaler_partial_fit_mean passed!!")


def test_standard_scaler_partial_fit_std():
    """Test that StandardScaler partial_fit updates std correctly over chunks."""
    scaler = StandardScaler()
    scaler.fit(np.array([["1.0"], ["2.0"], ["3.0"]], dtype=object))
    scaler.partial_fit(np.array([["4.0"], ["5.0"]], dtype=object))
    assert scaler.std_[0] > 0
    print("test_standard_scaler_partial_fit_std passed!!")


def test_standard_scaler_partial_fit_before_fit():
    """Test that partial_fit works even without calling fit first."""
    scaler = StandardScaler()
    scaler.partial_fit(np.array([["1.0"], ["2.0"], ["3.0"]], dtype=object))
    assert scaler.mean_[0] is not None
    print("test_standard_scaler_partial_fit_before_fit passed!!")


def test_standard_scaler_transform_after_partial_fit():
    """Test that transform works correctly after partial_fit."""
    scaler = StandardScaler()
    scaler.fit(np.array([["1.0"], ["2.0"], ["3.0"]], dtype=object))
    scaler.partial_fit(np.array([["4.0"], ["5.0"]], dtype=object))
    result = scaler.transform(np.array([["3.0"]], dtype=object))
    assert result.shape == (1, 1)
    print("test_standard_scaler_transform_after_partial_fit passed!!")


def test_standard_scaler_partial_fit_non_numeric():
    """Test that partial_fit ignores non-numeric columns."""
    scaler = StandardScaler()
    scaler.fit(np.array([["cat", "1.0"], ["dog", "2.0"]], dtype=object))
    scaler.partial_fit(np.array([["bird", "3.0"]], dtype=object))
    assert scaler.numeric_mask_[0] == False
    assert scaler.numeric_mask_[1] == True
    print("test_standard_scaler_partial_fit_non_numeric passed!!")


def test_imputer_partial_fit_mean():
    """Test that Imputer partial_fit updates running mean correctly."""
    imputer = Imputer(strategy="mean")
    imputer.fit(np.array([["1.0"], ["2.0"]], dtype=object))
    imputer.partial_fit(np.array([["3.0"], ["4.0"]], dtype=object))
    assert imputer.statistics_[0] > 1.0
    print("test_imputer_partial_fit_mean passed!!")


def test_imputer_partial_fit_before_fit():
    """Test that Imputer partial_fit works without calling fit first."""
    imputer = Imputer(strategy="mean")
    imputer.partial_fit(np.array([["1.0"], ["3.0"]], dtype=object))
    assert imputer.statistics_[0] is not None
    print("test_imputer_partial_fit_before_fit passed!!")


def test_imputer_partial_fit_categorical_mode():
    """Test that Imputer partial_fit updates mode for categorical columns."""
    imputer = Imputer()
    imputer.fit(np.array([["cat"], ["cat"], ["dog"]], dtype=object))
    imputer.partial_fit(np.array([["dog"], ["dog"]], dtype=object))
    assert imputer.statistics_[0] == "dog"
    print("test_imputer_partial_fit_categorical_mode passed!!")


def test_imputer_transform_after_partial_fit():
    """Test that transform fills missing values using updated statistics."""
    imputer = Imputer(strategy="mean")
    imputer.fit(np.array([["2.0"], ["4.0"]], dtype=object))
    imputer.partial_fit(np.array([["6.0"]], dtype=object))
    result = imputer.transform(np.array([[None]], dtype=object))
    assert result[0, 0] is not None
    print("test_imputer_transform_after_partial_fit passed!!")


def test_imputer_partial_fit_nan_ignored():
    """Test that NaN values are ignored during partial_fit."""
    imputer = Imputer(strategy="mean")
    imputer.fit(np.array([["2.0"], ["4.0"]], dtype=object))
    imputer.partial_fit(np.array([[np.nan]], dtype=object))
    assert np.isclose(imputer.statistics_[0], 3.0)
    print("test_imputer_partial_fit_nan_ignored passed!!")


def test_onehotencoder_partial_fit_new_category():
    """Test that OneHotEncoder partial_fit adds new unseen categories."""
    enc = OneHotEncoder()
    enc.fit(np.array([["cat"], ["dog"]], dtype=object))
    enc.partial_fit(np.array([["bird"]], dtype=object))
    assert "bird" in enc.categories_[0]
    print("test_onehotencoder_partial_fit_new_category passed!!")


def test_onehotencoder_partial_fit_no_duplicate():
    """Test that partial_fit does not add duplicate categories."""
    enc = OneHotEncoder()
    enc.fit(np.array([["cat"], ["dog"]], dtype=object))
    enc.partial_fit(np.array([["cat"]], dtype=object))
    assert enc.categories_[0].count("cat") == 1
    print("test_onehotencoder_partial_fit_no_duplicate passed!!")


def test_onehotencoder_partial_fit_before_fit():
    """Test that partial_fit works without calling fit first."""
    enc = OneHotEncoder()
    enc.partial_fit(np.array([["cat"], ["dog"]], dtype=object))
    assert "cat" in enc.categories_[0]
    print("test_onehotencoder_partial_fit_before_fit passed!!")


def test_onehotencoder_transform_after_partial_fit():
    """Test that transform uses updated categories after partial_fit."""
    enc = OneHotEncoder()
    enc.fit(np.array([["cat"], ["dog"]], dtype=object))
    enc.partial_fit(np.array([["bird"]], dtype=object))
    result = enc.transform(np.array([["bird"]], dtype=object))
    assert result.shape[1] == 3
    print("test_onehotencoder_transform_after_partial_fit passed!!")


def test_onehotencoder_partial_fit_numeric_ignored():
    """Test that partial_fit does not affect numeric columns."""
    enc = OneHotEncoder()
    enc.fit(np.array([["1.0"], ["2.0"]], dtype=object))
    enc.partial_fit(np.array([["3.0"]], dtype=object))
    assert enc.categorical_mask_[0] == False
    print("test_onehotencoder_partial_fit_numeric_ignored passed!!")


if __name__ == "__main__":
    test_standard_scaler_partial_fit_mean()
    test_standard_scaler_partial_fit_std()
    test_standard_scaler_partial_fit_before_fit()
    test_standard_scaler_transform_after_partial_fit()
    test_standard_scaler_partial_fit_non_numeric()
    test_imputer_partial_fit_mean()
    test_imputer_partial_fit_before_fit()
    test_imputer_partial_fit_categorical_mode()
    test_imputer_transform_after_partial_fit()
    test_imputer_partial_fit_nan_ignored()
    test_onehotencoder_partial_fit_new_category()
    test_onehotencoder_partial_fit_no_duplicate()
    test_onehotencoder_partial_fit_before_fit()
    test_onehotencoder_transform_after_partial_fit()
    test_onehotencoder_partial_fit_numeric_ignored()