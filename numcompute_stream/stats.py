import numpy as np
from numcompute.rank import percentile

def mean(X, axis=None):
    """
    Compute the mean of an array, ignoring NaNs.
    -Parameters:
    X: array-like
    axis: int or None, optional.
        Axis along which the means are computed. The default is to compute the mean of the flattened array.
    -Returns:
    mean: float or np.ndarray
        The mean of the array elements along the specified axis, ignoring NaNs.
    -Raises:
        ValueError: If X is not an array-like structure or if axis is out of bounds.
        ZeroDivisionError: If all elements are NaN, resulting in a count of zero.
    """
    X = np.asarray(X, dtype=float)
    return np.nanmean(X, axis=axis)


def mean_loop(X):
    """
    Compute the mean of an array, ignoring NaNs, using a loop.
    -Parameters:
    X: array-like
    -Returns:
    mean: float
    The mean of the array elements, ignoring NaNs.
    -Raises:
    ZeroDivisionError: If all elements are NaN, resulting in a count of zero.
    """
    X = np.asarray(X, dtype=float)
    total = 0.0
    count = 0

    if len(X) == 0:
        raise ZeroDivisionError("Mean of empty array is undefined")

    for x in X:
        if not np.isnan(x):
            total += x
            count += 1

    if count == 0:
        return np.nan

    return total / count


def std(X, axis=None):
    """
    Compute the standard deviation of an array, ignoring NaNs.
    -Parameters:
    X: array-like
    axis: int or None, optional.
        Axis along which the standard deviations are computed. The default is to compute the standard deviation of the flattened array.
    -Returns:
    std: float or np.ndarray
        The standard deviation of the array elements along the specified axis, ignoring NaNs.
    -Raises:
        ValueError: If X is not an array-like structure or if axis is out of bounds.
        ZeroDivisionError: If all elements are NaN, resulting in a count of zero.
    """
    X = np.asarray(X, dtype=float)
    return np.nanstd(X, axis=axis)


def std_loop(X):
    """
    Compute the standard deviation of an array, ignoring NaNs, using a loop.
    -Parameters:
    X: array-like
    -Returns:
    std: float
    The standard deviation of the array elements, ignoring NaNs.
    -Raises:
    ZeroDivisionError: If all elements are NaN, resulting in a count of zero.
    """
    X = np.asarray(X, dtype=float)
    total = 0.0
    count = 0

    for x in X:
        if not np.isnan(x):
            total += x
            count += 1

    if count == 0:
        return np.nan

    mean = total / count

    sq_sum = 0.0

    for x in X:
        if not np.isnan(x):
            sq_sum += (x - mean) ** 2

    return np.sqrt(sq_sum / count)


def min(X, axis=None):
    """Compute the minimum of an array, ignoring NaNs.
    -Parameters:
    X: array-like
    axis: int or None, optional.
        Axis along which the minima are computed. The default is to compute the minimum of the flattened array.
    -Returns:
    min: float or np.ndarray
        The minimum of the array elements along the specified axis, ignoring NaNs."""
    X = np.asarray(X, dtype=float)
    return np.nanmin(X, axis=axis)


def max(X, axis=None):
    """
    Compute the maximum of an array, ignoring NaNs.
    -Parameters:
    X: array-like
    axis: int or None, optional.
        Axis along which the maxima are computed. The default is to compute the maximum of the flattened array.
    -Returns:
    max: float or np.ndarray
        The maximum of the array elements along the specified axis, ignoring NaNs.
    """
    X = np.asarray(X, dtype=float)
    return np.nanmax(X, axis=axis)


def histogram(X, bins=10):
    """
    Compute the histogram of an array, ignoring NaNs.
    -Parameters:
    X: array-like
    bins: int or sequence of scalars, optional.
        If bins is an integer, it defines the number of equal-width bins in the range of X. If bins is a sequence, it defines the bin edges, including the rightmost edge.
    -Returns:
    counts: np.ndarray
        The number of occurrences of each bin.
    edges: np.ndarray
        The edges of the bins.
    """
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X)]

    counts, edges = np.histogram(X, bins=bins)
    return counts, edges


def quantile(X, q):
    """
    Compute the q-th quantile of an array, ignoring NaNs.
    -Parameters:
    X: array-like
    q: float or array-like
        Quantile(s) to compute, which must be between 0 and 1 inclusive.
    -Returns:
    quantiles: float or np.ndarray
        The computed quantile(s) of the array elements, ignoring NaNs.
    """
    X = np.asarray(X, dtype=float)

    if np.isscalar(q):
        return percentile(X, q * 100)
    else:
        return np.array([percentile(X, qi * 100) for qi in q])