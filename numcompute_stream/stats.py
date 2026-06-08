import numpy as np
from numcompute_stream.rank import percentile

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
    
class StreamStats:
    """
    Compute statistics incrementally over chunks of data using Welford's
    online algorithm for numerical stability.

    Supports per-feature mean, variance, min, max, histogram, and quantile
    estimates updated chunk by chunk.

    Optionally operates in sliding-window mode, keeping only the last
    window_size samples per feature.

    Example usage:
    ss = StreamStats(n_features=3)
    ss.update_stats(X_chunk)
    print(ss.result())
    """

    def __init__(self, n_features, window_size=None, bins=10):
        """
        Initialise StreamStats.
        Parameters:
        - n_features: int, number of features (columns) expected in each chunk
        - window_size: int or None. If set, only the last window_size samples
          are retained for quantile and histogram estimates.
        - bins: int, number of histogram bins.
        Raises:
        - ValueError: If n_features < 1 or window_size < 1.
        """
        if n_features < 1:
            raise ValueError("n_features must be at least 1.")
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.n_features = n_features
        self.window_size = window_size
        self.bins = bins

        self._n = np.zeros(n_features, dtype=float)
        self._mean = np.zeros(n_features, dtype=float)
        self._M2 = np.zeros(n_features, dtype=float)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)
        self._buffer = [[] for _ in range(n_features)]

    def update_stats(self, X_chunk):
        """
        Ingest a new chunk and update all running statistics.
        Parameters:
        - X_chunk: array-like of shape (n_samples, n_features) or (n_samples,)
          for single-feature data.
        Returns:
        - self
        Raises:
        - ValueError: If chunk has wrong number of features or is empty.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)

        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk must not be empty.")

        if X_chunk.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_chunk.shape[1]}."
            )

        for j in range(self.n_features):
            col = X_chunk[:, j]
            col = col[~np.isnan(col)]

            if col.size == 0:
                continue

            for x in col:
                self._n[j] += 1
                delta = x - self._mean[j]
                self._mean[j] += delta / self._n[j]
                delta2 = x - self._mean[j]
                self._M2[j] += delta * delta2

            self._min[j] = np.minimum(self._min[j], np.min(col))
            self._max[j] = np.maximum(self._max[j], np.max(col))

            if self.window_size is not None:
                self._buffer[j].extend(col.tolist())
                self._buffer[j] = self._buffer[j][-self.window_size:]
            else:
                self._buffer[j].extend(col.tolist())

        return self

    def result(self):
        """
        Return a dict of current streaming statistics per feature.
        Returns:
        - dict with keys: 'mean', 'variance', 'std', 'min', 'max',
          'quantiles' (0.25, 0.5, 0.75), 'histogram' (counts, edges).
          Each value is an np.ndarray of length n_features.
        Raises:
        - ValueError: If no data has been ingested yet.
        """
        if np.all(self._n == 0):
            raise ValueError("No data ingested yet. Call update_stats() first.")

        variance = np.where(
            self._n > 1, self._M2 / (self._n - 1), 0.0
        )

        quantiles = np.zeros((self.n_features, 3))
        histograms = []

        for j in range(self.n_features):
            buf = np.array(self._buffer[j], dtype=float)
            if buf.size > 0:
                quantiles[j] = np.array([
                    percentile(buf, 25),
                    percentile(buf, 50),
                    percentile(buf, 75),
                ])
                counts, edges = np.histogram(buf, bins=self.bins)
                histograms.append((counts, edges))
            else:
                quantiles[j] = np.array([np.nan, np.nan, np.nan])
                histograms.append((np.array([]), np.array([])))

        return {
            "mean": self._mean.copy(),
            "variance": variance,
            "std": np.sqrt(variance),
            "min": self._min.copy(),
            "max": self._max.copy(),
            "quantiles": quantiles,
            "histograms": histograms,
        }

    def reset(self):
        """
        Reset all accumulated statistics.
        Returns:
        - self
        """
        self._n = np.zeros(self.n_features, dtype=float)
        self._mean = np.zeros(self.n_features, dtype=float)
        self._M2 = np.zeros(self.n_features, dtype=float)
        self._min = np.full(self.n_features, np.inf)
        self._max = np.full(self.n_features, -np.inf)
        self._buffer = [[] for _ in range(self.n_features)]
        return self