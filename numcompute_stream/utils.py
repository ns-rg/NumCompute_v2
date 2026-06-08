import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function of x.
    -Parameters:
    x: array-like
    -Returns:
    sigmoid_x: array-like
        The sigmoid of the input x.
    """
    x = np.asarray(x)
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Compute the softmax function of x along the specified axis.
    -Parameters:
    x: array-like
    axis: int, optional
        Axis along which the softmax is computed. The default is -1 (the last axis).
    -Returns:
    softmax_x: array-like
        The softmax of the input x along the specified axis.
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_loop(x):
    """
    Compute the softmax function of x using a loop.
    -Parameters:
    x: array-like
    -Returns:
    softmax_x: array-like
        The softmax of the input x.
    -Raises:
    ValueError: If the input array is empty, since softmax is undefined for empty arrays.
    """
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        raise ValueError("softmax is undefined for empty array")

    # find max (for numerical stability)
    max_val = x[0]
    for i in range(len(x)):
        if x[i] > max_val:
            max_val = x[i]

    # compute exponentials (shifted)
    exps = []
    sum_exp = 0.0

    for i in range(len(x)):
        if not np.isnan(x[i]):
            e = np.exp(x[i] - max_val)
        else:
            e = np.nan

        exps.append(e)
        if not np.isnan(e):
            sum_exp += e

    # normalize
    result = []
    for e in exps:
        if np.isnan(e):
            result.append(np.nan)
        else:
            result.append(e / sum_exp)

    return np.array(result)


def logsumexp(x, axis=None):
    """
    Compute the log-sum-exp of x along the specified axis.
    -Parameters:
    x: array-like
    axis: int or None, optional
        Axis along which the log-sum-exp is computed. The default is to compute the log-sum-exp of the flattened array.
    -Returns:
    logsumexp_x: array-like
        The log-sum-exp of the input x along the specified axis.
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return np.squeeze(result)


def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two points a and b.
    -Parameters:
    a: array-like
        First point.
    b: array-like
        Second point.
    -Returns:
    distance: float
        The Euclidean distance between points a and b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a - b) ** 2))


def batch_iterator(X, batch_size):
    """
    Generate batches of data from the input array X.
    -Parameters:
    X: array-like
        Input data to be batched.
    batch_size: int
        The size of each batch.
    -Yields:
    batch: array-like
        A batch of data from the input array X.
    """
    n = len(X)
    for i in range(0, n, batch_size):
        yield X[i : i + batch_size]