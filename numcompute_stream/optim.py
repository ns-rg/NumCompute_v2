import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """
    Compute gradient for scalar-valued function.

    Parameters:
    -f (callable): The function for which to compute the gradient.
    -x (array-like): The point at which to compute the gradient.
    -h (float): The step size for finite difference approximation.
    -method (str): The finite difference method to use ("central" or "forward").

    Returns:
    -numpy.ndarray: The computed gradient.

    Raises:
    -ValueError: If an invalid method is specified.
    """

    x = np.asarray(x, dtype=float)

    # Scalar case
    if x.ndim == 0:
        if method == "central":
            return (f(x + h) - f(x - h)) / (2 * h)
        elif method == "forward":
            return (f(x + h) - f(x)) / h

    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        if method == "central":
            x1[i] += h
            x2[i] -= h
            grad[i] = (f(x1) - f(x2)) / (2 * h)

        elif method == "forward":
            x1[i] += h
            grad[i] = (f(x1) - f(x)) / h

        else:
            raise ValueError("Invalid method")

    return grad


def jacobian(f, x, h=1e-5):
    """
    Compute Jacobian for vector-valued function.

    Parameters:
    -f (callable): The function for which to compute the Jacobian.
    -x (array-like): The point at which to compute the Jacobian.
    -h (float): The step size for finite difference approximation.

    Returns:
    -numpy.ndarray: The computed Jacobian.
    """

    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(f(x))

    jac = np.zeros((f0.size, x.size), dtype=float)

    for i in range(x.size):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += h
        x2[i] -= h

        jac[:, i] = (np.atleast_1d(f(x1)) - np.atleast_1d(f(x2))) / (2 * h)

    return jac
