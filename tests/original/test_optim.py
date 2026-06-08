import numpy as np
from numcompute.optim import grad, jacobian


def test_grad_vector():
    """
    Test gradient for vector input.
    f(x, y) = x^2 + y^2 → grad = [2x, 2y]
    """

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = np.array([2.0, 3.0])
    g = grad(f, x)

    assert np.allclose(g, [4, 6], atol=1e-3)
    print("test_grad_vector passed!!")


def test_grad_scalar():
    """
    Test gradient for scalar input.
    f(x) = x^2 → grad = 2x
    """

    def f(x):
        return x**2

    x = 3.0
    g = grad(f, x)

    assert np.allclose(g, 6.0, atol=1e-3)
    print("test_grad_scalar passed!!")


def test_grad_forward():
    """
    Test forward difference method.
    f(x, y) = x^2 + y^2 → grad = [2x, 2y]
    """

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = np.array([2.0, 3.0])
    g = grad(f, x, method="forward")

    assert np.allclose(g, [4, 6], atol=1e-2)
    print("test_grad_forward passed!!")


def test_jacobian():
    """
    Test Jacobian for vector-valued function.
    f(x, y) = [x^2, y^2]
    Jacobian = [[2x, 0],
                [0, 2y]]
    """

    def f(x):
        return np.array([x[0] ** 2, x[1] ** 2])

    x = np.array([2.0, 3.0])
    J = jacobian(f, x)

    expected = np.array([[4, 0], [0, 6]])

    assert np.allclose(J, expected, atol=1e-3)
    print("test_jacobian passed!!")


if __name__ == "__main__":
    test_grad_vector()
    test_grad_scalar()
    test_grad_forward()
    test_jacobian()
