import numpy as np
from numcompute.utils import sigmoid, softmax, logsumexp

def test_sigmoid():
    """Test the sigmoid function with a simple input."""
    assert np.isclose(sigmoid(0), 0.5)
    print("Sigmoid test passed!!")


def test_softmax_sum():
    """Test the softmax function to ensure the probabilities sum to 1."""
    x = np.array([1, 2, 3])
    s = softmax(x)
    assert np.isclose(np.sum(s), 1.0)
    print("Softmax sum test passed!!")


def test_logsumexp():
    """Test the logsumexp function with a simple input."""
    x = np.array([1, 2, 3])
    result = logsumexp(x)
    expected = np.log(np.sum(np.exp(x)))
    assert np.isclose(result, expected)
    print("LogSumExp test passed!!")


if __name__ == "__main__":
    test_sigmoid()
    test_softmax_sum()
    test_logsumexp()