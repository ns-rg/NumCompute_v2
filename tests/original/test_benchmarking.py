import numpy as np
from numcompute.benchmarking import benchmark

def loop_sum(x):
    """
    A simple loop-based sum implementation.
    This is intentionally inefficient for benchmarking purposes.
    """
    total = 0
    for i in x:
        total += i
    return total


def vectorized_sum(x):
    """
    A vectorized sum implementation using NumPy.
    This should be much faster than the loop-based version.
    """
    return np.sum(x)


def test_benchmark():
    """
    Test the benchmarking function by comparing the performance of a loop-based sum and a vectorized sum.
    We expect the vectorized version to be significantly faster than the loop-based version.
    """
    x = np.random.rand(1000000)

    result = benchmark("Sum", vectorized_sum, loop_sum, x)

    assert result["speedup"] > 1
    print("test_benchmark passed!!")


if __name__ == "__main__":
    test_benchmark()
