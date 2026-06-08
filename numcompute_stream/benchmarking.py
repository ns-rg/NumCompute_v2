import time
import numpy as np


def time_function(func, *args, repeat=5):
    """
    Measure average execution time.

    Parameters:
    - func: function to time
    - args: arguments to pass to the function
    - repeat: number of times to repeat for averaging

    Returns:
    - Average execution time in seconds
    """
    times = []

    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times)


def benchmark(name, func_vectorized, func_loop, *args):
    """
    Compare vectorized vs loop implementation.

    Parameters:
    - name: str, name of the benchmark
    - func_vectorized: function implementing vectorized version
    - func_loop: function implementing loop version
    - args: arguments to pass to both functions

    Returns:
    - dict with benchmark results
    """
    t_vec = time_function(func_vectorized, *args)
    t_loop = time_function(func_loop, *args)

    speedup = t_loop / t_vec if t_vec > 0 else np.inf

    print(f"\n{name} Benchmark")
    print("-" * 30)
    print(f"Vectorized: {t_vec:.6f} sec")
    print(f"Loop:       {t_loop:.6f} sec")
    print(f"Speedup:    {speedup:.2f}x")

    return {"name": name, "vectorized": t_vec, "loop": t_loop, "speedup": speedup}
