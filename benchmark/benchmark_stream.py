import numpy as np
from numcompute_stream.benchmarking import benchmark, time_function
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier


def generate_data(n_samples, n_features, seed=42):
    """
    Generate a random classification dataset.
    Parameters:
    - n_samples: int, number of samples.
    - n_features: int, number of features.
    - seed: int, random seed for reproducibility.
    Returns:
    - X: np.ndarray of shape (n_samples, n_features).
    - y: np.ndarray of shape (n_samples,) with binary labels.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def benchmark_tree_vs_ensemble():
    """
    Compare fit time between a single DecisionTreeClassifier and
    an EnsembleClassifier on the same dataset.
    """
    X, y = generate_data(n_samples=500, n_features=5)

    tree = DecisionTreeClassifier(max_depth=5)
    ensemble = EnsembleClassifier(n_trees=10, method="random_forest", max_depth=5)

    def fit_tree():
        t = DecisionTreeClassifier(max_depth=5)
        t.fit(X, y)

    def fit_ensemble():
        e = EnsembleClassifier(n_trees=10, method="random_forest", max_depth=5)
        e.fit(X, y)

    tree_time = time_function(fit_tree, repeat=5)
    ensemble_time = time_function(fit_ensemble, repeat=5)

    print("=== Tree vs Ensemble Fit Time ===")
    print(f"DecisionTreeClassifier  : {tree_time:.4f}s avg over 5 runs")
    print(f"EnsembleClassifier      : {ensemble_time:.4f}s avg over 5 runs")
    print(f"Speedup ratio           : {ensemble_time / tree_time:.2f}x slower")
    print()


def benchmark_partial_fit_vs_fit():
    """
    Compare cumulative time of repeated partial_fit calls vs a single
    fit call on the full dataset for DecisionTreeClassifier.
    """
    X, y = generate_data(n_samples=500, n_features=5)
    chunk_size = 100
    chunks_X = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
    chunks_y = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]

    def fit_full():
        t = DecisionTreeClassifier(max_depth=5)
        t.fit(X, y)

    def fit_partial():
        t = DecisionTreeClassifier(max_depth=5)
        for cx, cy in zip(chunks_X, chunks_y):
            t.partial_fit(cx, cy)

    full_time = time_function(fit_full, repeat=5)
    partial_time = time_function(fit_partial, repeat=5)

    print("=== fit() vs partial_fit() over chunks ===")
    print(f"fit() on full data      : {full_time:.4f}s avg over 5 runs")
    print(f"partial_fit() in chunks : {partial_time:.4f}s avg over 5 runs")
    print(f"Overhead ratio          : {partial_time / full_time:.2f}x")
    print()


def benchmark_bagging_vs_random_forest():
    """
    Compare fit time between Bagging and Random Forest ensemble methods.
    """
    X, y = generate_data(n_samples=500, n_features=10)

    def fit_bagging():
        e = EnsembleClassifier(n_trees=10, method="bagging", max_depth=5)
        e.fit(X, y)

    def fit_rf():
        e = EnsembleClassifier(n_trees=10, method="random_forest", max_depth=5)
        e.fit(X, y)

    bagging_time = time_function(fit_bagging, repeat=5)
    rf_time = time_function(fit_rf, repeat=5)

    print("=== Bagging vs Random Forest Fit Time ===")
    print(f"Bagging       : {bagging_time:.4f}s avg over 5 runs")
    print(f"Random Forest : {rf_time:.4f}s avg over 5 runs")
    print()


if __name__ == "__main__":
    benchmark_tree_vs_ensemble()
    benchmark_partial_fit_vs_fit()
    benchmark_bagging_vs_random_forest()