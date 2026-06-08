import numpy as np
from numcompute_stream.tree import DecisionTreeClassifier


class EnsembleClassifier:
    """
    An ensemble classifier that combines multiple decision trees using
    either Bagging or Random Forest method.

    Bagging: each tree is trained on a random bootstrap sample of the data.
    Random Forest: same as Bagging but also restricts the number of features
    each tree can consider at each split.

    Supports incremental learning via partial_fit by accumulating all data
    seen so far and refitting all trees on bootstrap samples of that data.

    Example usage:
    clf = EnsembleClassifier(n_trees=10, method="random_forest")
    clf.partial_fit(X_chunk, y_chunk)
    predictions = clf.predict(X_new)
    """

    def __init__(self, n_trees=10, method="random_forest", max_depth=None,
                 min_samples_split=2, max_features=None, criterion="gini"):
        """
        Initialise EnsembleClassifier.
        Parameters:
        - n_trees: int. Number of trees in the ensemble.
        - method: str. Either "bagging" or "random_forest".
        - max_depth: int or None. Maximum depth of each tree.
        - min_samples_split: int. Minimum samples required to split a node.
        - max_features: int or None. Number of features per split.
          If None and method is "random_forest", defaults to sqrt(n_features).
          If None and method is "bagging", all features are used.
        - criterion: str. Split criterion, either "gini" or "entropy".
        Raises:
        - ValueError: If method is not "bagging" or "random_forest".
        - ValueError: If n_trees is less than 1.
        """
        if method not in ("bagging", "random_forest"):
            raise ValueError("method must be 'bagging' or 'random_forest'.")

        if n_trees < 1:
            raise ValueError("n_trees must be at least 1.")

        self.n_trees = n_trees
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion

        self.trees_ = []
        self._X_seen = None
        self._y_seen = None

    def _get_max_features(self, n_features):
        """
        Resolve the number of features to use per split.
        Parameters:
        - n_features: int, total number of features in the dataset.
        Returns:
        - int: number of features to consider at each split.
        """
        if self.max_features is not None:
            return self.max_features

        if self.method == "random_forest":
            return max(1, int(np.sqrt(n_features)))

        return n_features

    def fit(self, X, y):
        """
        Fit all trees on bootstrap samples of the full dataset.
        Parameters:
        - X: array-like of shape (n_samples, n_features).
        - y: array-like of shape (n_samples,).
        Returns:
        - self
        Raises:
        - ValueError: If X and y have different numbers of samples.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        self.trees_ = []

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Incrementally update the ensemble by accumulating the new chunk
        and refitting all trees on bootstrap samples of all data seen so far.
        Parameters:
        - X_chunk: array-like of shape (n_samples, n_features).
        - y_chunk: array-like of shape (n_samples,).
        Returns:
        - self
        Raises:
        - ValueError: If X_chunk and y_chunk have different numbers of samples.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        if X_chunk.shape[0] != y_chunk.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of samples.")

        if self._X_seen is None:
            self._X_seen = X_chunk
            self._y_seen = y_chunk
        else:
            self._X_seen = np.vstack([self._X_seen, X_chunk])
            self._y_seen = np.concatenate([self._y_seen, y_chunk])

        return self.fit(self._X_seen, self._y_seen)

    def predict(self, X):
        """
        Predict class labels by majority vote across all trees.
        Parameters:
        - X: array-like of shape (n_samples, n_features).
        Returns:
        - np.ndarray of shape (n_samples,) with predicted class labels.
        Raises:
        - ValueError: If no trees have been fitted yet.
        """
        if len(self.trees_) == 0:
            raise ValueError("Ensemble must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)

        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        result = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            classes, counts = np.unique(votes, return_counts=True)
            result.append(classes[np.argmax(counts)])

        return np.array(result)

    def score(self, X, y):
        """
        Compute accuracy of predictions on X against true labels y.
        Parameters:
        - X: array-like of shape (n_samples, n_features).
        - y: array-like of shape (n_samples,).
        Returns:
        - float: accuracy score between 0 and 1.
        Raises:
        - ValueError: If no trees have been fitted yet.
        """
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)