import numpy as np


class DecisionTreeClassifier:
    """
    A depth-limited decision tree classifier built from scratch.
    Supports Gini impurity and entropy as split criteria.
    Can be grown incrementally via partial_fit by refitting on
    all accumulated data seen so far.

    Example usage:
    tree = DecisionTreeClassifier(max_depth=3, criterion="gini")
    tree.partial_fit(X_chunk, y_chunk)
    predictions = tree.predict(X_new)
    """

    def __init__(self, max_depth=None, criterion="gini", min_samples_split=2, max_features=None):
        """
        Initialise DecisionTreeClassifier.
        Parameters:
        - max_depth: int or None. Maximum depth of the tree. None means unlimited.
        - criterion: str. Split quality measure, either "gini" or "entropy".
        - min_samples_split: int. Minimum samples required to split a node.
        - max_features: int or None. Number of features to consider at each split.
          None means all features are considered.
        Raises:
        - ValueError: If criterion is not "gini" or "entropy".
        """
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self._tree = None
        self._X_seen = None
        self._y_seen = None

    def _impurity(self, y):
        """
        Compute impurity of a node using the chosen criterion.
        Parameters:
        - y: np.ndarray of class labels at this node.
        Returns:
        - float: impurity value.
        """
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        else:
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        Parameters:
        - X: np.ndarray of shape (n_samples, n_features).
        - y: np.ndarray of shape (n_samples,).
        Returns:
        - best_feature: int, index of the best feature to split on.
        - best_threshold: float, threshold value for the split.
        Returns (None, None) if no valid split is found.
        """
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        parent_impurity = self._impurity(y)

        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)

        for j in feature_indices:
            thresholds = np.unique(X[:, j])

            for threshold in thresholds:
                left_mask = X[:, j] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                weighted_impurity = (
                    n_left / n_samples * left_impurity +
                    n_right / n_samples * right_impurity
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        Parameters:
        - X: np.ndarray of shape (n_samples, n_features).
        - y: np.ndarray of shape (n_samples,).
        - depth: int, current depth of the tree.
        Returns:
        - dict representing the tree node, with keys:
          'leaf' and 'label' for leaf nodes, or
          'feature', 'threshold', 'left', 'right' for internal nodes.
        """
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        if len(classes) == 1:
            return {"leaf": True, "label": majority_class}

        if self.max_depth is not None and depth >= self.max_depth:
            return {"leaf": True, "label": majority_class}

        if len(y) < self.min_samples_split:
            return {"leaf": True, "label": majority_class}

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return {"leaf": True, "label": majority_class}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _predict_sample(self, node, x):
        """
        Traverse the tree to predict the class for a single sample.
        Parameters:
        - node: dict, current tree node.
        - x: np.ndarray of shape (n_features,).
        Returns:
        - predicted class label.
        """
        if node["leaf"]:
            return node["label"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(node["left"], x)
        else:
            return self._predict_sample(node["right"], x)

    def fit(self, X, y):
        """
        Fit the decision tree on the full dataset.
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

        self._tree = self._build_tree(X, y, depth=0)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Incrementally update the tree by accumulating the new chunk
        and refitting the tree on all data seen so far.
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
        Predict class labels for samples in X.
        Parameters:
        - X: array-like of shape (n_samples, n_features).
        Returns:
        - np.ndarray of shape (n_samples,) with predicted class labels.
        Raises:
        - ValueError: If the tree has not been fitted yet.
        """
        if self._tree is None:
            raise ValueError("Tree must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)
        return np.array([self._predict_sample(self._tree, x) for x in X])

    def score(self, X, y):
        """
        Compute accuracy of predictions on X against true labels y.
        Parameters:
        - X: array-like of shape (n_samples, n_features).
        - y: array-like of shape (n_samples,).
        Returns:
        - float: accuracy score between 0 and 1.
        Raises:
        - ValueError: If the tree has not been fitted yet.
        """
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)