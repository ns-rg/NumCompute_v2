class Pipeline:
    """
    A simple pipeline to chain together multiple steps.
    Each step is a tuple of (name, transformer), where transformer is an object
    that implements fit and/or transform methods.
    Example usage:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    X_transformed = pipeline.transform(X_test)
    """

    def __init__(self, steps):
        """
        Initialize the pipeline with a list of steps.
        Parameters:
        - steps: list of tuples (name, transformer)

        Raises:
        - ValueError: If steps is not a list of tuples or if any transformer does not have fit/transform methods.
        """
        self.steps = steps

    def fit(self, X, y=None):
        """
        Fit each step sequentially.
        Parameters:
        - X: input data
        - y: target labels (optional, only needed for supervised steps)

        Raises:
        - ValueError: If any step does not have fit or transform methods.
        """
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X = step.fit_transform(X)
            else:
                if hasattr(step, "fit"):
                    step.fit(X, y)
                if hasattr(step, "transform"):
                    X = step.transform(X)
        return self

    def transform(self, X):
        """
        Apply transformations sequentially.
        Parameters:
        - X: input data

        Raises:
        - ValueError: If any step does not have a transform method.
        """
        for name, step in self.steps:
            if hasattr(step, "transform"):
                X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one pass (no duplicate work).
        Parameters:
        - X: input data
        - y: target labels (optional, only needed for supervised steps)

        Raises:
        - ValueError: If any step does not have fit or transform methods.
        """
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X = step.fit_transform(X)
            else:
                if hasattr(step, "fit"):
                    step.fit(X, y)
                if hasattr(step, "transform"):
                    X = step.transform(X)
        return X