import numpy as np
import sys
from numcompute_stream.metrics import accuracy, precision, recall, f1_score


class StreamTrainer:
    """
    Manages incremental training of a model through a pipeline over
    streaming data chunks.

    Logs per-chunk metrics, memory footprint of the model, and tracks
    cumulative accuracy across all chunks seen so far.

    Example usage:
    trainer = StreamTrainer(model=EnsembleClassifier(), pipeline=pipe)
    trainer.fit_chunk(X_chunk, y_chunk)
    print(trainer.get_logs())
    """

    def __init__(self, model, pipeline=None):
        """
        Initialise StreamTrainer.
        Parameters:
        - model: a classifier with partial_fit() and predict() methods.
        - pipeline: a Pipeline instance or None. If provided, data is
          transformed through the pipeline before being passed to the model.
        Raises:
        - ValueError: If model does not have partial_fit or predict methods.
        """
        if not hasattr(model, "partial_fit"):
            raise ValueError("model must have a partial_fit() method.")

        if not hasattr(model, "predict"):
            raise ValueError("model must have a predict() method.")

        self.model = model
        self.pipeline = pipeline

        self._logs = []
        self._chunk_index = 0
        self._y_true_all = []
        self._y_pred_all = []

    def _transform(self, X):
        """
        Apply pipeline transformations to X if a pipeline is set.
        Parameters:
        - X: array-like of shape (n_samples, n_features).
        Returns:
        - X transformed through the pipeline, or X unchanged if no pipeline.
        """
        if self.pipeline is not None:
            return self.pipeline.transform(X)
        return X

    def _memory_footprint(self):
        """
        Estimate memory footprint of the model in bytes using sys.getsizeof.
        Returns:
        - int: memory size of the model object in bytes.
        """
        return sys.getsizeof(self.model)

    def fit_chunk(self, X, y):
        """
        Fit the model incrementally on a new chunk of data.
        Transforms data through the pipeline if one is set, then calls
        partial_fit on the pipeline (if set) or directly on the model.
        Logs per-chunk metrics after fitting.
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

        if self.pipeline is not None:
            self.pipeline.partial_fit(X, y)
            X_transformed = self._transform(X)
        else:
            X_transformed = X

        self.model.partial_fit(X_transformed, y)

        y_pred = self.model.predict(X_transformed)

        self._y_true_all.extend(y.tolist())
        self._y_pred_all.extend(y_pred.tolist())

        chunk_acc = accuracy(y, y_pred)
        chunk_prec = precision(y, y_pred)
        chunk_rec = recall(y, y_pred)
        chunk_f1 = f1_score(y, y_pred)

        cumulative_acc = accuracy(
            np.asarray(self._y_true_all),
            np.asarray(self._y_pred_all),
        )

        log_entry = {
            "chunk": self._chunk_index,
            "n_samples": len(y),
            "accuracy": chunk_acc,
            "precision": chunk_prec,
            "recall": chunk_rec,
            "f1": chunk_f1,
            "cumulative_accuracy": cumulative_acc,
            "memory_bytes": self._memory_footprint(),
        }

        self._logs.append(log_entry)
        self._chunk_index += 1

        return self

    def score_chunk(self, X, y):
        """
        Score the model on a new chunk without updating the model.
        Parameters:
        - X: array-like of shape (n_samples, n_features).
        - y: array-like of shape (n_samples,).
        Returns:
        - dict with keys: 'accuracy', 'precision', 'recall', 'f1'.
        Raises:
        - ValueError: If X and y have different numbers of samples.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        X_transformed = self._transform(X)
        y_pred = self.model.predict(X_transformed)

        return {
            "accuracy": accuracy(y, y_pred),
            "precision": precision(y, y_pred),
            "recall": recall(y, y_pred),
            "f1": f1_score(y, y_pred),
        }

    def get_logs(self):
        """
        Return all per-chunk log entries collected so far.
        Returns:
        - list of dicts, one per chunk, with keys: 'chunk', 'n_samples',
          'accuracy', 'precision', 'recall', 'f1', 'cumulative_accuracy',
          'memory_bytes'.
        """
        return self._logs

    def reset_logs(self):
        """
        Clear all accumulated logs and reset chunk index.
        Returns:
        - self
        """
        self._logs = []
        self._chunk_index = 0
        self._y_true_all = []
        self._y_pred_all = []
        return self