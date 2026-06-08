import numpy as np


def validate_inputs(y_true, y_pred):
    """
    Validates the input arrays for classification metrics.

    Parameters:
    -y_true (array-like): True class labels.
    -y_pred (array-like): Predicted class labels.

    Returns:
    -tuple: A tuple containing the validated y_true and y_pred arrays.

    Raises:
    -ValueError: If the input arrays have different lengths or are empty.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.shape[0] == 0:
        raise ValueError("Input arrays cannot be empty.")

    return y_true, y_pred


def accuracy(y_true, y_pred):
    """
    Computes the accuracy of predictions.

    Parameters:
    -y_true (array-like): True class labels.
    -y_pred (array-like): Predicted class labels.

    Returns:
    -float: The accuracy of the predictions.
    """
    y_true, y_pred = validate_inputs(y_true, y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix for classification predictions.

    Parameters:
    -y_true (array-like): True class labels.
    -y_pred (array-like): Predicted class labels.

    Returns:
    -numpy.ndarray: The confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return matrix


def precision(y_true, y_pred):
    """
    Computes the precision of predictions.

    Parameters:
    -y_true (array-like): True class labels.
    -y_pred (array-like): Predicted class labels.

    Returns:
    -float: The precision of the predictions.
    """
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp

    return np.mean(tp / (tp + fp + 1e-8))


def recall(y_true, y_pred):
    """
    Computes the recall of predictions.

    Parameters:
    -y_true (array-like): True class labels.
    -y_pred (array-like): Predicted class labels.

    Returns:
    -float: The recall of the predictions.
    """
    cm = confusion_matrix(y_true, y_pred)

    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp

    return np.mean(tp / (tp + fn + 1e-8))


def f1_score(y_true, y_pred):
    """
    Computes the F1 score of predictions.

    Parameters:
    -y_true (array-like): True class labels.
    -y_pred (array-like): Predicted class labels.

    Returns:
    -float: The F1 score of the predictions.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2 * (p * r) / (p + r + 1e-8)


def mse(y_true, y_pred):
    """
    Computes the mean squared error of predictions.

    Parameters:
    -y_true (array-like): True values.
    -y_pred (array-like): Predicted values.

    Returns:
    -float: The mean squared error of the predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean((y_true - y_pred) ** 2)


class StreamMetrics:
    """
    Compute classification metrics incrementally over chunks of data.

    Accumulates predictions and true labels chunk by chunk, and supports
    a rolling window mode that only scores the last window_size samples.

    Supported metrics via result(): accuracy, precision, recall, f1,
    confusion matrix, and AUC (via trapezoidal approximation).

    Example usage:
    sm = StreamMetrics()
    sm.update(y_true_chunk, y_pred_chunk)
    print(sm.result())
    """

    def __init__(self, window_size=None):
        """
        Initialise StreamMetrics.
        Parameters:
        - window_size: int or None. If set, only the last window_size
          samples are used when computing result().
        Raises:
        - ValueError: If window_size is set and less than 1.
        """
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.window_size = window_size
        self._y_true = []
        self._y_pred = []

    def update(self, y_true_chunk, y_pred_chunk):
        """
        Ingest a new chunk of predictions and true labels.
        Parameters:
        - y_true_chunk: array-like of true class labels.
        - y_pred_chunk: array-like of predicted class labels.
        Returns:
        - self
        Raises:
        - ValueError: If chunk arrays have different lengths or are empty.
        """
        y_true_chunk, y_pred_chunk = validate_inputs(y_true_chunk, y_pred_chunk)

        self._y_true.extend(y_true_chunk.tolist())
        self._y_pred.extend(y_pred_chunk.tolist())

        if self.window_size is not None:
            self._y_true = self._y_true[-self.window_size:]
            self._y_pred = self._y_pred[-self.window_size:]

        return self

    def reset(self):
        """
        Clear all accumulated predictions and labels.
        Returns:
        - self
        """
        self._y_true = []
        self._y_pred = []
        return self

    def result(self):
        """
        Compute and return all metrics over accumulated data.
        Returns:
        - dict with keys: 'accuracy', 'precision', 'recall', 'f1',
          'confusion_matrix', 'auc'.
        Raises:
        - ValueError: If no data has been ingested yet.
        """
        if len(self._y_true) == 0:
            raise ValueError("No data ingested yet. Call update() first.")

        y_true = np.asarray(self._y_true)
        y_pred = np.asarray(self._y_pred)

        acc = accuracy(y_true, y_pred)
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        classes = np.unique(y_true)
        auc = 0.0

        if len(classes) == 2:
            tp = np.sum((y_pred == classes[1]) & (y_true == classes[1]))
            fp = np.sum((y_pred == classes[1]) & (y_true == classes[0]))
            tn = np.sum((y_pred == classes[0]) & (y_true == classes[0]))
            fn = np.sum((y_pred == classes[0]) & (y_true == classes[1]))

            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)

            auc = np.trapezoid([0, tpr, 1], [0, fpr, 1])

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "auc": auc,
        }