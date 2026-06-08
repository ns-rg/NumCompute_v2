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
