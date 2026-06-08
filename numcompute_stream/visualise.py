import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", save_path=None):
    """
    Plot a metric (e.g. accuracy) across chunks over time.
    Parameters:
    - metric_values: list or array-like of metric values, one per chunk.
    - title: str, title of the plot.
    - ylabel: str, label for the y axis.
    - save_path: str or None. If provided, saves the plot to this path.
      If None, displays the plot inline.
    Raises:
    - ValueError: If metric_values is empty.
    """
    metric_values = np.asarray(metric_values)

    if metric_values.size == 0:
        raise ValueError("metric_values must not be empty.")

    chunks = np.arange(len(metric_values))

    fig, ax = plt.subplots()

    ax.plot(chunks, metric_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def compare_models(metric1, metric2, labels, title="Model comparison", ylabel="Metric", save_path=None):
    """
    Compare two models by plotting their streaming metrics side by side.
    Parameters:
    - metric1: list or array-like of metric values for model 1.
    - metric2: list or array-like of metric values for model 2.
    - labels: list of two strings, names for model 1 and model 2.
    - title: str, title of the plot.
    - ylabel: str, label for the y axis.
    - save_path: str or None. If provided, saves the plot to this path.
      If None, displays the plot inline.
    Raises:
    - ValueError: If metric1 or metric2 is empty.
    - ValueError: If labels does not contain exactly two entries.
    """
    metric1 = np.asarray(metric1)
    metric2 = np.asarray(metric2)

    if metric1.size == 0 or metric2.size == 0:
        raise ValueError("metric1 and metric2 must not be empty.")

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries.")

    chunks1 = np.arange(len(metric1))
    chunks2 = np.arange(len(metric2))

    fig, ax = plt.subplots()

    ax.plot(chunks1, metric1, marker="o", label=labels[0])
    ax.plot(chunks2, metric2, marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth", save_path=None):
    """
    Visualise predictions against true labels for the latest chunk.
    Parameters:
    - y_true: array-like of true class labels.
    - y_pred: array-like of predicted class labels.
    - title: str, title of the plot.
    - save_path: str or None. If provided, saves the plot to this path.
      If None, displays the plot inline.
    Raises:
    - ValueError: If y_true and y_pred have different lengths or are empty.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    indices = np.arange(len(y_true))

    fig, ax = plt.subplots()

    ax.scatter(indices, y_true, marker="o", label="ground truth")
    ax.scatter(indices, y_pred, marker="x", label="predictions")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class")
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()