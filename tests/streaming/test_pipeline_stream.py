import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, OneHotEncoder


class DummyClassifier:
    """
    A simple dummy classifier for testing pipeline partial_fit.
    Stores the last chunk it was fitted on.
    """

    def __init__(self):
        self.fitted = False
        self.classes_ = None

    def partial_fit(self, X, y=None):
        self.fitted = True
        if y is not None:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


def test_pipeline_partial_fit_calls_last_step():
    """Test that partial_fit calls partial_fit on the last step."""
    clf = DummyClassifier()
    pipe = Pipeline([("clf", clf)])
    X = np.array([["1.0", "2.0"], ["3.0", "4.0"]], dtype=object)
    y = np.array([0, 1])
    pipe.partial_fit(X, y)
    assert clf.fitted == True
    print("test_pipeline_partial_fit_calls_last_step passed!!")


def test_pipeline_partial_fit_transforms_before_last():
    """Test that partial_fit transforms data through intermediate steps."""
    scaler = StandardScaler()
    clf = DummyClassifier()
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    X = np.array([["1.0", "2.0"], ["3.0", "4.0"]], dtype=object)
    y = np.array([0, 1])
    pipe.partial_fit(X, y)
    assert clf.fitted == True
    print("test_pipeline_partial_fit_transforms_before_last passed!!")


def test_pipeline_partial_fit_multiple_chunks():
    """Test that pipeline partial_fit can be called on multiple chunks."""
    scaler = StandardScaler()
    clf = DummyClassifier()
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    X1 = np.array([["1.0"], ["2.0"]], dtype=object)
    X2 = np.array([["3.0"], ["4.0"]], dtype=object)
    pipe.partial_fit(X1, np.array([0, 1]))
    pipe.partial_fit(X2, np.array([0, 1]))
    assert clf.fitted == True
    print("test_pipeline_partial_fit_multiple_chunks passed!!")


def test_pipeline_predict_after_partial_fit():
    """Test that predict works correctly after partial_fit."""
    scaler = StandardScaler()
    clf = DummyClassifier()
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    X = np.array([["1.0"], ["2.0"], ["3.0"]], dtype=object)
    pipe.partial_fit(X, np.array([0, 1, 0]))
    preds = pipe.predict(X)
    assert len(preds) == 3
    print("test_pipeline_predict_after_partial_fit passed!!")


def test_pipeline_predict_no_predict_method():
    """Test that predict raises ValueError if last step has no predict method."""
    scaler = StandardScaler()
    pipe = Pipeline([("scaler", scaler)])
    X = np.array([["1.0"], ["2.0"]], dtype=object)
    pipe.fit(X)
    try:
        pipe.predict(X)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_pipeline_predict_no_predict_method passed!!")


def test_pipeline_partial_fit_scaler_updates_mean():
    """Test that scaler mean is updated after each partial_fit call."""
    scaler = StandardScaler()
    clf = DummyClassifier()
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    pipe.partial_fit(np.array([["1.0"], ["2.0"]], dtype=object), np.array([0, 1]))
    mean_after_first = scaler.mean_[0]
    pipe.partial_fit(np.array([["10.0"], ["20.0"]], dtype=object), np.array([0, 1]))
    mean_after_second = scaler.mean_[0]
    assert mean_after_second > mean_after_first
    print("test_pipeline_partial_fit_scaler_updates_mean passed!!")


def test_pipeline_fit_then_partial_fit():
    """Test that fit followed by partial_fit works without errors."""
    scaler = StandardScaler()
    clf = DummyClassifier()
    pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    X = np.array([["1.0"], ["2.0"]], dtype=object)
    pipe.fit(X, np.array([0, 1]))
    pipe.partial_fit(np.array([["3.0"], ["4.0"]], dtype=object), np.array([0, 1]))
    assert clf.fitted == True
    print("test_pipeline_fit_then_partial_fit passed!!")


if __name__ == "__main__":
    test_pipeline_partial_fit_calls_last_step()
    test_pipeline_partial_fit_transforms_before_last()
    test_pipeline_partial_fit_multiple_chunks()
    test_pipeline_predict_after_partial_fit()
    test_pipeline_predict_no_predict_method()
    test_pipeline_partial_fit_scaler_updates_mean()
    test_pipeline_fit_then_partial_fit()