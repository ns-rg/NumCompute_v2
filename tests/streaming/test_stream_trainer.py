import numpy as np
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.tree import DecisionTreeClassifier


def test_stream_trainer_fit_chunk():
    """Test that fit_chunk runs without errors on simple data."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    assert len(trainer.get_logs()) == 1
    print("test_stream_trainer_fit_chunk passed!!")


def test_stream_trainer_logs_chunk_index():
    """Test that chunk index increments correctly across chunks."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    trainer.fit_chunk(X, y)
    assert trainer.get_logs()[0]["chunk"] == 0
    assert trainer.get_logs()[1]["chunk"] == 1
    print("test_stream_trainer_logs_chunk_index passed!!")


def test_stream_trainer_logs_accuracy():
    """Test that logs contain accuracy for each chunk."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    assert "accuracy" in trainer.get_logs()[0]
    print("test_stream_trainer_logs_accuracy passed!!")


def test_stream_trainer_logs_memory():
    """Test that logs contain memory footprint for each chunk."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    assert trainer.get_logs()[0]["memory_bytes"] > 0
    print("test_stream_trainer_logs_memory passed!!")


def test_stream_trainer_logs_cumulative_accuracy():
    """Test that cumulative accuracy is logged correctly."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    trainer.fit_chunk(X, y)
    assert "cumulative_accuracy" in trainer.get_logs()[1]
    print("test_stream_trainer_logs_cumulative_accuracy passed!!")


def test_stream_trainer_score_chunk():
    """Test that score_chunk returns correct metric keys."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    scores = trainer.score_chunk(X, y)
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in scores
    print("test_stream_trainer_score_chunk passed!!")


def test_stream_trainer_score_chunk_accuracy():
    """Test that score_chunk accuracy is between 0 and 1."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    scores = trainer.score_chunk(X, y)
    assert 0.0 <= scores["accuracy"] <= 1.0
    print("test_stream_trainer_score_chunk_accuracy passed!!")


def test_stream_trainer_reset_logs():
    """Test that reset_logs clears all logs and resets chunk index."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    trainer.reset_logs()
    assert len(trainer.get_logs()) == 0
    assert trainer._chunk_index == 0
    print("test_stream_trainer_reset_logs passed!!")


def test_stream_trainer_multiple_chunks():
    """Test that trainer accumulates logs over multiple chunks."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    for _ in range(5):
        trainer.fit_chunk(X, y)
    assert len(trainer.get_logs()) == 5
    print("test_stream_trainer_multiple_chunks passed!!")


def test_stream_trainer_invalid_model_no_partial_fit():
    """Test that model without partial_fit raises ValueError."""
    class BadModel:
        def predict(self, X):
            return X

    try:
        trainer = StreamTrainer(model=BadModel())
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_trainer_invalid_model_no_partial_fit passed!!")


def test_stream_trainer_invalid_model_no_predict():
    """Test that model without predict raises ValueError."""
    class BadModel:
        def partial_fit(self, X, y):
            return self

    try:
        trainer = StreamTrainer(model=BadModel())
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_trainer_invalid_model_no_predict passed!!")


def test_stream_trainer_mismatched_X_y():
    """Test that mismatched X and y raise ValueError."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    try:
        trainer.fit_chunk(np.array([[1.0], [2.0]]), np.array([0]))
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("test_stream_trainer_mismatched_X_y passed!!")


def test_stream_trainer_with_ensemble():
    """Test that StreamTrainer works correctly with EnsembleClassifier."""
    model = EnsembleClassifier(n_trees=5, method="random_forest")
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    assert len(trainer.get_logs()) == 1
    print("test_stream_trainer_with_ensemble passed!!")


def test_stream_trainer_logs_n_samples():
    """Test that logs record correct number of samples per chunk."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    assert trainer.get_logs()[0]["n_samples"] == 4
    print("test_stream_trainer_logs_n_samples passed!!")


def test_stream_trainer_logs_all_keys():
    """Test that each log entry contains all expected keys."""
    model = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model=model)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    trainer.fit_chunk(X, y)
    log = trainer.get_logs()[0]
    for key in ["chunk", "n_samples", "accuracy", "precision",
                "recall", "f1", "cumulative_accuracy", "memory_bytes"]:
        assert key in log
    print("test_stream_trainer_logs_all_keys passed!!")


if __name__ == "__main__":
    test_stream_trainer_fit_chunk()
    test_stream_trainer_logs_chunk_index()
    test_stream_trainer_logs_accuracy()
    test_stream_trainer_logs_memory()
    test_stream_trainer_logs_cumulative_accuracy()
    test_stream_trainer_score_chunk()
    test_stream_trainer_score_chunk_accuracy()
    test_stream_trainer_reset_logs()
    test_stream_trainer_multiple_chunks()
    test_stream_trainer_invalid_model_no_partial_fit()
    test_stream_trainer_invalid_model_no_predict()
    test_stream_trainer_mismatched_X_y()
    test_stream_trainer_with_ensemble()
    test_stream_trainer_logs_n_samples()
    test_stream_trainer_logs_all_keys()