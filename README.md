**NumCompute Stream** : A modular, streaming-compatible ensemble tree-based machine learning framework using plain Python + NumPy. 

This package extends the NumCompute package by building a decision tree-based ML framework supporting incremental learning, model ensembling, and real-time visualisation without external ML libraries.

**Features Includes :**

1. **Data I/O (io.py)**
CSV reader with missing value handling and flexible dtype support

2. **Preprocessing (preprocessing.py)**
Imputer, StandardScaler, OneHotEncoder — all with partial_fit() for streaming

3. **Statistics (stats.py)**
Mean, Std, Min, Max, Histogram, Quantiles (NaN-safe)
StreamStats class with update_stats() and sliding window support

4. **Metrics (metrics.py)**
Accuracy, Precision, Recall, F1, MSE, Confusion Matrix
StreamMetrics class with update(), reset(), result() and rolling window support

5. **Decision Tree (tree.py)**
DecisionTreeClassifier with Gini and entropy criteria
Supports max_depth, min_samples_split, max_features and partial_fit()

6. **Ensemble (ensemble.py)**
EnsembleClassifier with Bagging and Random Forest methods
partial_fit() and predict() fully streaming-compatible

7. **Streaming Trainer (stream.py)**
StreamTrainer with fit_chunk(), score_chunk()
Logs per-chunk metrics, cumulative accuracy and memory footprint

8. **Pipeline (pipeline.py)**
Modular transformation chaining with partial_fit() support
Consistent API: fit(), transform(), fit_transform(), predict()

9. **Visualisation (visualise.py)**
plot_metric_over_time(), compare_models(), plot_predictions_vs_ground_truth()
Supports saving to file or inline display

10. **Algorithms (sort_search.py)**
Top-K selection → O(n), Binary Search

11. **Ranking (rank.py)**
Tie-aware ranking: average, dense, ordinal
Percentile with interpolation: linear, lower, higher, midpoint

12. **Optimisation (optim.py)**
Finite difference gradients: central and forward difference
Jacobian computation for vector-valued functions

13. **Benchmarking (benchmarking.py)**
Compare vectorised vs loop implementations with timing utilities

**Installation**

    git clone https://github.com/ns-rg/NumCompute_v2.git
    cd NumCompute_v2
    pip install -e

**Quick Start**
Run the demo notebook:

    cd demo
    jupyter lab stream_demo.ipynb

**Example Usage**

1. Streaming Trainer Example

    ```py
    from numcompute_stream.tree import DecisionTreeClassifier
    from numcompute_stream.stream import StreamTrainer
    
    model = DecisionTreeClassifier(max_depth=5)
    trainer = StreamTrainer(model=model)
    trainer.fit_chunk(X_chunk, y_chunk)
    print(trainer.get_logs())
    ```

2. Ensemble Example

    ```py
    from numcompute_stream.ensemble import EnsembleClassifier

    clf = EnsembleClassifier(n_trees=10, method="random_forest")
    clf.partial_fit(X_chunk, y_chunk)
    predictions = clf.predict(X_new)
    ```
3. StreamMetrics Example
	```py
    from numcompute_stream.metrics import StreamMetrics

    sm = StreamMetrics()
    sm.update(y_true_chunk, y_pred_chunk)
    print(sm.result())
    ```
**Performance**

    Operation                       Time
    DecisionTreeClassifier fit()    ~0.08s avg
    EnsembleClassifier fit()        ~0.47s avg (10 trees)
    fit() on full data              ~0.08s avg
    partial_fit() over 5 chunks     ~0.23s avg

**Design Highlights**

1. Vectorisation-first approach for performance
2. Robust NaN handling, mixed data types and edge cases
3. Clean and consistent API across all modules
4. Streaming-compatible via partial_fit() and update() throughout
5. Numerical stability via Welford's algorithm and epsilon handling

**Testing**
Comprehensive unit tests (30+ cases) covering original and streaming functionality.

Run original tests:
		
    py -m tests.original.test_stats
    py -m tests.original.test_metrics
    py -m tests.original.test_preprocessing

Run streaming tests:

    py -m tests.streaming.test_stream_stats
    py -m tests.streaming.test_stream_metrics
    py -m tests.streaming.test_preprocessing_stream
    py -m tests.streaming.test_pipeline_stream
    py -m tests.streaming.test_tree
    py -m tests.streaming.test_ensemble
    py -m tests.streaming.test_stream_trainer

**Project Structure**

    NumCompute_v2/
    ├── numcompute_stream/
    │   ├── io.py
    │   ├── preprocessing.py
    │   ├── stats.py
    │   ├── metrics.py
    │   ├── pipeline.py
    │   ├── tree.py
    │   ├── ensemble.py
    │   ├── stream.py
    │   ├── visualise.py
    │   ├── sort_search.py
    │   ├── rank.py
    │   ├── optim.py
    │   ├── benchmarking.py
    │   └── utils.py
    ├── tests/
    │   ├── original/
    │   └── streaming/
    ├── demo/
    │   └── stream_demo.ipynb
    ├── benchmark/
    │   └── benchmark_stream.py
    ├── README.md
    └── setup.py
