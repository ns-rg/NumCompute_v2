from numcompute.pipeline import Pipeline
from numcompute.preprocessing import Imputer, StandardScaler
import numpy as np

def test_pipeline():
    """
    Test the Pipeline class with a simple imputation and scaling example.
    We will create a small dataset with missing values, apply the pipeline, and verify that the output is as expected.
    """
    X = np.array([[1, np.nan], [2, 3], [3, 4]], dtype=object)

    pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler())])

    X_out = pipe.fit_transform(X)

    assert X_out.shape == (3, 2)

    print("test_pipeline passed")


if __name__ == "__main__":
    test_pipeline()