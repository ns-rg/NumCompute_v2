import numpy as np
class Imputer:
    """
    Impute missing values in a dataset.
    For numeric columns, fills missing values with mean.
    For categorical columns, fills missing values with mode.
    Missing values are defined as empty strings, None, or np.nan.

    Example usage:
    imputer = Imputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    """

    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def _is_numeric_column(self, col):
        """
        Heuristic to determine if a column is numeric.
        Considers a column numeric if at least 50% of its valid (non-missing and non-empty) entries can be converted to float.

        Parameters:
        - col: array-like, the column to check
        Returns:
        - bool: True if column is considered numeric, False otherwise
        Raises:
        - ValueError: If the column is empty or contains only missing values.
        """
        valid_count = 0
        numeric_count = 0

        for val in col:
            if val in ("", b"", None):
                continue

            if isinstance(val, float) and np.isnan(val):
                continue

            valid_count += 1

            try:
                float(str(val))
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        return numeric_count >= valid_count / 2 if valid_count > 0 else False

    def fit(self, X):
        """
        Compute the statistics (mean for numeric, mode for categorical) for each column.
        Parameters:
        - X: 2D array-like, the input data to fit on
        Returns:
        - self: fitted Imputer instance
         Raises:
         - ValueError: If X is empty or not 2D.
        """
        X = np.array(X, dtype=object)
        n_features = X.shape[1]

        self.statistics_ = []

        for j in range(n_features):
            col = X[:, j]

            if self._is_numeric_column(col):
                col_float = []
                for val in col:
                    try:
                        col_float.append(float(val))
                    except NotImplementedError:
                        col_float.append(np.nan)

                col_float = np.array(col_float, dtype=float)

                if self.strategy == "mean":
                    stat = np.nanmean(col_float)
                else:
                    stat = np.nanmean(col_float)

            else:
                col_clean = [val for val in col if val not in (np.nan, "", b"", None)]

                if len(col_clean) == 0:
                    stat = None
                else:
                    values, counts = np.unique(col_clean, return_counts=True)
                    stat = values[np.argmax(counts)]

            self.statistics_.append(stat)

        return self

    def transform(self, X):
        """
        Impute missing values in X using the statistics computed in fit().
        Parameters:
        - X: 2D array-like, the input data to transform
        Returns:
        - X_out: 2D numpy array with imputed values
         Raises:
         - ValueError: If the imputer has not been fitted or if X is empty or not 2D.
        """
        if self.statistics_ is None:
            raise ValueError("Imputer must be fitted before calling transform().")

        X = np.array(X, dtype=object)
        X_out = X.copy()

        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                val = X_out[i, j]

                if val in ("", b"", None) or (isinstance(val, float) and np.isnan(val)):
                    X_out[i, j] = self.statistics_[j]

        return X_out

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Parameters:
        - X: 2D array-like, the input data to fit and transform
        Returns:
        - X_out: 2D numpy array with imputed values
         Raises:
         - ValueError: If X is empty or not 2D.
        """
        return self.fit(X).transform(X)


class StandardScaler:
    """
    Standardize numeric features by removing the mean and scaling to unit variance.
    Non-numeric columns are left unchanged.
    Missing values (empty strings, None, np.nan) are ignored when computing mean and std.
    Example usage:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.numeric_mask_ = None

    def _is_numeric(self, col):
        """
        Heuristic to determine if a column is numeric.
        Considers a column numeric if at least 50% of its valid (non-missing and non-empty) entries can be converted to float.
        Parameters:
        - col: array-like, the column to check
        Returns:
        - bool: True if column is considered numeric, False otherwise
        Raises:
        - ValueError: If the column is empty or contains only missing values.
        """
        valid_count = 0
        numeric_count = 0

        for val in col:
            if val in ("", b"", None):
                continue

            if isinstance(val, float) and np.isnan(val):
                continue

            valid_count += 1

            try:
                float(str(val))
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        return numeric_count >= valid_count / 2 if valid_count > 0 else False

    def fit(self, X, y=None):
        """
        Compute mean and std for numeric columns.
        Parameters:
        - X: 2D array-like, the input data to fit on
        - y: Ignored, present for API consistency by convention.
        Returns:
        - self: fitted StandardScaler instance
         Raises:
         - ValueError: If X is empty or not 2D.
        """
        X = np.array(X, dtype=object)
        n_features = X.shape[1]

        numeric_mask = []
        means = []
        stds = []

        for j in range(n_features):
            col = X[:, j]

            if self._is_numeric(col):
                col_float = col.astype(float)

                mean = np.nanmean(col_float)
                std = np.nanstd(col_float)

                if std == 0:
                    std = 1.0

                numeric_mask.append(True)
                means.append(mean)
                stds.append(std)
            else:
                numeric_mask.append(False)
                means.append(None)
                stds.append(None)

        self.numeric_mask_ = numeric_mask
        self.mean_ = means
        self.std_ = stds

        return self

    def transform(self, X):
        """
        Standardize numeric columns in X using the mean and std computed in fit().
        Non-numeric columns are left unchanged.
        Parameters:
        - X: 2D array-like, the input data to transform
        Returns:
        - X_out: 2D numpy array with standardized numeric columns
         Raises:
         - ValueError: If the scaler has not been fitted or if X is empty or not 2D.
        """
        if self.mean_ is None or self.std_ is None or self.numeric_mask_ is None:
            raise ValueError(
                "StandardScaler must be fitted before calling transform()."
            )

        X = np.array(X, dtype=object)
        X_out = X.copy()

        for j in range(X.shape[1]):
            if self.numeric_mask_[j]:
                col = np.array(
                    [float(x) if x not in ("", b"", None) else np.nan for x in X[:, j]],
                    dtype=float,
                )

                X_out[:, j] = (col - self.mean_[j]) / self.std_[j]

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    Non-categorical columns are left unchanged.
    Missing values (empty strings, None, np.nan) are treated as a separate category.
    Example usage:
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    """

    def __init__(self):
        self.categories_ = None
        self.categorical_mask_ = None

    def _is_numeric(self, col):
        """
        Heuristic to determine if a column is numeric.
        Considers a column numeric if at least 50% of its valid (non-missing and non-empty) entries can be converted to float.
        Parameters:
        - col: array-like, the column to check
        Returns:
        - bool: True if column is considered numeric, False otherwise
        Raises:
        - ValueError: If the column is empty or contains only missing values.
        """
        valid_count = 0
        numeric_count = 0

        for val in col:
            if val in ("", b"", None):
                continue

            if isinstance(val, float) and np.isnan(val):
                continue

            valid_count += 1

            try:
                _ = float(val)
                numeric_count += 1
            except (ValueError, TypeError):
                continue

        return numeric_count >= valid_count / 2 if valid_count > 0 else False

    def fit(self, X):
        """
        Compute categories for each categorical column.
        Parameters:
        - X: 2D array-like, the input data to fit on
        Returns:
        - self: fitted OneHotEncoder instance
        Raises:
        - ValueError: If X is empty or not 2D.
        """
        X = np.array(X, dtype=object)
        n_features = X.shape[1]

        self.categories_ = []
        self.categorical_mask_ = []

        for j in range(n_features):
            col = X[:, j]

            if not self._is_numeric(col):
                self.categorical_mask_.append(True)

                col_clean = [v for v in col if v not in ("", b"", None)]

                categories = np.unique(col_clean)
                self.categories_.append(categories)
            else:
                self.categorical_mask_.append(False)
                self.categories_.append(None)

        return self

    def transform(self, X):
        """
        Encode categorical features as a one-hot numeric array.
        Non-categorical columns are left unchanged.
        Parameters:
        - X: 2D array-like, the input data to transform
        Returns:
        - X_out: 2D numpy array with one-hot encoded categorical columns
         Raises:
         - ValueError: If the encoder has not been fitted or if X is empty or not 2D.
        """
        if self.categories_ is None or self.categorical_mask_ is None:
            raise ValueError("OneHotEncoder must be fitted before calling transform().")

        X = np.array(X, dtype=object)
        output_cols = []

        for j in range(X.shape[1]):
            col = X[:, j]

            if self.categorical_mask_[j]:
                categories = self.categories_[j]

                one_hot = np.zeros((X.shape[0], len(categories)))

                for i, val in enumerate(col):
                    if val in categories:
                        idx = np.where(categories == val)[0][0]
                        one_hot[i, idx] = 1

                output_cols.append(one_hot)

            else:
                col_numeric = np.array(
                    [float(x) if x not in ("", b"", None) else np.nan for x in col],
                    dtype=float,
                ).reshape(-1, 1)

                output_cols.append(col_numeric)

        return np.hstack(output_cols)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Parameters:
        - X: 2D array-like, the input data to fit and transform
        Returns:
        - X_out: 2D numpy array with one-hot encoded categorical columns
         Raises:
         - ValueError: If X is empty or not 2D.
        """
        return self.fit(X).transform(X)