import numpy as np


def load_csv(
    path,
    delimiter=",",
    skip_header=0,
    missing_values="",
    filling_values=np.nan,
    dtype=object,
):
    """
    Load CSV file into a 2D numpy array with robust handling of missing values and flexible data types.

    Parameters -
    -path : str (Path of CSV file)
    -delimiter : str, optional but default is ','
    -skip_header : int, optional (Number of rows to skip at beginning)
    -missing_values : str or sequence, optional (Representation of missing values default is empty string)
    -filling_values : scalar, optional (filling missing entries default is np.nan)
    -dtype : data-type, optional (To get desiered data type of result)

    Returns:
    - 2D numpy array containing data from CSV file

    Raises:
    - ValueError: If file cannot be loaded or data is empty
    """

    try:
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            skip_header=skip_header,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=filling_values,
            encoding="utf-8",
        )

        # To ensure at least get 2D output
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0:
            raise ValueError("Loaded data is empty.")

        # Normalizing missing values
        def _normalize_missing(x):
            # convert bytes → string
            if isinstance(x, bytes):
                x = x.decode("utf-8")

            # handle missing values
            if x in ("", None):
                return np.nan

            return x

        vectorized_fn = np.vectorize(_normalize_missing, otypes=[object])
        data = vectorized_fn(data)

        return data

    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")
