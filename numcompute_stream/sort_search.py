import numpy as np

def topk(values, k, largest=True, return_indices=True):
    """
    This function returns top-k elements from array in values or (values, indices).

    Parameters -
    values: np.ndarray
    k: int
    largest: bool
        If True - largest k, else smallest k
    return_indices: bool
    """

    values = np.asarray(values)

    if k <= 0 or k > values.size:
        raise ValueError("k must be between 1 and len(values)")

    if largest:
        indices = np.argpartition(-values, k - 1)[:k]
    else:
        indices = np.argpartition(values, k - 1)[:k]

    top_values = values[indices]

    # sort result
    sorted_idx = np.argsort(-top_values) if largest else np.argsort(top_values)

    top_values = top_values[sorted_idx]
    indices = indices[sorted_idx]

    return (top_values, indices) if return_indices else top_values


def topk_loop(values, k, largest=True, return_indices=True):
    """
    This function returns top-k elements from array in values or (values, indices) using loop.
    Parameters -
    values: np.ndarray
    k: int
    largest: bool
        If True - largest k, else smallest k
    return_indices: bool
    """
    values = list(values)

    if k <= 0 or k > len(values):
        raise ValueError("k must be between 1 and len(values)")

    # Tracking original indices
    indexed_values = [(v, i) for i, v in enumerate(values)]

    result_values = []
    result_indices = []

    for _ in range(k):
        # Find large element each iteration
        best_idx = 0

        for i in range(1, len(indexed_values)):
            v_i = indexed_values[i][0]
            v_best = indexed_values[best_idx][0]

            if largest:
                if v_i > v_best:
                    best_idx = i
            else:
                if v_i < v_best:
                    best_idx = i

        best_value, best_original_idx = indexed_values.pop(best_idx)

        result_values.append(best_value)
        result_indices.append(best_original_idx)

    return (
        (np.array(result_values), np.array(result_indices))
        if return_indices
        else np.array(result_values)
    )


def binary_search(arr, x):
    """
    This function performs binary search on a sorted array to find the index of element x.
    Parameters -
    arr: np.ndarray
        A sorted array to search through.
    x: any
        The element to search for in the array.
    Returns -
    A tuple (index, found) where index is the position of x in arr if found or the position where x can be inserted to maintain sorted order if not found, and found is a boolean indicating whether x was found in arr.
    """

    arr = np.asarray(arr)

    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == x:
            return mid, True
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1

    return left, False