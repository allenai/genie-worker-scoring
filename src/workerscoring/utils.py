"""Utilities"""

import numpy as np


def check_batchable_float(xs):
    if not isinstance(xs, (int, float, list, np.ndarray)):
        raise TypeError(
            f'The value must either be a float or a 1D array.'
        )

    if isinstance(xs, np.ndarray) and len(xs.shape) > 1:
        raise ValueError(
            f'The value must be a 1D array, not a {len(xs.shape)}D array.'
        )

    return np.array(xs, dtype=np.float64)
