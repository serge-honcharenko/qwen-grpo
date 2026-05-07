import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    i = np.arange(k)
    return 1.0 - np.exp(np.sum(np.log(n - c - i) - np.log(n - i)))

