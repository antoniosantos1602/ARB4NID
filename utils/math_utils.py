import numpy as np


def safe_mahalanobis(x, mean, cov_inv):
    m = (x - mean) @ cov_inv @ (x - mean).T
    return np.sqrt(max(m, 0))
