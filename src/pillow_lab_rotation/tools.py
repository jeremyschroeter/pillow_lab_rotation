import numpy as np

def vec(X: np.ndarray) -> np.ndarray:
    return X.reshape(-1, order='F')