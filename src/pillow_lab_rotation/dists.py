import numpy as np
from numpy.linalg import det, inv


class MultivariateNormal:
    def __init__(
            self,
            mean: np.ndarray,
            cov: np.ndarray
    ):
        
        self.mean = mean
        self.cov = cov
        self.k = mean.shape[0]
        
        self.cov_det = det(cov)
        self.precision = inv(cov)

    
    def pdf(
            self,
            x: np.ndarray
    ) -> float:
        
        centered = x - self.mean
        Z = (2 * np.pi) ** (self.k / 2) * np.sqrt(self.cov_det)

        if len(x.shape) == 1:
            exp_argument = -0.5 * centered.T @ self.precision @ centered
        else:
            exp_argument = -0.5 * np.sum(centered @ self.precision * centered, axis=1)
        
        return np.exp(exp_argument) / Z

    def log_pdf(
            self,
            x: np.ndarray
    ) -> float:
        return np.log(self.pdf(x))