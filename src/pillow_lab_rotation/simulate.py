import numpy as np


def simulate_ctds_data(
    A, C, Q, R, mu0, V0,
    n_trials: int,
    T: int,
    seed: int = 0
):
    """
    Simulate data from a CTDS generative model.

    Returns:
        Y: (n_trials, T, N, 1) observed data
        X: (n_trials, T, D, 1) latent states
    """
    rng = np.random.default_rng(seed)
    D = A.shape[0]
    N = C.shape[0]

    X = np.zeros((n_trials, T, D, 1))
    Y = np.zeros((n_trials, T, N, 1))

    for n in range(n_trials):
        x = rng.multivariate_normal(mu0.ravel(), V0)[:, None]
        for t in range(T):
            if t > 0:
                x = A @ x + rng.multivariate_normal(np.zeros(D), Q)[:, None]
            X[n, t] = x
            Y[n, t] = C @ x + rng.multivariate_normal(np.zeros(N), R)[:, None]

    return Y, X


def make_ctds_params(De, Di, Ne, Ni, seed=42):
    """
    Generate ground-truth CTDS parameters obeying Dale's law.

    Returns dict with A, C, Q, R, mu0, V0.
    """
    rng = np.random.default_rng(seed)
    D = De + Di
    N = Ne + Ni

    # Dynamics matrix obeying Dale's law
    A = np.zeros((D, D))
    # E columns: non-negative off-diagonal
    A[:, :De] = rng.uniform(0, 0.3, (D, De))
    # I columns: non-positive off-diagonal
    A[:, De:] = -rng.uniform(0, 0.3, (D, Di))
    # Diagonal: positive for stability (auto-correlation)
    np.fill_diagonal(A, rng.uniform(0.5, 0.9, D))
    # Ensure spectral radius < 1 for stability
    eigvals = np.linalg.eigvals(A)
    sr = np.max(np.abs(eigvals))
    if sr >= 1.0:
        A = A * (0.95 / sr)

    # Emission matrix: block-diagonal, non-negative
    C = np.zeros((N, D))
    C[:Ne, :De] = rng.uniform(0.1, 1.0, (Ne, De))
    C[Ne:, De:] = rng.uniform(0.1, 1.0, (Ni, Di))

    # Noise covariances
    Q = 0.1 * np.eye(D)
    R = 0.5 * np.eye(N)

    # Initial state
    mu0 = np.zeros((D, 1))
    V0 = np.eye(D)

    return dict(A=A, C=C, Q=Q, R=R, mu0=mu0, V0=V0)
