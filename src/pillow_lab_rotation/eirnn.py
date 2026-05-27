import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import nnls
from sklearn.decomposition import NMF


class EIRNNInit:
    def __init__(
            self,
            Ne: int,
            Ni: int,
            De: int,
            Di: int
    ):
        self.Ne = Ne
        self.Ni = Ni
        self.N = Ne + Ni
        self.De = De
        self.Di = Di
        self.D = De + Di

    def fit(
            self,
            observations: np.ndarray,
            nmf_runs: int = 10,
            start_seed: int = 0
    ) -> None:
        '''
        observations are expected to be shape (trials, time, N, 1)
        seed iterates +1 every run
        '''

        self.nmf_runs = nmf_runs
        self.start_seed = start_seed

        self.Yt = observations[:, :-1].squeeze(-1)
        self.Ytp1 = observations[:, 1:].squeeze(-1)
        
        self.Yt = self.Yt.reshape(-1, self.N)
        self.Ytp1 = self.Ytp1.reshape(-1, self.N)

        self._solve_constrained_regression()
        self._nmf()

        # Map to CTDS parameters
        self.C = self.U
        self.A = self.V @ self.U
        self._fit_initial_state(observations)
        self._fit_noise(observations)


    def _solve_constrained_regression(self):
        print('solving regression', flush=True)
        # Frobenius objective is row-separable and sign constraints are
        # column-wise, so each row j_n of J is an independent NNLS problem.
        # Reparameterize beta = [j_n[:Ne], -j_n[Ne:]] so all coefficients
        # are non-negative; the design matrix flips sign on the I block.
        X = np.hstack([self.Yt[:, :self.Ne], -self.Yt[:, self.Ne:]])

        J = np.zeros((self.N, self.N))
        for n in range(self.N):
            beta, _ = nnls(X, self.Ytp1[:, n])
            J[n, :self.Ne] = beta[:self.Ne]
            J[n, self.Ne:] = -beta[self.Ne:]
        self.J_fit = J
    
    def _nmf(self):
        Je = np.abs(self.J_fit[:self.Ne])
        Ji = np.abs(self.J_fit[self.Ne:])

        best_error = np.inf
        for i in range(self.nmf_runs):
            print(f'NMF run {i + 1} / {self.nmf_runs}', flush=True)
            e_model = NMF(n_components=self.De, init='random', random_state=self.start_seed + i)
            We = e_model.fit_transform(Je)
            He = e_model.components_

            i_model = NMF(n_components=self.Di, init='random', random_state=self.start_seed + i)
            Wi = i_model.fit_transform(Ji)
            Hi = i_model.components_

            U = block_diag(We, Wi)
            V = np.vstack([He, Hi])
            V[:, self.Ne:] *= -1

            error = np.linalg.norm(U @ V - self.J_fit, 'fro') / np.linalg.norm(self.J_fit, 'fro')
            if error < best_error:
                best_error = error
                self.U, self.V = U, V
            
    def _fit_initial_state(self, observations):
        C_pinv = np.linalg.pinv(self.C)
        y0 = observations[:, 0, :, 0]  # (n_trials, N)
        x0 = (y0 @ C_pinv.T)           # (n_trials, D)
        self.mu0 = x0.mean(axis=0, keepdims=True).T  # (D, 1)
        self.Q0 = np.cov(x0, rowvar=False)

    def _fit_noise(self, observations):
        C_pinv = np.linalg.pinv(self.C)
        y_all = observations.squeeze(-1).reshape(-1, self.N)
        x_all = (y_all @ C_pinv.T)

        # Emission noise: y = Cx + noise
        resid_y = y_all - x_all @ self.C.T
        self.R = np.diag(resid_y.var(axis=0))

        # Dynamics noise: x_{t+1} = A x_t + noise
        Yt = observations[:, :-1].squeeze(-1).reshape(-1, self.N)
        Ytp1 = observations[:, 1:].squeeze(-1).reshape(-1, self.N)
        Xt = Yt @ C_pinv.T
        Xtp1 = Ytp1 @ C_pinv.T
        resid_x = Xtp1 - Xt @ self.A.T
        self.Q = np.diag(resid_x.var(axis=0))