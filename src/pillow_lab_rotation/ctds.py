import numpy as np
import cvxpy as cp
from numpy.linalg import inv, slogdet
from pillow_lab_rotation.tools import vec

class CTDS:
    def __init__(
            self,
            De: int,
            Di: int,
            Ne: int,
            Ni: int
    ):
        
        self.De = De
        self.Di = Di
        self.D = De + Di

        self.Ne = Ne
        self.Ni = Ni
        self.N = Ne + Ni

        self._init_params()
        self._init_constraints()
    
    def _init_params(self):

        # Initial state params
        self.mu0 = np.random.standard_normal((self.D, 1))
        self.V0 = np.eye(self.D)

        # Latent params
        self.A = np.zeros((self.D, self.D))
        # E columns: small positive off-diagonal
        self.A[:, :self.De] = np.random.uniform(0, 0.1, (self.D, self.De))
        # I columns: small negative off-diagonal
        self.A[:, self.De:] = -np.random.uniform(0, 0.1, (self.D, self.Di))
        # Diagonal: positive autocorrelation
        np.fill_diagonal(self.A, np.random.uniform(0.5, 0.9, self.D))
        # Ensure spectral radius < 1
        sr = np.max(np.abs(np.linalg.eigvals(self.A)))
        if sr >= 1.0:
            self.A *= 0.95 / sr
        self.Q = np.eye(self.D)

        # Observation params
        e2e_block = np.random.uniform(0, 1, (self.Ne, self.De))
        e2i_block = np.zeros((self.Ni, self.De))
        i2i_block = np.random.uniform(0, 1, (self.Ni, self.Di))
        i2e_block = np.zeros((self.Ne, self.Di))

        self.C = np.block(
            [[e2e_block, i2e_block],
             [e2i_block, i2i_block]]
        )
        self.R = np.eye(self.N)

    def _init_constraints(self):
        '''
        K is the sign constrainer on A
        '''
        A_mask = np.zeros((self.D, self.D))
        A_mask[:, :self.De] = 1
        A_mask[:, self.De:] = -1
        diag_idx = np.arange(self.D)
        A_mask[diag_idx, diag_idx] = 0

        A_flat = A_mask.flatten(order='F')
        self.A_pos_idx = A_flat > 0
        self.A_neg_idx = A_flat < 0

        C_nonneg_mask = np.zeros((self.N, self.D), dtype=bool)
        C_nonneg_mask[:self.Ne, :self.De] = True
        C_nonneg_mask[self.Ne:, self.De:] = True

        self.C_nonneg_idx = C_nonneg_mask.flatten(order='F')
        self.C_zero_idx = ~C_nonneg_mask.flatten(order='F')

    def fit(self, data: np.ndarray):
        '''
        Assume data is shape (n_trials, T, ydim, 1)
        '''
        self.data = data
        self.n_trials, self.T, _, _ = data.shape

        # EM
        LL_old = -np.inf
        while True:
            self.e_step()
            self.m_step()
            LL_new = self.log_likelihood()
            if LL_new < LL_old:
                raise ValueError('New LL less than old LL, implementation error')
            if LL_new - LL_old < 1e-7:
                break
            LL_old = LL_new
    
    def e_step(self):
        self.run_filter()
        self.run_smoother()
        self._get_sufficient_stats()

    def m_step(self):
        self.update_mu()
        self.update_V()
        self.update_A()
        self.update_C()
        self.update_Q()
        self.update_R()

    def run_filter(self):
        '''
        Vectorized Kalman filter. Modified from lds.py
        Also computes the LL in parallel
        '''

        # Covariance pass (same across trials)
        K_all = np.zeros((self.T, self.D, self.N))
        P_pred_all = np.zeros((self.T, self.D, self.D))
        P_filt_all = np.zeros((self.T, self.D, self.D))
        P_obs_inv_all = np.zeros((self.T, self.N, self.N))
        log_det_all = np.zeros(self.T)

        Pt = self.V0
        for t in range(self.T):

            # Evolve latents
            P_pred = Pt if t == 0 else self.A @ Pt @ self.A.T + self.Q
            P_pred_all[t] = P_pred

            # Predict observations
            P_obs = self.C @ P_pred @ self.C.T + self.R
            P_obs_inv = inv(P_obs)
            P_obs_inv_all[t] = P_obs_inv
            _, log_det_all[t] = slogdet(P_obs)
            
            # Update
            K = P_pred @ self.C.T @ P_obs_inv
            K_all[t] = K
            Pt = P_pred - K @ self.C @ P_pred
            P_filt_all[t] = Pt
        
        # Mean pass (vectorized across trials)
        x_filt = np.zeros((self.n_trials, self.T, self.D, 1))
        x_pred = np.zeros((self.n_trials, self.T, self.D, 1))
        LL = 0.0

        xt = np.broadcast_to(self.mu0, (self.n_trials, self.D, 1)).copy()
        for t in range(self.T):
            xp = xt if t == 0 else (self.A @ xt)
            x_pred[:, t] = xp
            innov = self.data[:, t] - self.C @ xp
            quad = (innov[:, :, 0] @ P_obs_inv_all[t] * innov[:, :, 0]).sum()
            LL += -0.5 * (self.n_trials * (self.N * np.log(2 * np.pi) + log_det_all[t]) + quad)
            xt = xp + K_all[t] @ innov
            x_filt[:, t] = xt

        self.P_predicted = np.broadcast_to(P_pred_all, (self.n_trials, self.T, self.D, self.D)).copy()
        self.P_filtered = np.broadcast_to(P_filt_all, (self.n_trials, self.T, self.D, self.D)).copy()
        self.x_filtered = x_filt
        self.x_predicted = x_pred
        self.LL = LL / (self.n_trials * self.T)
    
    def run_smoother(self):
        '''
        Vectorized smoother
        '''

        # Covariance pass (same for all trials)
        J_all = np.zeros((self.T - 1, self.D, self.D))
        P_smooth_all = np.zeros((self.T, self.D, self.D))
        sigma_x_all = np.zeros((self.T, self.D, self.D))

        P_smooth_all[-1] = self.P_filtered[0, -1]

        for t in range(self.T - 2, -1, -1):
            P_filt_t = self.P_filtered[0, t]
            P_pred_tp1 = self.P_predicted[0, t + 1]

            J = P_filt_t @ self.A.T @ inv(P_pred_tp1)
            J_all[t] = J
            P_smooth_all[t] = P_filt_t + J @ (P_smooth_all[t + 1] - P_pred_tp1) @ J.T
            sigma_x_all[t + 1] = J @ P_smooth_all[t + 1]

        # Mean pass (vectorized across trials)
        m = np.zeros((self.n_trials, self.T, self.D, 1))
        m[:, -1] = self.x_filtered[:, -1]

        for t in range(self.T - 2, -1, -1):
            m[:, t] = self.x_filtered[:, t] + J_all[t] @ (m[:, t + 1] - self.x_predicted[:, t + 1])

        self.m = m
        self.sigma = np.broadcast_to(P_smooth_all, (self.n_trials, self.T, self.D, self.D)).copy()
        self.sigma_x = np.broadcast_to(sigma_x_all, (self.n_trials, self.T, self.D, self.D)).copy()

    def _get_sufficient_stats(self):
        m = self.m[..., 0]
        y = self.data[..., 0]

        def _second_moment(m_slice, sigma_slice):
            flat = m_slice.reshape(-1, self.D)
            return flat.T @ flat + sigma_slice.reshape(-1, self.D, self.D).sum(0)
        
        def _cross_moment(a, b):
            return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])

        self.M11 = _second_moment(m[:, :1], self.sigma[:, :1])
        self.M2T = _second_moment(m[:, 1:], self.sigma[:, 1:])
        self.M1Tm1 = _second_moment(m[:, :-1], self.sigma[:, :-1])
        self.M1T = self.M11 + self.M2T

        self.M_delta = _cross_moment(m[:, :-1], m[:, 1:]) + self.sigma_x[:, 1:].reshape(-1, self.D, self.D).sum(0)

        self.Y = _cross_moment(y, y)
        self.Y_hat = _cross_moment(m, y)

    def update_mu(self):
        self.mu0 = self.m[:, 0, :, 0].mean(0, keepdims=True).T
    
    def update_V(self):
        self.V0 = self.M11 / self.n_trials - self.mu0 @ self.mu0.T
    
    def update_A(self):
        A = cp.Variable(self.D * self.D)
        Q_inv = inv(self.Q)
        q = np.kron(self.M_delta.T, np.eye(self.D)).T @ vec(Q_inv)
        objective = cp.Maximize(q @ A - 0.5 * cp.quad_form(A, np.kron(self.M1Tm1.T, Q_inv)))
        constraints = [
            A[self.A_pos_idx] >= 0,
            A[self.A_neg_idx] <= 0
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, verbose=False)
        self.A = A.value.reshape(self.A.shape, order='F')
    
    def update_Q(self):
        self.Q = (1 / (self.n_trials*(self.T - 1))) * (self.M2T - self.A @ self.M_delta - self.M_delta.T @ self.A.T + self.A @ self.M1Tm1 @ self.A.T)
    
    def update_C(self):
        
        C = cp.Variable(self.N * self.D)
        R_inv = inv(self.R)
        q = np.kron(self.Y_hat, np.eye(self.N)) @ vec(R_inv)
        objective = cp.Maximize(q @ C - 0.5 * cp.quad_form(C, np.kron(self.M1T.T, R_inv)))
        constraints = [
            C[self.C_nonneg_idx] >= 0,
            C[self.C_zero_idx] == 0
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, verbose=False)
        self.C = C.value.reshape(self.C.shape, order='F')
    
    def update_R(self):
        self.R = (1 / (self.T*self.n_trials)) * (self.Y - self.C@self.Y_hat - self.Y_hat.T @ self.C.T + self.C @ self.M1T @ self.C.T)
    
    def log_likelihood(self):
        return self.LL
    
    def predict(self, Y: np.ndarray):

        trials, timesteps, _, _ = Y.shape

        # Covariance pass
        K_all = np.zeros((timesteps, self.D, self.N))
        P_pred_all = np.zeros((timesteps, self.D, self.D))
        P_filt_all = np.zeros((timesteps, self.D, self.D))
        P_obs_all = np.zeros((timesteps, self.N, self.N))
        P_obs_inv_all = np.zeros((timesteps, self.N, self.N))
        log_det_all = np.zeros(timesteps)

        Pt = self.V0
        for t in range(timesteps):
            P_pred = Pt if t == 0 else self.A @ Pt @ self.A.T + self.Q
            P_pred_all[t] = P_pred
            P_obs = self.C @ P_pred @ self.C.T + self.R
            P_obs_all[t] = P_obs
            P_obs_inv = inv(P_obs)
            P_obs_inv_all[t] = P_obs_inv
            _, log_det_all[t] = slogdet(P_obs)
            K = P_pred @ self.C.T @ P_obs_inv
            K_all[t] = K
            Pt = P_pred - K @ self.C @ P_pred
            P_filt_all[t] = Pt

        # Mean pass (vectorized across trials, computes log-likelihood)
        x_pred = np.zeros((trials, timesteps, self.D, 1))
        x_filt = np.zeros((trials, timesteps, self.D, 1))
        LL = 0.0

        xt = np.broadcast_to(self.mu0, (trials, self.D, 1)).copy()
        for t in range(timesteps):
            xp = xt if t == 0 else (self.A @ xt)
            x_pred[:, t] = xp
            innov = Y[:, t] - self.C @ xp
            quad = (innov[:, :, 0] @ P_obs_inv_all[t] * innov[:, :, 0]).sum()
            LL += -0.5 * (trials * (self.N * np.log(2 * np.pi) + log_det_all[t]) + quad)
            xt = xp + K_all[t] @ innov
            x_filt[:, t] = xt

        obs_mean = self.C @ x_pred
        pred_covs = np.broadcast_to(P_pred_all, (trials, timesteps, self.D, self.D)).copy()
        obs_cov = np.broadcast_to(P_obs_all, (trials, timesteps, self.N, self.N)).copy()
        post_covs = np.broadcast_to(P_filt_all, (trials, timesteps, self.D, self.D)).copy()

        return x_pred, pred_covs, obs_mean, obs_cov, x_filt, post_covs, LL