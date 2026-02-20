import numpy as np
from numpy.linalg import inv, slogdet


class LinearDynamicalSystem:
    def __init__(
            self,
            zdim: int,
            xdim: int
    ):
        self.zdim = zdim
        self.xdim = xdim
        self.init_params()
    
    def init_params(self):

        # Initial state params
        self.μ0 = np.random.standard_normal((self.zdim, 1))
        self.V0 = np.eye(self.zdim)

        # Latent params
        self.Γ = np.eye(self.zdim)
        self.A = np.eye(self.zdim)

        # Observation params
        self.R = np.eye(self.xdim)
        self.C = np.random.randn(self.xdim, self.zdim)


    def fit(
            self,
            X: np.ndarray
    ):
        '''
        Assumes X is shape (n_trials, T, xdim, 1)
        '''
        self.X = X
        self.n_trials, self.T, _, _ = X.shape
        LL_old = -np.inf
        while True:
            self.e_step()
            LL_new = self.LL
            if LL_new < LL_old:
                raise ValueError('New LL less than old LL, implementation error')
            if LL_new - LL_old < 1e-5:
                break
            LL_old = LL_new
            self.m_step()

    def e_step(self):
        self.run_filter()
        self.run_smoother()
        self._get_sufficient_stats()
    
    def m_step(self):
        self.update_μ_and_V()
        self.update_A()
        self.update_Γ()
        self.update_C()
        self.update_R()
    
    def run_filter(self):

        # Covariance pass
        K_all = np.zeros((self.T, self.zdim, self.xdim))
        P_pred_all = np.zeros((self.T, self.zdim, self.zdim))
        P_filt_all = np.zeros((self.T, self.zdim, self.zdim))
        P_obs_inv_all = np.zeros((self.T, self.xdim, self.xdim))
        log_det_all = np.zeros(self.T)

        Pt = self.V0
        for t in range(self.T):
            P_pred = Pt if t == 0 else self.A @ Pt @ self.A.T + self.Γ
            P_pred_all[t] = P_pred
            P_obs = self.C @ P_pred @ self.C.T + self.R
            P_obs_inv = inv(P_obs)
            P_obs_inv_all[t] = P_obs_inv
            _, log_det_all[t] = slogdet(P_obs)
            K = P_pred @ self.C.T @ P_obs_inv
            K_all[t] = K
            Pt = P_pred - K @ self.C @ P_pred
            P_filt_all[t] = Pt

        # Mean pass (vectorized across trials, computes log-likelihood)
        z_filt = np.zeros((self.n_trials, self.T, self.zdim, 1))
        z_pred = np.zeros((self.n_trials, self.T, self.zdim, 1))
        LL = 0.0

        zt = np.broadcast_to(self.μ0, (self.n_trials, self.zdim, 1)).copy()
        for t in range(self.T):
            zp = zt if t == 0 else (self.A @ zt)
            z_pred[:, t] = zp
            innov = self.X[:, t] - self.C @ zp
            quad = (innov[:, :, 0] @ P_obs_inv_all[t] * innov[:, :, 0]).sum()
            LL += -0.5 * (self.n_trials * (self.xdim * np.log(2 * np.pi) + log_det_all[t]) + quad)
            zt = zp + K_all[t] @ innov
            z_filt[:, t] = zt

        self.P_predicted = np.broadcast_to(P_pred_all, (self.n_trials, self.T, self.zdim, self.zdim)).copy()
        self.P_filtered = np.broadcast_to(P_filt_all, (self.n_trials, self.T, self.zdim, self.zdim)).copy()
        self.z_filtered = z_filt
        self.z_predicted = z_pred
        self.LL = LL / (self.n_trials * self.T)


    

    def run_smoother(self):

        # Covariance pass (same for all trials)
        J_all = np.zeros((self.T - 1, self.zdim, self.zdim))
        P_smooth_all = np.zeros((self.T, self.zdim, self.zdim))
        sigma_x_all = np.zeros((self.T, self.zdim, self.zdim))

        P_smooth_all[-1] = self.P_filtered[0, -1]

        for t in range(self.T - 2, -1, -1):
            P_filt_t = self.P_filtered[0, t]
            P_pred_tp1 = self.P_predicted[0, t + 1]

            J = P_filt_t @ self.A.T @ inv(P_pred_tp1)
            J_all[t] = J
            P_smooth_all[t] = P_filt_t + J @ (P_smooth_all[t + 1] - P_pred_tp1) @ J.T
            sigma_x_all[t + 1] = J @ P_smooth_all[t + 1]

        # Mean pass (vectorized across trials)
        m = np.zeros((self.n_trials, self.T, self.zdim, 1))
        m[:, -1] = self.z_filtered[:, -1]

        for t in range(self.T - 2, -1, -1):
            m[:, t] = self.z_filtered[:, t] + J_all[t] @ (m[:, t + 1] - self.z_predicted[:, t + 1])

        self.m = m
        self.sigma = np.broadcast_to(P_smooth_all, (self.n_trials, self.T, self.zdim, self.zdim)).copy()
        self.sigma_x = np.broadcast_to(sigma_x_all, (self.n_trials, self.T, self.zdim, self.zdim)).copy()

    def _get_sufficient_stats(self):
        m = self.m[..., 0] # (n_trials, T, zdim)
        x = self.X[..., 0] # (n_trials, T, xdim)

        def _second_moment(m_slice, sigma_slice):
            flat = m_slice.reshape(-1, self.zdim)
            return flat.T @ flat + sigma_slice.reshape(-1, self.zdim, self.zdim).sum(0)
        
        def _cross_moment(a, b):
            return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])

        self.M11 = _second_moment(m[:, :1], self.sigma[:, :1])
        self.M2T = _second_moment(m[:, 1:], self.sigma[:, 1:])
        self.M1Tm1 = _second_moment(m[:, :-1], self.sigma[:, :-1])
        self.M1T = self.M11 + self.M2T

        self.M_delta = _cross_moment(m[:, :-1], m[:, 1:]) + self.sigma_x[:, 1:].reshape(-1, self.zdim, self.zdim).sum(0)

        self.XXT = _cross_moment(x, x)
        self.XXT_hat = _cross_moment(m, x)



    def update_μ_and_V(self):
        self.μ0 = self.m[:, 0, :, 0].mean(0, keepdims=True).T
        self.V0 = self.M11 / self.n_trials
        
    def update_Γ(self):
        self.Γ = (1 / (self.n_trials*(self.T - 1))) * (self.M2T - self.A @ self.M_delta - self.M_delta.T @ self.A.T + self.A @ self.M1Tm1 @ self.A.T)
        

    def update_A(self):
        self.A = self.M_delta.T @ inv(self.M1Tm1)

    def update_R(self):
        self.R = (1 / (self.T*self.n_trials)) * (self.XXT - self.C@self.XXT_hat - self.XXT_hat.T @ self.C.T + self.C @ self.M1T @ self.C.T)
    
    def update_C(self):
        self.C = self.XXT_hat.T @ inv(self.M1T)
       

    def predict(self, X: np.ndarray):

        trials, timesteps, _, _ = X.shape

        # Covariance pass
        K_all = np.zeros((timesteps, self.zdim, self.xdim))
        P_pred_all = np.zeros((timesteps, self.zdim, self.zdim))
        P_filt_all = np.zeros((timesteps, self.zdim, self.zdim))
        P_obs_all = np.zeros((timesteps, self.xdim, self.xdim))
        P_obs_inv_all = np.zeros((timesteps, self.xdim, self.xdim))
        log_det_all = np.zeros(timesteps)

        Pt = self.V0
        for t in range(timesteps):
            P_pred = Pt if t == 0 else self.A @ Pt @ self.A.T + self.Γ
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
        z_pred = np.zeros((trials, timesteps, self.zdim, 1))
        z_filt = np.zeros((trials, timesteps, self.zdim, 1))
        LL = 0.0

        zt = np.broadcast_to(self.μ0, (trials, self.zdim, 1)).copy()
        for t in range(timesteps):
            zp = zt if t == 0 else (self.A @ zt)
            z_pred[:, t] = zp
            innov = X[:, t] - self.C @ zp
            quad = (innov[:, :, 0] @ P_obs_inv_all[t] * innov[:, :, 0]).sum()
            LL += -0.5 * (trials * (self.xdim * np.log(2 * np.pi) + log_det_all[t]) + quad)
            zt = zp + K_all[t] @ innov
            z_filt[:, t] = zt

        obs_mean = self.C @ z_pred
        pred_covs = np.broadcast_to(P_pred_all, (trials, timesteps, self.zdim, self.zdim)).copy()
        obs_cov = np.broadcast_to(P_obs_all, (trials, timesteps, self.xdim, self.xdim)).copy()
        post_covs = np.broadcast_to(P_filt_all, (trials, timesteps, self.zdim, self.zdim)).copy()

        return z_pred, pred_covs, obs_mean, obs_cov, z_filt, post_covs, LL

