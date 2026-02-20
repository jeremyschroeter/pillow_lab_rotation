import numpy as np
from numpy.linalg import inv, slogdet
import cvxpy as cp

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
    
    def _init_params(self):

        # Initial state params
        self.mu0 = np.random.standard_normal((self.D, 1))
        self.V0 = np.eye(self.D)

        # Latent params
        self.A = np.eye(self.D)
        self.A[:, self.De:] = -1
        self.Q = np.eye(self.D)

        # Observation params
        e2e_block = np.random.uniform(0, 1, (self.Ne, self.Ne))
        e2i_block = np.zeros((self.Ni, self.Ne))
        i2i_block = np.random.uniform(0, 1, (self.Ni, self.Ni))
        i2e_block = np.zeros((self.Ne, self.Ni))

        self.C = np.block(
            [[e2e_block, i2e_block],
             [e2i_block, i2i_block]]
        )
        self.R = np.eye(self.N)
    

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
                raise ValueError('New LL less than old LL, implementatino erro')
            if LL_new - LL_old < 1e-5:
                break
            LL_old = LL_new
    
    def e_step(self):
        self.run_filter()
        self.run_smoother()

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
        z_filt = np.zeros((self.n_trials, self.T, self.zdim, 1))
        z_pred = np.zeros((self.n_trials, self.T, self.zdim, 1))
        LL = 0.0

        zt = np.broadcast_to(self.mu0, self.n_trials, self.zdim, 1).copy()
        for t in range(self.T):
            zp = zt if t == 0 else (self.A @ zt)
            z_pred[:, t] = zp
            innov = self.data[:, t] - self.C @ zp
            quad = (innov[:, :, 0] @ P_obs_inv_all[t] * innov[:, :, 0].sum())
            LL += -0.5 * (self.n_trials * (self.xdim * np.log(2 * np.pi) + log_det_all[t]) + quad)
            zt = zp + K_all[t] @ innov
            z_filt[:, t] = zt
        
        self.P_predicted = np.broadcast_to(P_pred_all, (self.n_trials, self.T, self.D, self.D)).copy()
        self.P_filtered = np.broadcast_to(P_filt_all, (self.n_trials, self.T, self.D, self.D)).copy()
        self.z_filtered = z_filt
        self.z_predicted = z_pred
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
        m[:, -1] = self.z_filtered[:, -1]

        for t in range(self.T - 2, -1, -1):
            m[:, t] = self.z_filtered[:, t] + J_all[t] @ (m[:, t + 1] - self.z_predicted[:, t + 1])

        self.m = m
        self.sigma = np.broadcast_to(P_smooth_all, (self.n_trials, self.T, self.D, self.D)).copy()
        self.sigma_x = np.broadcast_to(sigma_x_all, (self.n_trials, self.T, self.D, self.D)).copy()

    def _get_sufficient_stats(self):
        m = self.m[..., 0] # (n_trials, T, zdim)
        y = self.data[..., 0] # (n_trials, T, xdim)

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
        NotImplemented
    
    def update_V(self):
        NotImplemented
    
    def update_A(self):
        NotImplemented
    
    def update_Q(self):
        NotImplemented
    
    def update_C(self):
        NotImplemented
    
    def update_R(self):
        NotImplemented

    def log_likelihood(self):
        NotImplemented