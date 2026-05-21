import numpy as np
from numpy.linalg import inv, slogdet


class LinearDynamicalSystem:
    def __init__(
            self,
            xdim: int,
            ydim: int,
            udim: int | None = None,
            feedthrough: bool = True,
            fit_mu0: bool = True,
            fit_b: bool = False,
            fit_d_bias: bool = False
    ):
        self.xdim = xdim
        self.ydim = ydim
        self.udim = udim if udim is not None else 0
        self.feedthrough = feedthrough
        self.fit_mu0 = fit_mu0
        self.fit_b = fit_b
        self.fit_d_bias = fit_d_bias
        self.init_params()

    def init_params(self):

        # Initial state params
        self.mu0 = np.random.standard_normal((self.xdim, 1))
        self.Q0 = np.eye(self.xdim)

        # Latent params
        self.Q = np.eye(self.xdim)
        self.A = np.eye(self.xdim)

        # Observation params
        self.R = np.eye(self.ydim)
        self.C = np.random.randn(self.ydim, self.xdim)

        # Input params
        self.B = np.random.randn(self.xdim, self.udim) if self.udim > 0 else np.zeros((self.xdim, self.udim))
        if self.udim > 0 and self.feedthrough:
            self.D = np.random.randn(self.ydim, self.udim)
        else:
            self.D = np.zeros((self.ydim, self.udim))

        # Affine bias params; zero when not fit so all formulas are no-ops in that case
        self.b = np.zeros((self.xdim, 1))
        self.d_bias = np.zeros((self.ydim, 1))


    def fit(
            self,
            observations: np.ndarray,
            inputs: np.ndarray | None = None,
            verbose: bool = False,
            max_iter: int | None = None,
            criterion: float = 1e-8
    ):
        '''
        Assumes observations is shape (n_trials, T, ydim, 1)
        '''
        self.observations = observations
        self.n_trials, self.T, _, _ = observations.shape
        self.inputs = inputs if inputs is not None else np.zeros((self.n_trials, self.T, self.udim, 1))
        self.ll_history = []
        LL_old = -np.inf
        iteration = 0
        while True:
            self.e_step()
            LL_new = self.LL
            self.ll_history.append(LL_new)
            if verbose:
                print(f"Iteration {iteration}: LL = {LL_new:.6f}")
            if LL_new < LL_old:
                raise ValueError('New LL less than old LL, implementation error')
            if max_iter is None and LL_new - LL_old < criterion:
                break
            if max_iter is not None and iteration >= max_iter:
                break
            LL_old = LL_new
            self.m_step()
            iteration += 1

    def e_step(self):
        self.run_filter()
        self.run_smoother()
        self._get_sufficient_stats()

    def m_step(self):
        if self.udim > 0:
            self.update_A_B()
            if self.feedthrough:
                self.update_C_D()
            else:
                self.update_C()
        else:
            self.update_A()
            self.update_C()
        self.update_b()
        self.update_d_bias()
        self.update_mu_and_Q0()
        self.update_Q()
        self.update_R()

    def run_filter(self):

        # Covariance pass
        K_all = np.zeros((self.T, self.xdim, self.ydim))
        P_pred_all = np.zeros((self.T, self.xdim, self.xdim))
        P_filt_all = np.zeros((self.T, self.xdim, self.xdim))
        P_obs_inv_all = np.zeros((self.T, self.ydim, self.ydim))
        log_det_all = np.zeros(self.T)

        Pt = self.Q0
        for t in range(self.T):
            P_pred = Pt if t == 0 else self.A @ Pt @ self.A.T + self.Q
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
        x_filt = np.zeros((self.n_trials, self.T, self.xdim, 1))
        x_pred = np.zeros((self.n_trials, self.T, self.xdim, 1))
        LL = 0.0

        xt = np.broadcast_to(self.mu0, (self.n_trials, self.xdim, 1)).copy()
        for t in range(self.T):
            yt = self.observations[:, t]
            ut = self.inputs[:, t]

            # Evolve latents (dynamics bias b enters at transitions, not at t=0)
            xp = (xt + self.B @ ut) if t == 0 else (self.A @ xt + self.B @ ut + self.b)
            x_pred[:, t] = xp

            # Update
            if self.feedthrough:
                innov = yt - self.C @ xp - self.D @ ut - self.d_bias
            else:
                innov = yt - self.C @ xp - self.d_bias
            xt = xp + K_all[t] @ innov
            x_filt[:, t] = xt

            # Accumulate LL
            quad = (innov[:, :, 0] @ P_obs_inv_all[t] * innov[:, :, 0]).sum()
            LL += -0.5 * (self.n_trials * (self.ydim * np.log(2 * np.pi) + log_det_all[t]) + quad)


        self.P_predicted = np.broadcast_to(P_pred_all, (self.n_trials, self.T, self.xdim, self.xdim)).copy()
        self.P_filtered = np.broadcast_to(P_filt_all, (self.n_trials, self.T, self.xdim, self.xdim)).copy()
        self.x_filtered = x_filt
        self.x_predicted = x_pred
        self.LL = LL / (self.n_trials * self.T)




    def run_smoother(self):

        # Covariance pass (same for all trials)
        J_all = np.zeros((self.T - 1, self.xdim, self.xdim))
        P_smooth_all = np.zeros((self.T, self.xdim, self.xdim))
        sigma_x_all = np.zeros((self.T, self.xdim, self.xdim))

        P_smooth_all[-1] = self.P_filtered[0, -1]

        for t in range(self.T - 2, -1, -1):
            P_filt_t = self.P_filtered[0, t]
            P_pred_tp1 = self.P_predicted[0, t + 1]

            J = P_filt_t @ self.A.T @ inv(P_pred_tp1)
            J_all[t] = J
            P_smooth_all[t] = P_filt_t + J @ (P_smooth_all[t + 1] - P_pred_tp1) @ J.T
            sigma_x_all[t + 1] = J @ P_smooth_all[t + 1]

        # Mean pass (vectorized across trials)
        m = np.zeros((self.n_trials, self.T, self.xdim, 1))
        m[:, -1] = self.x_filtered[:, -1]

        for t in range(self.T - 2, -1, -1):
            m[:, t] = self.x_filtered[:, t] + J_all[t] @ (m[:, t + 1] - self.x_predicted[:, t + 1])

        self.m = m
        self.sigma = np.broadcast_to(P_smooth_all, (self.n_trials, self.T, self.xdim, self.xdim)).copy()
        self.sigma_x = np.broadcast_to(sigma_x_all, (self.n_trials, self.T, self.xdim, self.xdim)).copy()

    def _get_sufficient_stats(self):
        m = self.m[..., 0] # (n_trials, T, xdim)
        y = self.observations[..., 0] # (n_trials, T, ydim)
        u = self.inputs[..., 0]

        def _second_moment(m_slice, sigma_slice):
            flat = m_slice.reshape(-1, self.xdim)
            return flat.T @ flat + sigma_slice.reshape(-1, self.xdim, self.xdim).sum(0)

        def _cross_moment(a, b):
            return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])

        self.M11 = _second_moment(m[:, :1], self.sigma[:, :1])
        self.M2T = _second_moment(m[:, 1:], self.sigma[:, 1:])
        self.M1Tm1 = _second_moment(m[:, :-1], self.sigma[:, :-1])
        self.M1T = self.M11 + self.M2T

        self.M_delta = _cross_moment(m[:, :-1], m[:, 1:]) + self.sigma_x[:, 1:].reshape(-1, self.xdim, self.xdim).sum(0)

        self.Y = _cross_moment(y, y)
        self.Y_hat = _cross_moment(m, y)

        # First-order sums for affine-bias updates (b and d_bias) and the general
        # Q0/Q/R formulas that account for nonzero biases.
        self.m_sum = m.reshape(-1, self.xdim).sum(0).reshape(self.xdim, 1)
        self.m_sum_1 = m[:, 0].sum(0).reshape(self.xdim, 1)
        self.m_sum_T = m[:, -1].sum(0).reshape(self.xdim, 1)
        self.y_sum = y.reshape(-1, self.ydim).sum(0).reshape(self.ydim, 1)

        if self.udim > 0:
            self.U1T = _cross_moment(u, u)
            self.U2T = _cross_moment(u[:, 1:], u[:, 1:])
            self.U_hat_2T = _cross_moment(u[:, 1:], m[:, 1:])
            self.Uy = _cross_moment(u, y)
            self.U_delta = _cross_moment(u[:, 1:], m[:, :-1])
            self.U_hat_1T = _cross_moment(u, m)
            self.U11 = _cross_moment(u[:, :1], u[:, :1])
            self.U_hat_11 = _cross_moment(u[:, :1], m[:, :1])
            self.u_sum = u.reshape(-1, self.udim).sum(0).reshape(self.udim, 1)
            self.u_sum_1 = u[:, 0].sum(0).reshape(self.udim, 1)
        else:
            self.u_sum = np.zeros((0, 1))
            self.u_sum_1 = np.zeros((0, 1))




    def update_mu_and_Q0(self):
        # Optimal mu0 (assuming x_1 ~ N(mu0 + B u_1, Q0)); skipped when fit_mu0=False
        # so the user can hold mu0 at a fixed value (e.g. zero, matching ssm).
        if self.fit_mu0:
            self.mu0 = self.m[:, 0, :, 0].mean(0, keepdims=True).T
            if self.udim > 0:
                u1_mean = self.inputs[:, 0, :, 0].mean(0, keepdims=True).T
                self.mu0 = self.mu0 - self.B @ u1_mean

        # General Q0 formula: residual covariance under x_1 ~ N(mu0 + B u_1, Q0).
        # Reduces to the original simplified form when mu0 is the optimal value above.
        mbar_1 = self.m[:, 0, :, 0].mean(0, keepdims=True).T
        Q0 = self.M11 / self.n_trials - mbar_1 @ self.mu0.T - self.mu0 @ mbar_1.T + self.mu0 @ self.mu0.T
        if self.udim > 0:
            ubar_1 = self.inputs[:, 0, :, 0].mean(0, keepdims=True).T
            Q0 = Q0 - self.B @ self.U_hat_11 / self.n_trials - self.U_hat_11.T @ self.B.T / self.n_trials
            Q0 = Q0 + self.mu0 @ ubar_1.T @ self.B.T + self.B @ ubar_1 @ self.mu0.T
            Q0 = Q0 + self.B @ self.U11 @ self.B.T / self.n_trials
        self.Q0 = Q0

    def update_Q(self):
        Q = self.M2T + self.A @ self.M1Tm1 @ self.A.T - self.A @ self.M_delta - self.M_delta.T @ self.A.T
        if self.udim > 0:
            Q += (self.B @ self.U2T @ self.B.T
                  - self.B @ self.U_hat_2T - self.U_hat_2T.T @ self.B.T
                  + self.B @ self.U_delta @ self.A.T + self.A @ self.U_delta.T @ self.B.T)
        # Dynamics-bias contributions (all zero when b=0)
        m_sum_2T = self.m_sum - self.m_sum_1
        m_sum_1Tm1 = self.m_sum - self.m_sum_T
        Q += ((self.T - 1) * self.n_trials * self.b @ self.b.T
              - self.b @ m_sum_2T.T - m_sum_2T @ self.b.T
              + self.A @ m_sum_1Tm1 @ self.b.T + self.b @ m_sum_1Tm1.T @ self.A.T)
        if self.udim > 0:
            u_sum_2T = self.u_sum - self.u_sum_1
            Q += self.B @ u_sum_2T @ self.b.T + self.b @ u_sum_2T.T @ self.B.T
        self.Q = Q / (self.n_trials * (self.T - 1))

    def update_A(self):
        # Effective target is x_t - b, so the cross-moment shifts by m_sum_1Tm1 @ b^T.
        m_sum_1Tm1 = self.m_sum - self.m_sum_T
        M_delta_eff = self.M_delta - m_sum_1Tm1 @ self.b.T
        self.A = M_delta_eff.T @ inv(self.M1Tm1)

    def update_A_B(self):
        m_sum_1Tm1 = self.m_sum - self.m_sum_T
        M_delta_eff = self.M_delta - m_sum_1Tm1 @ self.b.T
        u_sum_2T = self.u_sum - self.u_sum_1
        U_hat_2T_eff = self.U_hat_2T - u_sum_2T @ self.b.T
        first_matrix = np.block([M_delta_eff.T, U_hat_2T_eff.T])
        second_matrix = inv(np.block(
            [[self.M1Tm1, self.U_delta.T],
             [self.U_delta, self.U2T]]
        ))
        AB = first_matrix @ second_matrix
        self.A = AB[:, :self.xdim]
        self.B = AB[:, self.xdim:]

    def update_b(self):
        if not self.fit_b:
            return
        m_sum_2T = self.m_sum - self.m_sum_1
        m_sum_1Tm1 = self.m_sum - self.m_sum_T
        denom = (self.T - 1) * self.n_trials
        b = (m_sum_2T - self.A @ m_sum_1Tm1) / denom
        if self.udim > 0:
            u_sum_2T = self.u_sum - self.u_sum_1
            b = b - self.B @ u_sum_2T / denom
        self.b = b

    def update_R(self):
        R = self.Y + self.C @ self.M1T @ self.C.T - self.C @ self.Y_hat - self.Y_hat.T @ self.C.T
        if self.udim > 0 and self.feedthrough:
            R += (self.D @ self.U1T @ self.D.T
                  - self.D @ self.Uy - self.Uy.T @ self.D.T
                  + self.D @ self.U_hat_1T @ self.C.T + self.C @ self.U_hat_1T.T @ self.D.T)
        # Observation-bias contributions (all zero when d_bias=0)
        R += (self.T * self.n_trials * self.d_bias @ self.d_bias.T
              - self.d_bias @ self.y_sum.T - self.y_sum @ self.d_bias.T
              + self.C @ self.m_sum @ self.d_bias.T + self.d_bias @ self.m_sum.T @ self.C.T)
        if self.udim > 0 and self.feedthrough:
            R += self.D @ self.u_sum @ self.d_bias.T + self.d_bias @ self.u_sum.T @ self.D.T
        self.R = R / (self.T * self.n_trials)

    def update_C(self):
        Y_hat_eff = self.Y_hat - self.m_sum @ self.d_bias.T
        self.C = Y_hat_eff.T @ inv(self.M1T)

    def update_C_D(self):
        Y_hat_eff = self.Y_hat - self.m_sum @ self.d_bias.T
        Uy_eff = self.Uy - self.u_sum @ self.d_bias.T
        first_matrix = np.block([Y_hat_eff.T, Uy_eff.T])
        second_matrix = inv(np.block(
            [[self.M1T, self.U_hat_1T.T],
            [self.U_hat_1T, self.U1T]]
        ))

        CD = first_matrix @ second_matrix
        self.C = CD[:, :self.xdim]
        self.D = CD[:, self.xdim:]

    def update_d_bias(self):
        if not self.fit_d_bias:
            return
        total = self.T * self.n_trials
        d = (self.y_sum - self.C @ self.m_sum) / total
        if self.udim > 0 and self.feedthrough:
            d = d - self.D @ self.u_sum / total
        self.d_bias = d


    def predict(self, Y: np.ndarray, inputs: np.ndarray | None = None):

        trials, timesteps, _, _ = Y.shape
        U = inputs if inputs is not None else np.zeros((trials, timesteps, self.udim, 1))

        # Covariance pass
        K_all = np.zeros((timesteps, self.xdim, self.ydim))
        P_pred_all = np.zeros((timesteps, self.xdim, self.xdim))
        P_filt_all = np.zeros((timesteps, self.xdim, self.xdim))
        P_obs_all = np.zeros((timesteps, self.ydim, self.ydim))
        P_obs_inv_all = np.zeros((timesteps, self.ydim, self.ydim))
        log_det_all = np.zeros(timesteps)

        Pt = self.Q0
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
        x_pred = np.zeros((trials, timesteps, self.xdim, 1))
        x_filt = np.zeros((trials, timesteps, self.xdim, 1))
        LL = 0.0

        xt = np.broadcast_to(self.mu0, (trials, self.xdim, 1)).copy()
        for t in range(timesteps):
            ut = U[:, t]
            xp = (xt + self.B @ ut) if t == 0 else (self.A @ xt + self.B @ ut + self.b)
            x_pred[:, t] = xp
            innov = Y[:, t] - self.C @ xp - self.D @ ut - self.d_bias
            quad = (innov[:, :, 0] @ P_obs_inv_all[t] * innov[:, :, 0]).sum()
            LL += -0.5 * (trials * (self.ydim * np.log(2 * np.pi) + log_det_all[t]) + quad)
            xt = xp + K_all[t] @ innov
            x_filt[:, t] = xt

        obs_mean = self.C @ x_pred + self.D @ U + self.d_bias
        pred_covs = np.broadcast_to(P_pred_all, (trials, timesteps, self.xdim, self.xdim)).copy()
        obs_cov = np.broadcast_to(P_obs_all, (trials, timesteps, self.ydim, self.ydim)).copy()
        post_covs = np.broadcast_to(P_filt_all, (trials, timesteps, self.xdim, self.xdim)).copy()

        return x_pred, pred_covs, obs_mean, obs_cov, x_filt, post_covs, LL


    def sample(self, T: int, n_trials: int = 1, inputs: np.ndarray | None = None):
        U = inputs if inputs is not None else np.zeros((n_trials, T, self.udim, 1))

        x = np.zeros((n_trials, T, self.xdim, 1))
        y = np.zeros((n_trials, T, self.ydim, 1))

        chol_Q0 = np.linalg.cholesky(self.Q0)
        chol_Q = np.linalg.cholesky(self.Q)
        chol_R = np.linalg.cholesky(self.R)

        x[:, 0] = self.mu0 + self.B @ U[:, 0] + chol_Q0 @ np.random.standard_normal((n_trials, self.xdim, 1))
        y[:, 0] = self.C @ x[:, 0] + self.D @ U[:, 0] + self.d_bias + chol_R @ np.random.standard_normal((n_trials, self.ydim, 1))

        for t in range(1, T):
            x[:, t] = self.A @ x[:, t - 1] + self.B @ U[:, t] + self.b + chol_Q @ np.random.standard_normal((n_trials, self.xdim, 1))
            y[:, t] = self.C @ x[:, t] + self.D @ U[:, t] + self.d_bias + chol_R @ np.random.standard_normal((n_trials, self.ydim, 1))

        return x, y
