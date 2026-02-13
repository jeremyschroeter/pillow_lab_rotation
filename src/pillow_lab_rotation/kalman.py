import numpy as np
from pillow_lab_rotation.dists import MultivariateNormal
inv = np.linalg.inv


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
            self.m_step()
            LL_new = self.log_likelihood()
            print(f'{LL_new:0.5f}')
            if LL_new < LL_old:
                raise ValueError('New LL less than old LL, implementation error')
            if LL_new - LL_old < 1e-5:
                break
            LL_old = LL_new

    def e_step(self):
        self.run_filter()
        self.run_smoother()
    
    def m_step(self):
        self.update_μ_and_V()
        self.update_A()
        self.update_Γ()
        self.update_C()
        self.update_R()
    
    def run_filter(self):
        
        self.z_filtered = np.zeros(shape=(self.n_trials, self.T, self.zdim, 1))
        self.P_filtered = np.zeros(shape=(self.n_trials, self.T, self.zdim, self.zdim))
        self.z_predicted = np.zeros(shape=(self.n_trials, self.T, self.zdim, 1))
        self.P_predicted = np.zeros(shape=(self.n_trials, self.T, self.zdim, self.zdim))

        for trial in range(self.n_trials):
            Pt = self.V0
            zt = self.μ0
            for t in range(self.T):
                
                # Predict
                if t == 0:
                    z_pred = zt
                    P_pred = Pt
                else:
                    z_pred = self.A @ zt
                    P_pred = self.A @ Pt @ self.A.T + self.Γ

                x_pred = self.C @ z_pred

                # Store prediction
                self.z_predicted[trial, t] = z_pred.copy()
                self.P_predicted[trial, t] = P_pred.copy()

                # Update
                xt = self.X[trial, t]
                K = P_pred @ self.C.T @inv(self.C @ P_pred @ self.C.T + self.R)
                z_filt = z_pred + K @ (xt - x_pred)
                P_filt = P_pred - K @ self.C @ P_pred

                # Store filtered updates
                self.z_filtered[trial, t] = z_filt
                self.P_filtered[trial, t] = P_filt
                zt = z_filt
                Pt = P_filt
    

    def run_smoother(self):

        self.Ez = np.zeros(shape=(self.n_trials, self.T, self.zdim, 1))
        self.EzzT = np.zeros(shape=(self.n_trials, self.T, self.zdim, self.zdim))
        self.Ezzm1T = np.zeros(shape=(self.n_trials, self.T, self.zdim, self.zdim))

        for trial in range(self.n_trials):
            z_smooth_prev = self.z_filtered[trial, -1]
            P_smooth_prev = self.P_filtered[trial, -1]
            self.Ez[trial, -1] = z_smooth_prev
            self.EzzT[trial, -1] = P_smooth_prev + z_smooth_prev @ z_smooth_prev.T

            for t in range(self.T - 2, -1, -1):
                z_from_filt = self.z_filtered[trial, t]
                P_from_filt = self.P_filtered[trial, t]
                z_from_pred = self.z_predicted[trial, t+1]
                P_from_pred = self.P_predicted[trial, t+1]

                J = P_from_filt @ self.A.T @ inv(P_from_pred)
                z_smooth = z_from_filt + J @ (z_smooth_prev - z_from_pred)
                P_smooth = P_from_filt + J @ (P_smooth_prev - P_from_pred) @ J.T

                self.Ez[trial, t] = z_smooth
                self.EzzT[trial, t] = P_smooth + z_smooth @ z_smooth.T
                self.Ezzm1T[trial, t+1] = P_smooth_prev @ J.T + z_smooth_prev @ z_smooth.T

                z_smooth_prev = z_smooth
                P_smooth_prev = P_smooth


    def update_μ_and_V(self):
        self.μ0 = self.Ez[:, 0].mean(0)
        self.V0 = self.EzzT[:, 0].mean(0) - self.μ0 @ self.μ0.T

    def update_Γ(self):
        Γ_new = np.zeros((self.zdim, self.zdim))
        for trial in range(self.n_trials):
            for t in range(1, self.T):
                Γ_first = self.EzzT[trial, t]
                Γ_second = self.Ezzm1T[trial, t] @ self.A.T
                Γ_third = self.A @ self.Ezzm1T[trial, t].T
                Γ_fourth = self.A @ self.EzzT[trial, t - 1] @ self.A.T
                Γ_new += Γ_first - Γ_second - Γ_third + Γ_fourth
        Γ_new /= (self.n_trials * (self.T - 1))
        self.Γ = Γ_new
    
    def update_A(self):
        A_new = np.zeros((2, self.zdim, self.zdim))
        for trial in range(self.n_trials):
            for t in range(1, self.T):
                A_new[0] += self.Ezzm1T[trial, t]
                A_new[1] += self.EzzT[trial, t - 1]
        self.A = A_new[0] @ inv(A_new[1])


    def update_R(self):
        R_new = np.zeros((self.xdim, self.xdim))
        for trial in range(self.n_trials):
            for t in range(self.T):
                xt = self.X[trial, t]
                zt = self.Ez[trial, t]
                R_first = xt @ xt.T
                R_second = xt @ zt.T @ self.C.T
                R_third = self.C @ zt @ xt.T
                R_fourth = self.C @ self.EzzT[trial, t] @ self.C.T
                R_new +=  R_first - R_second - R_third + R_fourth
        R_new /= (self.n_trials * self.T)
        self.R = R_new
    
    def update_C(self):
        C_new = np.zeros((2, self.xdim, self.zdim))
        for trial in range(self.n_trials):
            for t in range(self.T):
                C_new[0] += self.X[trial, t] @ self.Ez[trial, t].T
                C_new[1] += self.EzzT[trial, t]
        self.C = C_new[0] @ inv(C_new[1])

    def log_likelihood(self):
        LL = 0
        for trial in range(self.n_trials):
            Pt = self.V0
            zt = self.μ0
            for t in range(self.T):
    
                # Predict
                if t == 0:
                    z_pred = zt
                    P_pred = Pt
                else:
                    z_pred = self.A @ zt
                    P_pred = self.A @ Pt @ self.A.T + self.Γ

                x_pred = self.C @ z_pred
                P_obs = self.C @ P_pred @ self.C.T + self.R
                xt = self.X[trial, t]

                LL += MultivariateNormal(x_pred.squeeze(), P_obs).log_pdf(xt.squeeze())

                K = P_pred @ self.C.T @ inv(P_obs)
                zt = z_pred + K @ (xt - x_pred)
                Pt = P_pred - K @ self.C @ P_pred
        
        return LL / (self.n_trials * self.T)
    
    def predict(self, X: np.ndarray):
        
        trials, timesteps, _, _ = X.shape

        LL = 0
        pred_means = np.zeros(shape=(trials, timesteps, self.zdim, 1))
        pred_covs = np.zeros(shape=(trials, timesteps, self.zdim, self.zdim))
        obs_mean = np.zeros(shape=(trials, timesteps, self.xdim, 1))
        obs_cov = np.zeros(shape=(trials, timesteps, self.xdim, self.xdim))
        post_means = np.zeros(shape=(trials, timesteps, self.zdim, 1))
        post_covs = np.zeros(shape=(trials, timesteps, self.zdim, self.zdim))

        for trial in range(trials):
            Pt = self.V0
            zt = self.μ0
            for t in range(timesteps):
                if t == 0:
                    z_pred = zt
                    P_pred = Pt
                else:
                    z_pred = self.A @ zt
                    P_pred = self.A @ Pt @ self.A.T + self.Γ
                
                pred_means[trial, t] = z_pred
                pred_covs[trial, t] = P_pred
                
                x_pred = self.C @ z_pred
                P_obs = self.C @ P_pred @ self.C.T + self.R

                obs_mean[trial, t] = x_pred
                obs_cov[trial, t] = P_obs

                xt = X[trial, t]
                LL += MultivariateNormal(x_pred.squeeze(), P_obs).log_pdf(xt.squeeze())
                
                K = P_pred @ self.C.T @ inv(P_obs)
                zt = z_pred + K @ (xt - x_pred)
                Pt = P_pred - K @ self.C @ P_pred

                post_means[trial, t] = zt
                post_covs[trial, t] = Pt
        
        return pred_means, pred_covs, obs_mean, obs_cov, post_means, post_covs

