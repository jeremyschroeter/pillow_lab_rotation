import numpy as np
from numpy.linalg import inv
import cvxpy as cp
from pillow_lab_rotation.dists import MultivariateNormal

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
    

    def fit(self, Y: np.ndarray):
        '''
        Assume Y is shape (n_trials, T, ydim, 1)
        '''
        self.Y = Y
        self.n_trials, self.T, _, _ = Y.shape

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
        Standard Kalman filtering, implementation lifted from lds.py
        '''
        self.x_filtered = np.zeros(shape=(self.n_trials, self.T, self.D, 1))
        self.P_filtered = np.zeros(shape=(self.n_trials, self.T, self.D, self.D))
        self.x_predicted = np.zeros(shape=(self.n_trials, self.T, self.D, 1))
        self.P_predicted = np.zeros(shape=(self.n_trials, self.T, self.D, self.D))

        # Iterate over trials
        for trial in range(self.n_trials):
            xt = self.mu0
            Pt = self.V0

            # Iterate over timesteps
            for t in range(self.T):

                # Predict:
                if t == 0:
                    x_pred = xt
                    P_pred = Pt
                else:
                    x_pred = self.A @ xt
                    P_pred = self.A @ Pt @ self.A.T + self.Q
                
                y_pred = self.C @ x_pred

                # Store predictions
                self.x_predicted[trial, t] = x_pred.copy()
                self.P_predicted[trial, t] = P_pred.copy()

                # Update
                yt = self.Y[trial, t]
                K = P_pred @ self.C.T @ inv(self.C @ P_pred @ self.C.T + self.R)
                x_filt = x_pred + K @ (yt - y_pred)
                P_filt = P_pred - K @ self.C @ P_pred

                # Store filtered updates
                self.x_filtered[trial, t] = x_filt
                self.P_filtered[trial, t] = P_filt
                xt = x_filt
                Pt = P_filt

    
    def run_smoother(self):
        '''
        Standard Kalman smoothing, implementation lifted from lds.py
        '''

        # Init sufficient statistics needed for updating
        self.Ex = np.zeros(shape=(self.n_trials, self.T, self.D, 1))
        self.ExxT = np.zeros(shape=(self.n_trials, self.T, self.D, self.D))
        self.Exxm1T = np.zeros(shape=(self.n_trials, self.T, self.D, self.D))

        # Iterate over trials
        for trial in range(self.n_trials):
            x_smooth_prev = self.x_filtered[trial, -1]
            P_smooth_prev = self.P_filtered[trial, -1]
            self.Ex[trial, -1] = x_smooth_prev
            self.ExxT[trial, -1] = P_smooth_prev + x_smooth_prev @ x_smooth_prev.T

            for t in range(self.T-2, -1, -1):
                x_filt = self.x_filtered[trial, t]
                P_filt = self.P_filtered[trial, t]
                x_pred = self.x_predicted[trial, t+1]
                P_pred = self.P_predicted[trial, t+1]

                J = P_filt @ self.A.T @ inv(P_pred)
                x_smooth = x_filt + J @ (x_smooth_prev - x_pred)
                P_smooth = P_filt + J @ (P_smooth_prev - P_pred) @ J.T

                self.Ex[trial, t] = x_smooth
                self.ExxT[trial, t] = P_smooth + x_smooth @ x_smooth.T
                self.Exxm1T[trial, t] = P_smooth_prev @ J.T + x_smooth_prev @ x_smooth.T

                x_smooth_prev = x_smooth
                P_smooth_prev = P_smooth


    def update_mu(self):
        self.mu0 = self.Ex[:, 0].mean(0)
    
    def update_V(self):
        self.V0 = self.ExxT[:, 0].mean(0) - self.mu0 @ self.mu0.T
    
    def update_A(self):
        self.A_var = cp.Variable(shape=(self.D, self.D))
        

    def _A_objective(self, A_var):
        cur_ExxT = self.ExxT[:, 1:].reshape(-1, self.D, self.D).sum(0)
        cur_Exxm1T = self.Exxm1T[:, 1:].reshape(-1, self.D, self.D).swapaxes(1, 2).sum(0)
        cur_ExxT2 = self.ExxT[:, :-1].reshape(-1, self.D, self.D).sum(0)
        Q_inv = inv(self.Q)

        first_term = Q_inv @ cur_ExxT
        second_term = 2 * Q_inv @ A_var @ cur_Exxm1T
        third_term = A_var.T @ Q_inv @ A_var @ cur_ExxT2

        return -0.5 * np.trace(first_term - second_term + third_term)
    


        

    
    def update_Q(self):
        NotImplemented
    
    def update_C(self):
        NotImplemented
    
    def update_R(self):
        NotImplemented

    
    def log_likelihood(self):
        NotImplemented