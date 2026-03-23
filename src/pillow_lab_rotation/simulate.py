import numpy as np
from pillow_lab_rotation.lds import LinearDynamicalSystem


class LDSSim:
    def __init__(
            self,
            xdim: int,
            ydim: int
    ):
        self.xdim = xdim
        self.ydim = ydim

    def create_A_matrix(self):
        A = np.random.standard_normal((self.xdim, self.xdim))
        A /= (np.max(np.abs(np.linalg.eigvals(A))) + 0.5)
        self.A = A

    def create_C_matrix(self):
        self.C = np.random.standard_normal((self.ydim, self.xdim))

    def create_Q_matrix(self):
        L = np.random.standard_normal((self.xdim, self.xdim))
        self.Q = L @ L.T

    def create_Q0_matrix(self):
        L = np.random.standard_normal((self.xdim, self.xdim))
        self.Q0 = L @ L.T

    def create_mu0(self):
        self.mu0 = np.random.standard_normal((self.xdim, 1))

    def create_R_matrix(self):
        self.R = np.diag(np.random.uniform(0.5, 2.0, self.ydim))

    def get_A(self):
        return self.A

    def get_C(self):
        return self.C

    def get_Q(self):
        return self.Q

    def get_Q0(self):
        return self.Q0

    def get_mu0(self):
        return self.mu0

    def get_R(self):
        return self.R

    def get_params(self):
        return self.A, self.C, self.Q, self.Q0, self.mu0, self.R

    def create_params(self):
        self.create_A_matrix()
        self.create_C_matrix()
        self.create_Q_matrix()
        self.create_Q0_matrix()
        self.create_mu0()
        self.create_R_matrix()

    def simulate(self, T: int, n_trials: int = 1):
        lds = LinearDynamicalSystem(self.xdim, self.ydim)
        lds.A = self.A
        lds.C = self.C
        lds.Q = self.Q
        lds.Q0 = self.Q0
        lds.mu0 = self.mu0
        lds.R = self.R

        x, y = lds.sample(T, n_trials)
        return x, y





class CTDSSim:
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

    def create_A_matrix(self):
        A = np.zeros((self.D, self.D))
        A[:, :self.De] = np.random.uniform(0, 1, size=(self.D, self.De))
        A[:, self.De:] = -np.random.uniform(0, 1, size=(self.D, self.Di))
        A /= (np.max(np.abs(np.linalg.eigvals(A))) + 0.5)
        self.A = A

    def create_C_matrix(self):
        e2e_block = np.random.uniform(0, 1, (self.Ne, self.De))
        e2i_block = np.zeros((self.Ni, self.De))
        i2i_block = np.random.uniform(0, 1, (self.Ni, self.Di))
        i2e_block = np.zeros((self.Ne, self.Di))

        self.C = np.block(
            [[e2e_block, i2e_block],
             [e2i_block, i2i_block]]
        )

    def create_Q_matrix(self):
        L = np.random.standard_normal((self.D, self.D))
        self.Q = L @ L.T


    def create_Q0_matrix(self):
        L = np.random.standard_normal((self.D, self.D))
        self.Q0 = L @ L.T

    def create_mu0(self):
        self.mu0 = np.random.standard_normal((self.D, 1))

    def create_R_matrix(self):
        self.R = np.diag(np.random.uniform(0.5, 2.0, self.N))


    def get_A(self):
        return self.A

    def get_C(self):
        return self.C

    def get_Q(self):
        return self.Q

    def get_Q0(self):
        return self.Q0

    def get_mu0(self):
        return self.mu0

    def get_R(self):
        return self.R

    def get_params(self):
        return self.A, self.C, self.Q, self.Q0, self.mu0, self.R

    def create_params(self):
        self.create_A_matrix()
        self.create_C_matrix()
        self.create_Q_matrix()
        self.create_Q0_matrix()
        self.create_mu0()
        self.create_R_matrix()

    def simulate(self, T: int, n_trials: int = 1):
        lds = LinearDynamicalSystem(self.D, self.N)
        lds.A = self.A
        lds.C = self.C
        lds.Q = self.Q
        lds.Q0 = self.Q0
        lds.mu0 = self.mu0
        lds.R = self.R

        x, y = lds.sample(T, n_trials)
        return x, y
