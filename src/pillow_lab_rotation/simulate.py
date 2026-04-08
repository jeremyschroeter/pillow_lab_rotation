import numpy as np
from pillow_lab_rotation.lds import LinearDynamicalSystem


def generate_pulsatile_inputs(n_trials, T, udim, pulse_prob=0.04):
    """Random binary pulses: each timestep has pulse_prob chance of a 1-2 step pulse."""
    U = np.zeros((n_trials, T, udim, 1))
    for n in range(n_trials):
        for t in range(T):
            if np.random.rand() < pulse_prob:
                pulse = np.random.randint(0, 2, size=(udim, 1)).astype(float)
                duration = np.random.choice([1, 2])
                U[n, t:min(t + duration, T), :, :] = pulse
    return U


def generate_gaussian_inputs(n_trials, T, udim, mean=0.0, std=1.0):
    """IID Gaussian noise inputs at each timestep."""
    return np.random.normal(mean, std, size=(n_trials, T, udim, 1))


class LDSSim:
    def __init__(
            self,
            xdim: int,
            ydim: int,
            udim: int = 0,
            feedthrough: bool = True
    ):
        self.xdim = xdim
        self.ydim = ydim
        self.udim = udim
        self.feedthrough = feedthrough

    def create_A_matrix(self):
        A = np.eye(self.xdim) - 0.1
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

    def create_B_matrix(self):
        self.B = np.random.standard_normal((self.xdim, self.udim))

    def create_D_matrix(self):
        if self.feedthrough:
            self.D = np.random.standard_normal((self.ydim, self.udim))
        else:
            self.D = np.zeros((self.ydim, self.udim))

    def create_R_matrix(self):
        self.R = np.diag(np.random.uniform(0.5, 2.0, self.ydim))

    def get_B(self):
        return self.B

    def get_D(self):
        return self.D

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
        if self.udim > 0 and self.feedthrough:
            return self.A, self.C, self.Q, self.Q0, self.mu0, self.R, self.B, self.D
        elif self.udim > 0:
            return self.A, self.C, self.Q, self.Q0, self.mu0, self.R, self.B
        return self.A, self.C, self.Q, self.Q0, self.mu0, self.R

    def create_params(self):
        self.create_A_matrix()
        self.create_C_matrix()
        self.create_Q_matrix()
        self.create_Q0_matrix()
        self.create_mu0()
        self.create_R_matrix()
        if self.udim > 0:
            self.create_B_matrix()
            self.create_D_matrix()

    def simulate(self, T: int, n_trials: int = 1, inputs: np.ndarray | None = None):
        lds = LinearDynamicalSystem(self.xdim, self.ydim, udim=self.udim, feedthrough=self.feedthrough)
        lds.A = self.A
        lds.C = self.C
        lds.Q = self.Q
        lds.Q0 = self.Q0
        lds.mu0 = self.mu0
        lds.R = self.R
        if self.udim > 0:
            lds.B = self.B
            lds.D = self.D

        x, y = lds.sample(T, n_trials, inputs=inputs)
        return x, y



class CTDSSim(LDSSim):
    def __init__(
            self,
            De: int,
            Di: int,
            Ne: int,
            Ni: int,
            udim: int = 0
    ):
        self.De = De
        self.Di = Di
        self.Ne = Ne
        self.Ni = Ni
        self.D = De + Di
        self.N = Ne + Ni
        super().__init__(xdim=self.D, ydim=self.N, udim=udim, feedthrough=False)

    def create_A_matrix(self):
        A = np.zeros((self.D, self.D))
        A[:, :self.De] = np.random.uniform(0, 1, size=(self.D, self.De))
        A[:, self.De:] = -np.random.uniform(0, 1, size=(self.D, self.Di))
        A /= (np.max(np.abs(np.linalg.eigvals(A))) + 1.0)
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

    def create_R_matrix(self):
        self.R = np.diag(np.random.uniform(0.5, 2.0, self.N))
