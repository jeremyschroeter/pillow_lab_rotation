import numpy as np
import cvxpy as cp
from numpy.linalg import inv
from pillow_lab_rotation.tools import vec
from pillow_lab_rotation.lds import LinearDynamicalSystem
from pillow_lab_rotation.eirnn import EIRNNInit



class CTDS(LinearDynamicalSystem):
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
        # D = total latent dim, N = total observation dim
        self.D = De + Di
        self.N = Ne + Ni
        # Initialize the parent LDS with inputs to latents only (no feedthrough)
        super().__init__(xdim=self.D, ydim=self.N, udim=udim, feedthrough=False)
        self.init_constraints()

    def init_constraints(self):
        """Build index masks for the sign constraints on A and C.

        A constraints (Dale's law):
          - Excitatory columns (first De cols): off-diagonal entries >= 0
          - Inhibitory columns (last Di cols): off-diagonal entries <= 0
          - Diagonal entries: unconstrained (or zero — your choice)

        C constraints:
          - Block diagonal: C[:Ne, :De] >= 0, C[Ne:, De:] >= 0
          - Off-diagonal blocks are zero
        """

        # A constraints
        A_mask = np.zeros((self.D, self.D))
        A_mask[:, :self.De] = 1
        A_mask[:, self.De:] = -1
        diag_idx = np.arange(self.D)
        A_mask[diag_idx, diag_idx] = 0

        A_flat = A_mask.flatten(order='F')
        # Pad with zeros for the B portion so masks work on vec([A, B])
        B_pad = np.zeros(self.udim * self.D)
        A_flat_padded = np.concatenate([A_flat, B_pad])
        self.A_pos_idx = A_flat_padded > 0
        self.A_neg_idx = A_flat_padded < 0

        # C constraints
        C_nonneg_mask = np.zeros((self.D, self.D), dtype=bool)
        C_nonneg_mask[:self.Ne, :self.De] = True
        C_nonneg_mask[self.Ne:, self.De:] = True

        self.C_nonneg_idx = C_nonneg_mask.flatten(order='F')
        self.C_zero_idx = ~C_nonneg_mask.flatten(order='F')

    def init_params(self, observations: np.ndarray|None=None, start_seed: int=0):
        """
        Initialize CTDS parameters.

        If observations are provided, uses a structured initialization:
            1. Regress J from consecutive observations: y_{t+1} = J y_t
            subject to Dale's law sign constraints on columns of J.
            2. Decompose J ≈ UV via NMF split by cell type, yielding
            C = U  (N x D, block-diagonal non-negative)
            A = VU (D x D, Dale's law signs)
            3. Invert observations through C to estimate latent states,
            then fit mu0, Q0, Q, and R from residual statistics.

        If observations are None, falls back to random initialization
        respecting the structural constraints.

        Parameters
        ----------
        observations : np.ndarray or None
            Shape (n_trials, T, N, 1). Neural recordings used for
            data-driven initialization. If None, uses random init.
        start_seed : int
            Base random seed for NMF restarts. Each of the 10 restarts
            uses start_seed + i.
        """
        # Base params shared by both branches
        # (can't call super().init_params() because it overwrites self.D
        #  which is the latent dim integer here, not the feedthrough matrix)
        self.mu0 = np.random.standard_normal((self.D, 1))
        self.Q0 = np.eye(self.D)
        self.Q = np.eye(self.D)
        self.B = np.random.randn(self.D, self.udim) if self.udim > 0 else np.zeros((self.D, self.udim))

        if observations is not None:
            init = EIRNNInit(self.Ne, self.Ni, self.De, self.Di)
            init.fit(observations, start_seed=start_seed)
            self.A = init.A
            self.C = init.C
            self.R = init.R
            self.Q = init.Q
            self.mu0 = init.mu0
            self.Q0 = init.Q0

        else:
            self.A = np.zeros((self.D, self.D))

            e2e_block = np.random.uniform(0, 1, (self.Ne, self.De))
            e2i_block = np.zeros((self.Ni, self.De))
            i2i_block = np.random.uniform(0, 1, (self.Ni, self.Di))
            i2e_block = np.zeros((self.Ne, self.Di))

            self.C = np.block(
                [[e2e_block, i2e_block],
                [e2i_block, i2i_block]]
            )
            self.R = np.eye(self.N)

    # -------------------------------------------------------------------------
    # Inherited from LDS (no changes needed):
    #   - fit()
    #   - e_step()
    #   - run_filter()          (handles inputs via B)
    #   - run_smoother()
    #   - _get_sufficient_stats()  (computes input-related stats when udim > 0)
    #   - update_mu_and_Q0()     (handles Q0 and mu0, accounts for B)
    #   - update_Q()            (accounts for B when udim > 0)
    #   - predict()
    #   - sample()
    # -------------------------------------------------------------------------

    def m_step(self):
        """Override to call the constrained update methods.

        When udim > 0, A and B are updated jointly via update_A_B().
        Otherwise, A is updated alone via update_A().
        C always uses the constrained update (no feedthrough, so no D).
        """
        if self.udim > 0:
            self.update_A_B()
            self.update_C()
        else:
            self.update_A()
            self.update_C()
        self.update_mu_and_Q0()
        self.update_Q()
        self.update_R()

    def update_A(self):
        """Constrained update for A using quadratic programming (Dale's law).

        Maximize:  vec(Q^{-1} M_delta)^T vec(A) - 0.5 vec(A)^T K vec(A)
        where K = I ⊗ (Q^{-1} M_{1:T-1})
        subject to sign constraints from self.A_pos_idx, self.A_neg_idx.

        Hint: use cvxpy with cp.quad_form and index into the flattened
        (column-major / Fortran order) variable.
        """
        # Solve QP problem: A.T K A + q A
        A = cp.Variable(self.D * self.D)
        Q_inv = inv(self.Q)
        Q_tilde = Q_inv / np.max(np.abs(Q_inv))
        L = np.linalg.cholesky(Q_tilde)

        # Equation 80 from Adithis doc
        K = np.kron(np.eye(self.D), L) @ np.kron(self.M1Tm1, np.eye(self.D)) @ np.kron(np.eye(self.D), L.T)
        q = vec(Q_tilde.T @ self.M_delta.T)

        objective = cp.Maximize(q.T @ A - 0.5 * cp.quad_form(A, cp.psd_wrap(K)))
        constraints = [
            A[self.A_pos_idx] >= 0,
            A[self.A_neg_idx] <= 0
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, verbose=False)
        self.A = A.value.reshape(self.A.shape, order='F')

    def update_A_B(self):
        """Constrained joint update for A and B.

        Same QP structure as update_A, but the optimization variable is
        vec([A, B]) and the sufficient statistics include input terms.
        Sign constraints apply only to the A portion of the variable.
        B is unconstrained.
        """
        # Equation 86 from Adithis notes
        A_tilde = cp.Variable(self.D * self.D + self.udim * self.D)
        Q_inv = inv(self.Q)
        Q_tilde = Q_inv / np.max(np.abs(Q_inv))
        L = np.linalg.cholesky(Q_tilde)

        # See equation 87 in Adithis notes
        M_delta_tilde = np.vstack([self.M_delta, self.U_hat_2T])
        M_tilde_1Tm1 = np.block(
            [[self.M1Tm1, self.U_delta.T],
             [self.U_delta, self.U2T]]
        )

        K = np.kron(np.eye(self.D + self.udim), L) @ np.kron(M_tilde_1Tm1, np.eye(self.D)) @ np.kron(np.eye(self.D + self.udim), L.T)
        q = vec(Q_tilde.T @ M_delta_tilde.T)

        objective = cp.Maximize(q.T @ A_tilde - 0.5 * cp.quad_form(A_tilde, cp.psd_wrap(K)))
        constraints = [
            A_tilde[self.A_pos_idx] >= 0,
            A_tilde[self.A_neg_idx] <= 0
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, verbose=False)
        AB = A_tilde.value
        A = AB[:self.D * self.D]
        B = AB[self.D * self.D:]
        self.A = A.reshape(self.A.shape, order='F')
        self.B = B.reshape(self.D, self.udim, order='F')



    def update_C(self):
        """Constrained update for C with block-diagonal non-negativity.

        Each row c_n of C is solved independently:
          - If n < Ne: c_n has De free (non-negative) entries, Di zeros
          - If n >= Ne: c_n has De zeros, Di free (non-negative) entries

        Minimize:  0.5 c_n^T P c_n - q^T c_n   s.t.  c_n >= 0
        where P and q are the relevant sub-blocks of M1T and Y_hat.
        """
        rows = []
        for n in range(self.N):
            if n < self.Ne:
                c_n = cp.Variable(self.De)
                P = self.M1T[:self.De, :self.De]
                q = self.Y_hat[:self.De, n]
            else:
                c_n = cp.Variable(self.Di)
                P = self.M1T[self.De:, self.De:]
                q = self.Y_hat[self.De:, n]
            objective = cp.Minimize(0.5 * cp.quad_form(c_n, cp.psd_wrap(P)) - q.T @ c_n)
            prob = cp.Problem(objective, [c_n >= 0])
            prob.solve(solver=cp.MOSEK, verbose=False)
            if n < self.Ne:
                rows.append(np.hstack([c_n.value, np.zeros(self.Di)]))
            else:
                rows.append(np.hstack([np.zeros(self.De), c_n.value]))
        self.C = np.array(rows)

    def update_R(self):
        """Update R with diagonal constraint.

        Same formula as the LDS update_R, but then set off-diagonal to zero:
            R = diag(diag(R))
        """
        normalizer = (1 / (self.T * self.n_trials))
        unnormalized = (self.Y - self.C@self.Y_hat - self.Y_hat.T @ self.C.T + self.C @ self.M1T @ self.C.T)

        self.R = np.diag(np.diag(normalizer * unnormalized))
