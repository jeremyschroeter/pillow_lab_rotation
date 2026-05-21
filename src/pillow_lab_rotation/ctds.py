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
            udim: int = 0,
            fit_mu0: bool = True,
            fit_b: bool = False,
            fit_d_bias: bool = False
    ):
        self.De = De
        self.Di = Di
        self.Ne = Ne
        self.Ni = Ni
        # D = total latent dim, N = total observation dim
        self.D_lat = De + Di
        self.N = Ne + Ni
        # Initialize the parent LDS with inputs to latents only (no feedthrough)
        super().__init__(
            xdim=self.D_lat,
            ydim=self.N,
            udim=udim,
            feedthrough=False,
            fit_mu0=fit_mu0,
            fit_b=fit_b,
            fit_d_bias=fit_d_bias,
        )
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
        A_mask = np.zeros((self.D_lat, self.D_lat))
        A_mask[:, :self.De] = 1
        A_mask[:, self.De:] = -1
        diag_idx = np.arange(self.D_lat)
        A_mask[diag_idx, diag_idx] = 0

        A_flat = A_mask.flatten(order='F')
        # Pad with zeros for the B portion so masks work on vec([A, B])
        B_pad = np.zeros(self.udim * self.D_lat)
        A_flat_padded = np.concatenate([A_flat, B_pad])
        self.A_pos_idx = A_flat_padded > 0
        self.A_neg_idx = A_flat_padded < 0

        # C constraints
        C_nonneg_mask = np.zeros((self.D_lat, self.D_lat), dtype=bool)
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
        # (can't call super().init_params() because it overwrites self.D_lat
        #  which is the latent dim integer here, not the feedthrough matrix)
        self.mu0 = np.random.standard_normal((self.D_lat, 1))
        self.Q0 = np.eye(self.D_lat)
        self.Q = np.eye(self.D_lat)
        self.B = np.random.randn(self.D_lat, self.udim) if self.udim > 0 else np.zeros((self.D_lat, self.udim))
        self.D = np.zeros((self.N, self.udim))  # feedthrough matrix; CTDS uses feedthrough=False so always zero
        self.b = np.zeros((self.D_lat, 1))
        self.d_bias = np.zeros((self.N, 1))

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
            self.A = np.zeros((self.D_lat, self.D_lat))

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
        self.update_b()
        self.update_d_bias()
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
        A = cp.Variable(self.D_lat * self.D_lat)
        Q_inv = inv(self.Q)
        Q_tilde = Q_inv / np.max(np.abs(Q_inv))
        L = np.linalg.cholesky(Q_tilde)

        # Effective cross-moment with dynamics bias: target is x_t - b.
        m_sum_1Tm1 = self.m_sum - self.m_sum_T
        M_delta_eff = self.M_delta - m_sum_1Tm1 @ self.b.T

        # Equation 80 from Adithis doc
        K = np.kron(np.eye(self.D_lat), L) @ np.kron(self.M1Tm1, np.eye(self.D_lat)) @ np.kron(np.eye(self.D_lat), L.T)
        q = vec(Q_tilde.T @ M_delta_eff.T)

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
        A_tilde = cp.Variable(self.D_lat * self.D_lat + self.udim * self.D_lat)
        Q_inv = inv(self.Q)
        Q_tilde = Q_inv / np.max(np.abs(Q_inv))
        L = np.linalg.cholesky(Q_tilde)

        # Effective cross-moments with dynamics bias: target is x_t - b.
        m_sum_1Tm1 = self.m_sum - self.m_sum_T
        u_sum_2T = self.u_sum - self.u_sum_1
        M_delta_eff = self.M_delta - m_sum_1Tm1 @ self.b.T
        U_hat_2T_eff = self.U_hat_2T - u_sum_2T @ self.b.T

        # See equation 87 in Adithis notes
        M_delta_tilde = np.vstack([M_delta_eff, U_hat_2T_eff])
        M_tilde_1Tm1 = np.block(
            [[self.M1Tm1, self.U_delta.T],
             [self.U_delta, self.U2T]]
        )

        K = np.kron(np.eye(self.D_lat + self.udim), L) @ np.kron(M_tilde_1Tm1, np.eye(self.D_lat)) @ np.kron(np.eye(self.D_lat + self.udim), L.T)
        q = vec(Q_tilde.T @ M_delta_tilde.T)

        objective = cp.Maximize(q.T @ A_tilde - 0.5 * cp.quad_form(A_tilde, cp.psd_wrap(K)))
        constraints = [
            A_tilde[self.A_pos_idx] >= 0,
            A_tilde[self.A_neg_idx] <= 0
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK, verbose=False)
        AB = A_tilde.value
        A = AB[:self.D_lat * self.D_lat]
        B = AB[self.D_lat * self.D_lat:]
        self.A = A.reshape(self.A.shape, order='F')
        self.B = B.reshape(self.D_lat, self.udim, order='F')



    def update_C(self):
        """Constrained update for C with block-diagonal non-negativity.

        Each row c_n of C is solved independently:
          - If n < Ne: c_n has De free (non-negative) entries, Di zeros
          - If n >= Ne: c_n has De zeros, Di free (non-negative) entries

        When fit_d_bias is True, c_n and d_n are updated *jointly* per row via an
        augmented predictor [x_t; 1] — matches a single closed-form OLS step
        instead of the block-coordinate (c then d) sweep. update_d_bias becomes
        a no-op in that case.
        """
        rows = []
        d_new = np.zeros((self.N, 1))
        T_N = self.T * self.n_trials
        for n in range(self.N):
            if n < self.Ne:
                block = slice(None, self.De)
                free_dim = self.De
            else:
                block = slice(self.De, None)
                free_dim = self.Di

            P_block = self.M1T[block, block]
            q_block = self.Y_hat[block, n]
            m_sum_block = self.m_sum[block, 0]

            if self.fit_d_bias:
                # Joint per-row OLS over (c_n, d_n) with c_n >= 0, d_n free.
                P = np.block([[P_block, m_sum_block[:, None]],
                              [m_sum_block[None, :], np.array([[T_N]])]])
                q = np.concatenate([q_block, [self.y_sum[n, 0]]])
                z = cp.Variable(free_dim + 1)
                objective = cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(P)) - q.T @ z)
                prob = cp.Problem(objective, [z[:free_dim] >= 0])
                prob.solve(solver=cp.MOSEK, verbose=False)
                c_val = z.value[:free_dim]
                d_new[n, 0] = z.value[free_dim]
            else:
                # d_n is held at its current value; subtract its contribution from q.
                d_n = self.d_bias[n, 0]
                q = q_block - m_sum_block * d_n
                c_n = cp.Variable(free_dim)
                objective = cp.Minimize(0.5 * cp.quad_form(c_n, cp.psd_wrap(P_block)) - q.T @ c_n)
                prob = cp.Problem(objective, [c_n >= 0])
                prob.solve(solver=cp.MOSEK, verbose=False)
                c_val = c_n.value
                d_new[n, 0] = self.d_bias[n, 0]

            if n < self.Ne:
                rows.append(np.hstack([c_val, np.zeros(self.Di)]))
            else:
                rows.append(np.hstack([np.zeros(self.De), c_val]))
        self.C = np.array(rows)
        if self.fit_d_bias:
            # d_bias was solved jointly with C above; the inherited update_d_bias
            # would otherwise re-update it via the block-coordinate formula.
            self.d_bias = d_new
            self._d_bias_already_updated = True
        else:
            self._d_bias_already_updated = False

    def update_d_bias(self):
        # When fit_d_bias=True and we already solved (c_n, d_n) jointly in
        # update_C, skip the block-coordinate update here so we keep the joint
        # optimum.
        if getattr(self, '_d_bias_already_updated', False):
            return
        super().update_d_bias()

    def update_R(self):
        """Update R with diagonal constraint.

        Delegate to LDS update_R (which handles the d_bias contribution via the
        full residual-variance formula), then zero the off-diagonal.
        """
        super().update_R()
        self.R = np.diag(np.diag(self.R))
