import numpy as np
import cvxpy as cp

import jax.numpy as jnp
from jax import Array
from jax import random

from pillow_lab_rotation.tools import vec
from pillow_lab_rotation.jax_lds import LinearDynamicalSystemJAX
from pillow_lab_rotation.eirnn import EIRNNInit


class CTDSJax(LinearDynamicalSystemJAX):
    def __init__(
            self,
            De: int,
            Di: int,
            Ne: int,
            Ni: int,
            udim: int = 0,
            key: Array | None = None
    ):
        self.De = De
        self.Di = Di
        self.Ne = Ne
        self.Ni = Ni
        # D = total latent dim, N = total observation dim
        self.D_lat = De + Di
        self.N = Ne + Ni

        # Initialize the parent LDS with inputs to latents only (no feedthrough)
        super().__init__(xdim=self.D_lat, ydim=self.N, udim=udim, feedthrough=False, key=key)
        self.init_constraints()

    def init_constraints(self):
        """
        Build index masks for the sign constraints on A and C.

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
    

    def init_params(
            self,
            observations: Array | None = None,
            start_seed: int = 0
    ):
        keys = random.split(self.key, 4)

        self.mu0 = random.normal(keys[0], (self.D_lat, 1))
        self.Q0 = jnp.eye(self.D_lat)
        self.Q = jnp.eye(self.D_lat)
        self.B = random.normal(keys[1], (self.D_lat, self.udim)) if self.udim > 0 else jnp.zeros((self.D_lat, self.udim))
        self.D = jnp.zeros((self.N, self.udim))  # feedthrough matrix; CTDS uses feedthrough=False so always zero

        if observations is not None:
            init = EIRNNInit(self.Ne, self.Ni, self.De, self.Di)
            init.fit(np.asarray(observations), start_seed=start_seed)
            self.A = jnp.asarray(init.A)
            self.C = jnp.asarray(init.C)
            self.R = jnp.asarray(init.R)
            self.Q = jnp.asarray(init.Q)
            self.mu0 = jnp.asarray(init.mu0)
            self.Q0 = jnp.asarray(init.Q0)

        else:
            self.A = jnp.zeros((self.D_lat, self.D_lat))

            e2e_block = random.uniform(keys[2], shape=(self.Ne, self.De))
            e2i_block = jnp.zeros((self.Ni, self.De))
            i2i_block = random.uniform(keys[3], (self.Ni, self.Di))
            i2e_block = jnp.zeros((self.Ne, self.Di))

            self.C = jnp.block(
                [[e2e_block, i2e_block],
                [e2i_block, i2i_block]]
            )
            self.R = jnp.eye(self.N)
    
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
        '''
        Constrained upate for A using QP

        Need to convert jax arrays to np arrays to feed to
        cvxpy
        '''

        Q_np = np.asarray(self.Q)
        M1Tm1_np = np.asarray(self.M1Tm1)
        M_delta_np = np.asarray(self.M_delta)

        Q_inv = np.linalg.inv(Q_np)
        Q_tilde = Q_inv / np.max(np.abs(Q_inv))
        L = np.linalg.cholesky(Q_tilde)

        K = np.kron(np.eye(self.D_lat), L) @ np.kron(M1Tm1_np, np.eye(self.D_lat)) @ np.kron(np.eye(self.D_lat), L.T)
        q = vec(Q_tilde.T @ M_delta_np.T)

        A = cp.Variable(self.D_lat * self.D_lat)
        objective = cp.Maximize(q.T @ A - 0.5 * cp.quad_form(A, cp.psd_wrap(K)))
        constraints = [
            A[self.A_pos_idx] >= 0,
            A[self.A_neg_idx] <= 0
        ]
        cp.Problem(objective, constraints).solve(solver=cp.MOSEK, verbose=False)
        self.A = jnp.asarray(A.value.reshape((self.D_lat, self.D_lat), order='F'))
    
    def update_A_B(self):
        """Constrained joint update for A and B.

        Same QP structure as update_A, but the optimization variable is
        vec([A, B]) and the sufficient statistics include input terms.
        Sign constraints apply only to the A portion of the variable.
        B is unconstrained.
        """
        # Equation 86 from Adithis notes
        Q_np = np.asarray(self.Q)
        M_delta_np = np.asarray(self.M_delta)
        M1Tm1_np = np.asarray(self.M1Tm1)
        U_hat_2T_np = np.asarray(self.U_hat_2T)
        U_delta_np = np.asarray(self.U_delta)
        U2T_np = np.asarray(self.U2T)

        Q_inv = np.linalg.inv(Q_np)
        Q_tilde = Q_inv / np.max(np.abs(Q_inv))
        L = np.linalg.cholesky(Q_tilde)

        # See equation 87 in Adithis notes
        M_delta_tilde = np.vstack([M_delta_np, U_hat_2T_np])
        M_tilde_1Tm1 = np.block(
            [[M1Tm1_np, U_delta_np.T],
             [U_delta_np, U2T_np]]
        )

        K = np.kron(np.eye(self.D_lat + self.udim), L) @ np.kron(M_tilde_1Tm1, np.eye(self.D_lat)) @ np.kron(np.eye(self.D_lat + self.udim), L.T)
        q = vec(Q_tilde.T @ M_delta_tilde.T)

        A_tilde = cp.Variable(self.D_lat * self.D_lat + self.udim * self.D_lat)
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
        self.A = jnp.asarray(A.reshape((self.D_lat, self.D_lat), order='F'))
        self.B = jnp.asarray(B.reshape(self.D_lat, self.udim, order='F'))

    def update_C(self):
        """Constrained update for C with block-diagonal non-negativity.

        Each row c_n of C is solved independently:
          - If n < Ne: c_n has De free (non-negative) entries, Di zeros
          - If n >= Ne: c_n has De zeros, Di free (non-negative) entries

        Minimize:  0.5 c_n^T P c_n - q^T c_n   s.t.  c_n >= 0
        where P and q are the relevant sub-blocks of M1T and Y_hat.
        """
        M1T_np = np.asarray(self.M1T)
        Y_hat_np = np.asarray(self.Y_hat)


        rows = []
        for n in range(self.N):
            if n < self.Ne:
                c_n = cp.Variable(self.De)
                P = M1T_np[:self.De, :self.De]
                q = Y_hat_np[:self.De, n]
            else:
                c_n = cp.Variable(self.Di)
                P = M1T_np[self.De:, self.De:]
                q = Y_hat_np[self.De:, n]
            objective = cp.Minimize(0.5 * cp.quad_form(c_n, cp.psd_wrap(P)) - q.T @ c_n)
            prob = cp.Problem(objective, [c_n >= 0])
            prob.solve(solver=cp.MOSEK, verbose=False)
            if n < self.Ne:
                rows.append(np.hstack([c_n.value, np.zeros(self.Di)]))
            else:
                rows.append(np.hstack([np.zeros(self.De), c_n.value]))
        self.C = jnp.array(rows)

    def update_R(self):
        """Update R with diagonal constraint.

        Same formula as the LDS update_R, but then set off-diagonal to zero:
            R = diag(diag(R))
        """
        super().update_R()
        self.R = jnp.diag(jnp.diag(self.R))