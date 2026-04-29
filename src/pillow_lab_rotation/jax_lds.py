import jax
import jax.numpy as jnp
from jax import random, Array
from jax.numpy.linalg import inv, slogdet

from functools import partial

# ------------------------------------------------------------------
# E-step
# ------------------------------------------------------------------
@partial(jax.jit, static_argnames=('n_trials', 'T', 'xdim', 'ydim'))
def _run_filter(
        A: Array,
        B: Array,
        C: Array,
        D: Array,
        Q: Array,
        R: Array,
        mu0: Array,
        Q0: Array,
        observations: Array,
        inputs: Array,
        n_trials: int,
        T: int,
        xdim: int,
        ydim: int
) -> tuple[Array]:

    # Carry: predicted mu, P at the current timestep
    # Init carry: the prior at t=1: mu0, Q0
    def _cov_step(P_pred: Array, _: None):
        P_obs = C @ P_pred @ C.T + R
        P_obs_inv = inv(P_obs)
        _, log_det = slogdet(P_obs)
        K = P_pred @ C.T @ P_obs_inv
        P_filt = P_pred - K @ C @ P_pred

        # Predict step for the next timepoint
        P_pred_next = A @ P_filt @ A.T + Q
        return P_pred_next, (P_pred, P_obs_inv, log_det, K, P_filt)
    
    
    # Carry: previous filtered state x_{t-1|t-1}, running LL, timestep counter.
    # Init: x_prev set to mu0 (its value is ignored at t=0 by the where below).
    def _mean_step(carry: tuple[Array], inputs: tuple[Array]):
        x_prev, LL, t = carry
        yt, ut, K_t, P_obs_inv_t, log_det_t = inputs

        # Prediction for the CURRENT timestep, using the current input.
        # At t=0: xp = mu0 + B @ u_0; at t>=1: xp = A @ x_{t-1|t-1} + B @ u_t.
        xp = jnp.where(
            t == 0,
            mu0 + B @ ut,
            A @ x_prev + B @ ut,
        )

        # Update step
        innov = yt - C @ xp - D @ ut
        xt = xp + K_t @ innov

        # LL accumulation
        quad = (innov[..., 0] @ P_obs_inv_t * innov[..., 0]).sum()
        LL_new = LL + -0.5 * (n_trials * (ydim * jnp.log(2*jnp.pi) + log_det_t) + quad)

        return (xt, LL_new, t + 1), (xp, xt)

    # Covariance pass
    _, (P_pred_all, P_obs_inv_all, log_det_all, K_all, P_filt_all) = jax.lax.scan(
            _cov_step,
            init=Q0,
            xs=None,
            length=T
    )

    # Mean pass
    (_, LL_total, _), (x_pred, x_filt) = jax.lax.scan(
        _mean_step,
        init=(
            jnp.broadcast_to(mu0, (n_trials, xdim, 1)),
            jnp.array(0.0),
            jnp.array(0),
        ),
        xs=(observations.swapaxes(0, 1),
            inputs.swapaxes(0, 1),
            K_all, P_obs_inv_all, log_det_all),
    )

    P_predicted = jnp.broadcast_to(P_pred_all, (n_trials, T, xdim, xdim))
    P_filtered = jnp.broadcast_to(P_filt_all, (n_trials, T, xdim, xdim))
    x_predicted = x_pred.swapaxes(0, 1)
    x_filtered  = x_filt.swapaxes(0, 1)
    LL = LL_total / (n_trials * T)
    
    return P_predicted, P_filtered, x_predicted, x_filtered, LL

@partial(jax.jit, static_argnames=('n_trials', 'T', 'xdim'))
def _run_smoother(
        P_predicted: Array,
        P_filtered: Array,
        x_predicted: Array,
        x_filtered: Array,
        A: Array,
        n_trials: int,
        T: int,
        xdim: int
):
    
    def _smoother_step(carry: tuple[jax.Array], inputs: tuple[jax.Array]):
        P_smooth_tp1, m_tp1 = carry
        P_filt_t, P_pred_tp1, x_filt_t, x_pred_tp1 = inputs
        J = P_filt_t @ A.T @ inv(P_pred_tp1)
        P_smooth_t = P_filt_t + J @ (P_smooth_tp1 - P_pred_tp1) @ J.T
        sigma_x_tp1 = J @ P_smooth_tp1
        m_t = x_filt_t + J @ (m_tp1 - x_pred_tp1)
        return (P_smooth_t, m_t), (P_smooth_t, sigma_x_tp1, m_t, J)
    
    
    P_filt_all = P_filtered[0]
    P_pred_all = P_predicted[0]
    P_filt_xs = P_filt_all[:-1]
    P_pred_xs = P_pred_all[1:]
    x_filt_xs = x_filtered.swapaxes(0, 1)[:-1]
    x_pred_xs = x_predicted.swapaxes(0, 1)[1:]

    _, (P_smooth_inner, sigma_x_inner, m_inner, _) = jax.lax.scan(
        _smoother_step,
        init=(P_filt_all[-1], x_filtered[:, -1]),
        xs=(P_filt_xs, P_pred_xs, x_filt_xs, x_pred_xs),
        reverse=True
    )

    # Stitch on boundary at t = T-1
    P_smooth_all = jnp.concatenate(
        [P_smooth_inner, P_filt_all[-1][None]], axis=0
    )
    m_all = jnp.concatenate(
        [m_inner.swapaxes(0, 1), x_filtered[:, -1:]], axis=1
    )
    sigma_x_all = jnp.concatenate(
        [jnp.zeros((1, xdim, xdim)), sigma_x_inner], axis=0
    )

    m = m_all
    sigma = jnp.broadcast_to(P_smooth_all, (n_trials, T, xdim, xdim))
    sigma_x = jnp.broadcast_to(sigma_x_all, (n_trials, T, xdim, xdim))

    return m, sigma, sigma_x

@partial(jax.jit, static_argnames=('xdim',))
def _get_posterior_stats(
        mean: Array,
        sigma: Array,
        sigma_x: Array,
        observations: Array,
        xdim: int
) -> tuple[Array]:
    
    def _second_moment(m_slice: Array, sigma_slice: Array):
        flat = m_slice.reshape(-1, xdim)
        return flat.T @ flat + sigma_slice.reshape(-1, xdim, xdim).sum(0)
    
    def _cross_moment(a: Array, b: Array):
        return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])
    
    m = mean[..., 0]
    y = observations[..., 0]
    
    M11 = _second_moment(m[:, :1], sigma[:, :1])
    M2T = _second_moment(m[:, 1:], sigma[:, 1:])
    M1Tm1 = _second_moment(m[:, :-1], sigma[:, :-1])
    M1T = M11 + M2T

    M_delta = _cross_moment(m[:, :-1], m[:, 1:]) + sigma_x[:, 1:].reshape(-1, xdim, xdim).sum(0)

    Y = _cross_moment(y, y)
    Y_hat = _cross_moment(m, y)

    return M11, M2T, M1Tm1, M1T, M_delta, Y, Y_hat


@partial(jax.jit, static_argnames=('xdim',))
def _get_posterior_stats_w_inputs(
        mean: Array,
        sigma: Array,
        sigma_x: Array,
        observations: Array,
        inputs: Array,
        xdim: int
) -> tuple[Array]:
    m = mean[..., 0]
    y = observations[..., 0]
    u = inputs[..., 0]

    def _second_moment(m_slice: jax.Array, sigma_slice: jax.Array):
        flat = m_slice.reshape(-1, xdim)
        return flat.T @ flat + sigma_slice.reshape(-1, xdim, xdim).sum(0)
    
    def _cross_moment(a: jax.Array, b: jax.Array):
        return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])

    M11 = _second_moment(m[:, :1], sigma[:, :1])
    M2T = _second_moment(m[:, 1:], sigma[:, 1:])
    M1Tm1 = _second_moment(m[:, :-1], sigma[:, :-1])
    M1T = M11 + M2T

    M_delta = _cross_moment(m[:, :-1], m[:, 1:]) + sigma_x[:, 1:].reshape(-1, xdim, xdim).sum(0)

    Y = _cross_moment(y, y)
    Y_hat = _cross_moment(m, y)

    # Need statistics from inputs as well
    U1T = _cross_moment(u, u)
    U2T = _cross_moment(u[:, 1:], u[:, 1:])
    U_hat_2T = _cross_moment(u[:, 1:], m[:, 1:])
    Uy = _cross_moment(u, y)
    U_delta = _cross_moment(u[:, 1:], m[:, :-1])
    U_hat_1T = _cross_moment(u, m)
    U11 = _cross_moment(u[:, :1], u[:, :1])
    U_hat_11 = _cross_moment(u[:, :1], m[:, :1])

    return M11, M2T, M1Tm1, M1T, M_delta, Y, Y_hat, U1T, U2T, U_hat_2T, Uy, U_delta, U_hat_1T, U11, U_hat_11

# ------------------------------------------------------------------
# M-step
# ------------------------------------------------------------------
@jax.jit
def _update_A(
        M_delta: Array,
        M1Tm1: Array
) -> Array:
    '''
    Update dynamics matrix given posterior statistics
    '''
    return M_delta.T @ inv(M1Tm1)


@partial(jax.jit, static_argnames=('xdim',))
def _update_A_B(
        M_delta: Array,
        M1Tm1: Array,
        U_hat_2T: Array,
        U_delta: Array,
        U2T: Array,
        xdim: int
) -> tuple[Array]:
    '''
    Update dynamics and inputs matrix given posterior statistics
    '''
    first_matrix = jnp.block([M_delta.T, U_hat_2T.T])
    second_matrix = inv(jnp.block(
        [[M1Tm1, U_delta.T],
         [U_delta, U2T]]
    ))
    AB = first_matrix @ second_matrix
    A = AB[:, :xdim]
    B = AB[:, xdim:]
    return A, B


@jax.jit
def _update_C(
        Y_hat: Array,
        M1T: Array
) -> Array:
    '''
    Update emissions matrix given posterior statistics
    '''
    return Y_hat.T @ inv(M1T)


@partial(jax.jit, static_argnames=('xdim',))
def _update_C_D(
        Y_hat: Array,
        Uy: Array,
        M1T: Array,
        U_hat_1T: Array,
        U1T: Array,
        xdim: int
) -> tuple[Array]:
    '''
    Update emissions and feedthrough matrices given posterior statistics
    '''
    first_matrix = jnp.block([Y_hat.T, Uy.T])
    second_matrix = inv(jnp.block(
        [[M1T, U_hat_1T.T],
         [U_hat_1T, U1T]]
    ))

    CD = first_matrix @ second_matrix
    C = CD[:, :xdim]
    D = CD[:, xdim:]
    return C, D


@jax.jit
def _update_mu0_and_Q0(
        m: Array,
        M11: Array,
        n_trials: int
) -> tuple[Array]:
    '''
    Update initial conditions and covariance given posterior statistics
    '''
    mu0 = m[:, 0, :, 0].mean(0, keepdims=True).T
    Q0 = M11 / n_trials - mu0 @ mu0.T
    return mu0, Q0


@jax.jit
def _update_mu0_and_Q0_with_inputs(
        m: Array,
        inputs: Array,
        B: Array,
        M11: Array,
        U11: Array,
        U_hat_11: Array,
        n_trials: int
) -> tuple[Array]:
    '''
    Update initial conditions and covariance given posterior statistics
    '''
    mu0 = m[:, 0, :, 0].mean(0, keepdims=True).T
    u1_mean = inputs[:, 0, :, 0].mean(0, keepdims=True).T

    mu0 = mu0 - B @ u1_mean
    Q0 = M11 + B @ U11 @ B.T - B @ U_hat_11 - U_hat_11.T @ B.T
    Q0 = Q0 / n_trials - mu0 @ mu0.T
    return mu0, Q0


@jax.jit
def _update_Q(
        M2T: Array,
        A: Array,
        M1Tm1: Array,
        M_delta: Array,
        n_trials: int,
        T: int
) -> Array:
    '''
    Update dynamics covariance given posterior statistics
    '''
    Q = M2T + A @ M1Tm1 @ A.T - A @ M_delta - M_delta.T @ A.T
    Q = Q / (n_trials * (T - 1))
    return Q


@jax.jit
def _update_Q_with_inputs(
        M2T: Array,
        A: Array,
        M1Tm1: Array,
        M_delta: Array,
        B: Array,
        U2T: Array,
        U_hat_2T: Array,
        U_delta: Array,
        n_trials: int,
        T: int
) -> Array:
    '''
    Update dynamics covariance given posterior statistics
    '''
    Q = M2T + A @ M1Tm1 @ A.T - A @ M_delta - M_delta.T @ A.T
    Q = Q + (B @ U2T @ B.T - \
             B @ U_hat_2T - \
             U_hat_2T.T @ B.T + \
             B @ U_delta @ A.T + \
             A @ U_delta.T @ B.T)
    Q = Q / (n_trials * (T - 1))
    return Q


@jax.jit
def _update_R(
        Y: Array,
        C: Array,
        M1T: Array,
        Y_hat: Array,
        n_trials: int,
        T: int
) -> Array:
    '''
    Update observations covariance given posterior statistics
    '''
    R = Y + C @ M1T @ C.T - C @ Y_hat - Y_hat.T @ C.T
    R = R / (T * n_trials)
    return R


@jax.jit
def _update_R_with_inputs(
        Y: Array,
        C: Array,
        M1T: Array,
        Y_hat: Array,
        D: Array,
        U1T: Array,
        Uy: Array,
        U_hat_1T: Array,
        n_trials: int,
        T: int
) -> Array:
    '''
    Update observation covariance given posterior statistics
    '''
    R = Y + C @ M1T @ C.T - C @ Y_hat - Y_hat.T @ C.T
    R = R + (D @ U1T @ D.T - \
             D @ Uy - \
             Uy.T @ D.T + \
             D @ U_hat_1T @ C.T + \
             C @ U_hat_1T.T @ D.T)
    R = R / (T * n_trials)
    return R


class LinearDynamicalSystemJAX:
    def __init__(
            self,
            xdim: int,
            ydim: int,
            udim: int | None = None,
            feedthrough: bool = True,
            key: jax.Array | None = None
    ):
        self.xdim = xdim
        self.ydim = ydim
        self.udim = udim if udim is not None else 0
        self.feedthrough = feedthrough
        self.key = key if key is not None else random.PRNGKey(0)
        self.init_params()

    def init_params(self):
        keys = jax.random.split(self.key, 4)

        # Initial state params
        self.mu0 = jax.random.normal(keys[0], (self.xdim, 1))
        self.Q0 = jnp.eye(self.xdim)

        # Latent params
        self.Q = jnp.eye(self.xdim)
        self.A = jnp.eye(self.xdim)

        # Observation params
        self.R = jnp.eye(self.ydim)
        self.C = jax.random.normal(keys[1], (self.ydim, self.xdim))

        # Inputs params
        if self.udim > 0:
            self.B = jax.random.normal(keys[2], (self.xdim, self.udim))
        else:
            self.B = jnp.zeros((self.xdim, self.udim))
        if self.udim > 0 and self.feedthrough:
            self.D = jax.random.normal(keys[3], (self.ydim, self.udim))
        else:
            self.D = jnp.zeros((self.ydim, self.udim))

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
            self,
            observations: Array,
            inputs: Array | None = None,
            verbose: bool = False,
            max_iter: int | None = None,
            criterion: float = 1e-8,
    ):
        """EM loop with LL-monotonicity check.

        observations: (n_trials, T, ydim, 1)
        inputs:       (n_trials, T, udim, 1) or None
        """
        self.observations = observations
        self.n_trials, self.T, _, _ = observations.shape
        self.inputs = inputs if inputs is not None else jnp.zeros((self.n_trials, self.T, self.udim, 1))
        self.ll_history = []
        LL_old = -jnp.inf
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

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------
    def e_step(self):
        self.run_filter()
        self.run_smoother()
        self._get_sufficient_stats()

    def run_filter(self):
        (self.P_predicted, self.P_filtered,
         self.x_predicted, self.x_filtered, self.LL) = _run_filter(
            self.A, self.B, self.C, self.D, self.Q, self.R,
            self.mu0, self.Q0, self.observations, self.inputs,
            self.n_trials, self.T, self.xdim, self.ydim,
        )

    def run_smoother(self):
        self.m, self.sigma, self.sigma_x = _run_smoother(
            self.P_predicted, self.P_filtered,
            self.x_predicted, self.x_filtered,
            self.A, self.n_trials, self.T, self.xdim,
        )

    def _get_sufficient_stats(self):
        if self.udim > 0:
            (self.M11, self.M2T, self.M1Tm1, self.M1T, self.M_delta, self.Y, self.Y_hat,
             self.U1T, self.U2T, self.U_hat_2T, self.Uy,
             self.U_delta, self.U_hat_1T, self.U11, self.U_hat_11) = _get_posterior_stats_w_inputs(
                self.m, self.sigma, self.sigma_x, self.observations, self.inputs, self.xdim,
            )
        else:
            (self.M11, self.M2T, self.M1Tm1, self.M1T,
             self.M_delta, self.Y, self.Y_hat) = _get_posterior_stats(
                self.m, self.sigma, self.sigma_x, self.observations, self.xdim,
            )

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------
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
        self.update_mu_and_Q0()
        self.update_Q()
        self.update_R()

    def update_A(self):
        self.A = _update_A(self.M_delta, self.M1Tm1)

    def update_A_B(self):
        self.A, self.B = _update_A_B(
            self.M_delta, self.M1Tm1, self.U_hat_2T, self.U_delta, self.U2T, self.xdim,
        )

    def update_C(self):
        self.C = _update_C(self.Y_hat, self.M1T)

    def update_C_D(self):
        self.C, self.D = _update_C_D(
            self.Y_hat, self.Uy, self.M1T, self.U_hat_1T, self.U1T, self.xdim,
        )

    def update_mu_and_Q0(self):
        if self.udim > 0:
            self.mu0, self.Q0 = _update_mu0_and_Q0_with_inputs(
                self.m, self.inputs, self.B,
                self.M11, self.U11, self.U_hat_11, self.n_trials,
            )
        else:
            self.mu0, self.Q0 = _update_mu0_and_Q0(self.m, self.M11, self.n_trials)

    def update_Q(self):
        if self.udim > 0:
            self.Q = _update_Q_with_inputs(
                self.M2T, self.A, self.M1Tm1, self.M_delta,
                self.B, self.U2T, self.U_hat_2T, self.U_delta,
                self.n_trials, self.T,
            )
        else:
            self.Q = _update_Q(
                self.M2T, self.A, self.M1Tm1, self.M_delta, self.n_trials, self.T,
            )

    def update_R(self):
        if self.udim > 0 and self.feedthrough:
            self.R = _update_R_with_inputs(
                self.Y, self.C, self.M1T, self.Y_hat,
                self.D, self.U1T, self.Uy, self.U_hat_1T,
                self.n_trials, self.T,
            )
        else:
            self.R = _update_R(
                self.Y, self.C, self.M1T, self.Y_hat, self.n_trials, self.T,
            )

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def predict(self, Y: Array, inputs: Array | None = None):
        """Run filter at current params on a held-out batch without mutating model state.

        Returns: (P_predicted, P_filtered, x_predicted, x_filtered, LL).
        LL is the per-(trial, timestep) average — same convention as run_filter.
        """
        trials, timesteps, _, _ = Y.shape
        U = inputs if inputs is not None else jnp.zeros((trials, timesteps, self.udim, 1))
        return _run_filter(
            self.A, self.B, self.C, self.D, self.Q, self.R,
            self.mu0, self.Q0, Y, U,
            trials, timesteps, self.xdim, self.ydim,
        )

    def sample(self, T: int, n_trials: int = 1, inputs: Array | None = None,
               key: Array | None = None):
        """Generate (x, y) samples from the model.

        Splits the key into one (process, obs) pair per timestep.
        Returns x: (n_trials, T, xdim, 1), y: (n_trials, T, ydim, 1).
        """
        key = key if key is not None else self.key
        U = inputs if inputs is not None else jnp.zeros((n_trials, T, self.udim, 1))

        chol_Q0 = jnp.linalg.cholesky(self.Q0)
        chol_Q  = jnp.linalg.cholesky(self.Q)
        chol_R  = jnp.linalg.cholesky(self.R)

        # One (process, obs) key pair per timestep.
        keys = random.split(key, 2 * T).reshape(T, 2, 2)

        def _step(carry, inputs):
            x_prev, t = carry
            ut, key_pair = inputs
            k_proc, k_obs = key_pair[0], key_pair[1]

            is_first = (t == 0)
            chol_for_x = jnp.where(is_first, chol_Q0, chol_Q)
            mean_x = jnp.where(
                is_first,
                self.mu0 + self.B @ ut,
                self.A @ x_prev + self.B @ ut,
            )

            x_t = mean_x + chol_for_x @ random.normal(k_proc, (n_trials, self.xdim, 1))
            y_t = self.C @ x_t + self.D @ ut + chol_R @ random.normal(k_obs, (n_trials, self.ydim, 1))
            return (x_t, t + 1), (x_t, y_t)

        _, (x, y) = jax.lax.scan(
            _step,
            init=(jnp.zeros((n_trials, self.xdim, 1)), jnp.array(0)),
            xs=(U.swapaxes(0, 1), keys),
        )
        return x.swapaxes(0, 1), y.swapaxes(0, 1)