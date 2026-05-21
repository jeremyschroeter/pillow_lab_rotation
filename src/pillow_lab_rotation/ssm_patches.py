"""Monkey-patches for the vendored Jha-lab ssm CTDS so its EM matches local CTDS.

Three fixes:

1. ``_solve_constrained_A`` uses ``cp.trace(A @ Var @ B)`` which CVXPY 1.8.1
   silently evaluates as the Frobenius inner product ``sum((A @ Var) * B)``
   instead of the matrix trace. Routing through ``cp.sum(cp.diag(...))``
   evaluates correctly.

2. The dynamics M-step uses MAP shrinkage on Q (``(sqerr + Psi0) / (nu0 + N + D + 1)``)
   and a tiny L2 ridge on A. The patched version uses MLE (``sqerr / Ens``) and
   no ridge.

3. ``Sigmas_init`` is a ``@property`` that returns a fresh array on every access;
   the original M-step assigns ``self.Sigmas_init[k] = ...`` which mutates the
   throwaway. The patched M-step writes through ``self._sqrt_Sigmas_init[k]``
   directly, averaging ``E[x_1 x_1^T]`` over trials. ``mu_init`` is held at zero.

The emissions M-step also gets a MLE variant (``Psi0=0, nu0=-N-1``, no ridge) so
``[C, d, R]`` are estimated without the MAP prior.

Usage:

>>> import ssm.lds as ssm_lds
>>> m = ssm_lds.LDS(..., dynamics='gaussian_ctds', emissions='gaussian_ctds', ...)
>>> patch_ssm_ctds(m)              # mutates in place
>>> m.fit(datas, method='laplace_em', ...)
"""
import types
import numpy as np
import cvxpy as cp

from ssm.regression import fit_constrained_linear_regression


def _fixed_solve_constrained_A(self, k, ExuxuTs_k, ExuyTs_k, Sigmas_k):
    """Bug-free version of the constrained A QP — CVXPY trace-bug workaround."""
    D, M, lags = self.D, self.M, self.lags
    Q_inv = np.linalg.inv(Sigmas_k)
    Q_inv = Q_inv / np.max(np.abs(Q_inv))
    L = np.linalg.cholesky(Q_inv)
    kron_ExuxuTs = np.kron((ExuxuTs_k + self.J0_k).T, np.eye(D))
    W = cp.Variable((D, D * lags + M))
    constraints = self.within_region_constraints(W) + self.across_region_constraints(W)
    objective = cp.Minimize(
        cp.quad_form((L.T @ W).flatten(order='F'), cp.psd_wrap(kron_ExuxuTs))
        - 2 * cp.sum(cp.diag(Q_inv @ W @ (ExuyTs_k + self.h0_k)))
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False, warm_start=True)
    if prob.status != 'optimal':
        print('Warning: patched M step for A failed to converge!')
    return W.value


def _patched_dynamics_m_step(self, expectations, datas, inputs, masks, tags,
                              continuous_expectations=None, **kwargs):
    """MLE dynamics M-step: no MAP priors, no L2 ridge, Q0 averaged over trials."""
    K, D, M, lags = self.K, self.D, self.M, self.lags
    assert continuous_expectations is not None, "Patched path requires laplace_em with exact stats"

    ExuxuTs = np.zeros((K, D * lags + M + 1, D * lags + M + 1))
    ExuyTs = np.zeros((K, D * lags + M + 1, D))
    EyyTs = np.zeros((K, D, D))
    Ens = np.zeros(K)
    M11 = np.zeros((K, D, D))
    n_trials_counted = 0

    for (Ez, _, _), (_, Ex, smoothed_sigmas, Exxn), u in \
            zip(expectations, continuous_expectations, inputs):
        ExxT = smoothed_sigmas + np.einsum('ti,tj->tij', Ex, Ex)
        u_aft = u[lags:]
        for k in range(K):
            w = Ez[lags:, k]
            ExuxuTs[k, :D, :D] += np.einsum('t,tij->ij', w, ExxT[:-1])
            ExuxuTs[k, :D, D:D + M] += np.einsum('t,ti,tj->ij', w, Ex[:-1], u_aft)
            ExuxuTs[k, :D, -1] += np.einsum('t,ti->i', w, Ex[:-1])
            ExuxuTs[k, D:D + M, D:D + M] += np.einsum('t,ti,tj->ij', w, u_aft, u_aft)
            ExuxuTs[k, D:D + M, -1] += np.einsum('t,ti->i', w, u_aft)
            ExuxuTs[k, -1, -1] += np.sum(w)
            ExuyTs[k, :D, :] += np.einsum('t,tij->ij', w, Exxn)
            ExuyTs[k, D:D + M, :] += np.einsum('t,ti,tj->ij', w, u_aft, Ex[1:])
            ExuyTs[k, -1, :] += np.einsum('t,ti->i', w, Ex[1:])
            EyyTs[k] += np.einsum('t,tij->ij', w, ExxT[1:])
            Ens[k] += np.sum(w)
            M11[k] += ExxT[0]
        n_trials_counted += 1

    for k in range(K):
        ExuxuTs[k, D:D + M, :D] = ExuxuTs[k, :D, D:D + M].T
        ExuxuTs[k, -1, :D] = ExuxuTs[k, :D, -1].T
        ExuxuTs[k, -1, D:D + M] = ExuxuTs[k, D:D + M, -1].T

    # Q0 averaged over trials, mu_init held at zero. Write through _sqrt_Sigmas_init
    # because the Sigmas_init property returns a throwaway copy.
    for k in range(K):
        self.mu_init[k] = np.zeros(D)
        Q0_new = M11[k] / n_trials_counted
        self._sqrt_Sigmas_init[k] = np.linalg.cholesky(Q0_new + 1e-8 * np.eye(D))

    As = np.zeros((K, D, D * lags))
    Vs = np.zeros((K, D, M))
    bs = np.zeros((K, D))
    Sigmas = np.zeros((K, D, D))
    for k in range(K):
        ExuxuTs_k = ExuxuTs[k][:D * lags + M, :D * lags + M]
        ExuyTs_k = ExuyTs[k][:D * lags + M]
        self.J0_k = np.zeros_like(self.J0[k][:D * lags + M, :D * lags + M])
        self.h0_k = np.zeros_like(self.h0[k][:D * lags + M])
        Wk = self._solve_constrained_A(k, ExuxuTs_k, ExuyTs_k, self.Sigmas[k])
        As[k] = Wk[:, :D * lags]
        Vs[k] = Wk[:, D * lags:D * lags + M]
        bs[k] = np.zeros(D)
        EWxyT = Wk @ ExuyTs_k
        sqerr = EyyTs[k] - EWxyT.T - EWxyT + Wk @ ExuxuTs_k @ Wk.T
        Sigmas[k] = sqerr / Ens[k]

    self.As = As
    self.Vs = Vs
    self.bs = bs
    self._sqrt_Sigmas[0] = np.linalg.cholesky(Sigmas[0])
    self.Sigmas = Sigmas


def _patched_emissions_m_step(self, discrete_expectations, continuous_expectations,
                               datas, inputs, masks, tags, **kwargs):
    """MLE emissions M-step: Psi0=0, nu0=-N-1, no ridge."""
    ys = datas
    Xs = [np.column_stack([x]) for (_, x, _, _), u in
            zip(continuous_expectations, inputs)]
    region_identity, cell_identity = self.region_identity, self.cell_identity
    ExxT, ExyT, EyyT, weight_sum = self._compute_statistics_for_m_step(
        ys, inputs, masks, tags, continuous_expectations, discrete_expectations,
    )
    list_of_dims = kwargs.get('list_of_dimensions')
    assert self.single_subspace and all(np.all(mask) for mask in masks)

    initial_C = np.hstack([self.Cs[0], self.ds[0][:, None]])
    expectations = [ExxT[0], ExyT[0], EyyT[0], weight_sum]
    CF, d, Sigma = fit_constrained_linear_regression(
        Xs, ys, expectations=expectations,
        Psi0=0,
        nu0=-self.N - 1,
        prior_ExxT=np.zeros((self.D + 1, self.D + 1)),
        prior_ExyT=np.zeros((self.D + 1, self.N)),
        list_of_dims=list_of_dims,
        region_identity=region_identity, cell_identity=cell_identity,
        initial_C=initial_C,
    )
    self.Cs = CF[None, :, :self.D]
    self.inv_etas = Sigma[None, :]
    self.Fs = np.zeros((1, self.N, self.M))
    self.ds = d[None, :]


def patch_ssm_ctds(model):
    """Bind the MLE M-steps to a freshly constructed ssm CTDS LDS instance.

    The model must have been built with ``dynamics='gaussian_ctds'`` and
    ``emissions='gaussian_ctds'``. After patching, ``model.fit(method='laplace_em',
    ...)`` produces an MLE fit that matches Jeremy's local ``CTDS`` to
    floating-point precision (with ``fit_mu0=False, fit_b=False, fit_d_bias=True``
    on the local side).
    """
    model.dynamics._solve_constrained_A = types.MethodType(
        _fixed_solve_constrained_A, model.dynamics)
    model.dynamics.m_step = types.MethodType(_patched_dynamics_m_step, model.dynamics)
    model.emissions.m_step = types.MethodType(_patched_emissions_m_step, model.emissions)
    return model
