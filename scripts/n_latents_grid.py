from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from pillow_lab_rotation.jax_ctds import CTDSJax
from pillow_lab_rotation.dataio import RunyanSession

MOUSE = 'FU1-00'
DATE = '2022-03-17'
MAX_ITER = 30
INIT = 'random'
DE_RANGE = range(1, 8)
DI_RANGE = range(1, 8)
TRAIN_FRAC = 0.8
SPLIT_SEED = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / 'results' / 'n_latents_grid'
FIGURES_DIR = PROJECT_ROOT / 'figures'


def stratified_split(
        conditions: np.ndarray,
        train_frac: float = TRAIN_FRAC,
        seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split trial indices into train/test, preserving per-condition proportions."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cond in np.unique(conditions):
        idx = rng.permutation(np.where(conditions == cond)[0])
        cut = int(round(train_frac * len(idx)))
        train.append(idx[:cut])
        test.append(idx[cut:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def fit_model(obs_train_jax: Array, De: int, Di: int, Ne: int, Ni: int) -> CTDSJax:
    model = CTDSJax(
        De=De,
        Di=Di,
        Ne=Ne,
        Ni=Ni,
        key=jax.random.PRNGKey(0),
    )
    model.fit(obs_train_jax, max_iter=MAX_ITER, init=INIT)
    return model


def evaluate_on_test(model: CTDSJax, obs_test_jax: Array) -> float:
    """Run filter + smoother on test trials. Returns per trial-step test LL."""
    n_test, T = obs_test_jax.shape[0], obs_test_jax.shape[1]
    model.observations = obs_test_jax
    model.inputs = jnp.zeros((n_test, T, model.udim, 1))
    model.n_trials = n_test
    model.T = T
    model.run_filter()
    model.run_smoother()
    return float(model.LL) / (n_test * T)


def compute_r2(m: np.ndarray, C: np.ndarray, d_bias: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-neuron R² of y_hat = m C^T + d_bias against obs."""
    N = C.shape[0]
    m2 = m.squeeze(-1) if m.ndim == 4 else m
    y_obs = obs.squeeze(-1).reshape(-1, N)
    y_hat = (m2 @ C.T).reshape(-1, N) + d_bias.squeeze(-1)
    resid = y_obs - y_hat
    return 1.0 - resid.var(axis=0) / y_obs.var(axis=0)


def _pearson_per_column(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-column Pearson r between equal-shape (T, N) arrays."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    num = (Ac * Bc).sum(axis=0)
    denom = np.sqrt((Ac ** 2).sum(axis=0) * (Bc ** 2).sum(axis=0))
    return np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), np.nan)


def _condition_psth_stack(
        y: np.ndarray,                # (n_trials, T, N)
        conditions: np.ndarray,       # (n_trials,)
) -> np.ndarray:
    """Per-condition trial-mean PSTHs concatenated along time → (T*n_cond, N)."""
    psths = [y[conditions == c].mean(axis=0) for c in np.unique(conditions)]
    return np.concatenate(psths, axis=0)


def compute_psth_r(
        m_test: np.ndarray,           # (n_test, T, D_lat, 1)
        C: np.ndarray,                # (N, D_lat)
        d_bias: np.ndarray,           # (N, 1)
        obs_test: np.ndarray,         # (n_test, T, N, 1)
        conditions_test: np.ndarray,  # (n_test,)
) -> np.ndarray:
    """Per-neuron Pearson r between predicted and empirical condition PSTHs.

    PSTHs are computed per condition (e.g. SOM, PV) and concatenated along
    time, so the r captures both within-condition shape and across-condition
    differences. Returns shape (N,).
    """
    m2 = m_test.squeeze(-1)
    d2 = d_bias.squeeze(-1)
    y_obs = obs_test.squeeze(-1)
    y_hat = m2 @ C.T + d2

    pred = _condition_psth_stack(y_hat, conditions_test)
    emp = _condition_psth_stack(y_obs, conditions_test)
    return _pearson_per_column(pred, emp)


def compute_ceiling_r(
        obs_train: np.ndarray,
        conditions_train: np.ndarray,
        obs_test: np.ndarray,
        conditions_test: np.ndarray,
) -> np.ndarray:
    """Per-neuron r between train PSTH and test PSTH (model-free ceiling).

    Uses common conditions only. Returns shape (N,).
    """
    y_train = obs_train.squeeze(-1)
    y_test = obs_test.squeeze(-1)
    common = sorted(set(np.unique(conditions_train)) & set(np.unique(conditions_test)))
    tr_psths = [y_train[conditions_train == c].mean(axis=0) for c in common]
    te_psths = [y_test[conditions_test == c].mean(axis=0) for c in common]
    return _pearson_per_column(
        np.concatenate(tr_psths, axis=0),
        np.concatenate(te_psths, axis=0),
    )


def save_model(model: CTDSJax, path: Path, m_test: np.ndarray) -> None:
    np.savez(
        path,
        A=np.asarray(model.A),
        B=np.asarray(model.B),
        C=np.asarray(model.C),
        D=np.asarray(model.D),
        Q=np.asarray(model.Q),
        R=np.asarray(model.R),
        Q0=np.asarray(model.Q0),
        mu0=np.asarray(model.mu0),
        b=np.asarray(model.b),
        d_bias=np.asarray(model.d_bias),
        m_test=m_test,
        ll_history=np.asarray(model.ll_history),
    )


def plot_grids(
        LL_test: np.ndarray,
        LL_train: np.ndarray,
        R2_E: np.ndarray,
        R2_I: np.ndarray,
        PSTH_E: np.ndarray,
        PSTH_I: np.ndarray,
        ceil_E: float,
        ceil_I: float,
        De_vals: list[int],
        Di_vals: list[int],
        save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        # (ax_idx, grid, title, vmin, vmax, cmap)
        ((0, 0), LL_test,  'test LL (per trial-step)',                 None, None, 'viridis'),
        ((0, 1), R2_E,     r'mean $R^2$ — E (test)',                   0.0,  1.0,  'magma'),
        ((0, 2), PSTH_E,   rf'mean PSTH r — E (test, ceiling={ceil_E:.2f})', 0.0, 1.0, 'magma'),
        ((1, 0), LL_train, 'train LL (per trial-step, final)',         None, None, 'viridis'),
        ((1, 1), R2_I,     r'mean $R^2$ — I (test)',                   0.0,  1.0,  'magma'),
        ((1, 2), PSTH_I,   rf'mean PSTH r — I (test, ceiling={ceil_I:.2f})', 0.0, 1.0, 'magma'),
    ]
    for (r, c), grid, title, vmin, vmax, cmap in panels:
        ax = axes[r, c]
        im = ax.imshow(
            grid,
            origin='lower',
            aspect='equal',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[Di_vals[0] - 0.5, Di_vals[-1] + 0.5,
                    De_vals[0] - 0.5, De_vals[-1] + 0.5],
        )
        ax.set_xticks(Di_vals)
        ax.set_yticks(De_vals)
        ax.set_xlabel('Di')
        ax.set_ylabel('De')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_r2_histograms(
        r2_per_neuron: np.ndarray,  # (n_de, n_di, N)
        Ne: int,
        De_vals: list[int],
        Di_vals: list[int],
        save_path: Path,
) -> None:
    """7×7 grid of per-neuron R² histograms, E vs I, one cell per (De, Di)."""
    n_de, n_di = len(De_vals), len(Di_vals)
    lo = min(0.0, float(np.nanmin(r2_per_neuron)))
    bins = np.linspace(lo, 1.0, 30)

    fig, axes = plt.subplots(
        n_de, n_di,
        figsize=(2.0 * n_di, 1.6 * n_de),
        sharex=True, sharey=True,
    )
    for i, De in enumerate(De_vals):
        for j, Di in enumerate(Di_vals):
            ax = axes[n_de - 1 - i, j]
            r2 = r2_per_neuron[i, j]
            r2_E, r2_I = r2[:Ne], r2[Ne:]
            ax.hist(r2_E, bins=bins, alpha=0.55, color='C0', density=True)
            ax.hist(r2_I, bins=bins, alpha=0.55, color='C3', density=True)
            ax.axvline(np.median(r2_E), color='C0', ls='--', lw=0.8)
            ax.axvline(np.median(r2_I), color='C3', ls='--', lw=0.8)
            ax.set_title(f'De={De}, Di={Di}', fontsize=8)
            ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$R^2$', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('density', fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='C0', alpha=0.55, label='E'),
        plt.Rectangle((0, 0), 1, 1, color='C3', alpha=0.55, label='I'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = RunyanSession(MOUSE, DATE)
    obs_all = np.asarray(dataset.obs_evt)
    conditions = dataset.conditions
    Ne, Ni = dataset.Ne, dataset.Ni
    N = Ne + Ni

    train_idx, test_idx = stratified_split(conditions)
    obs_train_np = obs_all[train_idx]
    obs_test_np = obs_all[test_idx]
    obs_train_jax = jnp.asarray(obs_train_np)
    obs_test_jax = jnp.asarray(obs_test_np)
    conditions_train = conditions[train_idx]
    conditions_test = conditions[test_idx]
    print(
        f'split: {len(train_idx)} train trials, {len(test_idx)} test trials '
        f'(seed={SPLIT_SEED}, stratified by condition)',
        flush=True,
    )

    ceiling_per_neuron = compute_ceiling_r(
        obs_train_np, conditions_train, obs_test_np, conditions_test,
    )
    ceiling_E = float(np.nanmean(ceiling_per_neuron[:Ne]))
    ceiling_I = float(np.nanmean(ceiling_per_neuron[Ne:]))
    print(
        f'PSTH ceiling (train PSTH vs test PSTH): E={ceiling_E:.3f}, I={ceiling_I:.3f}',
        flush=True,
    )

    De_vals = list(DE_RANGE)
    Di_vals = list(DI_RANGE)
    n_de, n_di = len(De_vals), len(Di_vals)
    n_total = n_de * n_di

    LL_test = np.full((n_de, n_di), np.nan)
    LL_train_final = np.full((n_de, n_di), np.nan)
    R2_E = np.full((n_de, n_di), np.nan)
    R2_I = np.full((n_de, n_di), np.nan)
    PSTH_E = np.full((n_de, n_di), np.nan)
    PSTH_I = np.full((n_de, n_di), np.nan)
    r2_per_neuron = np.full((n_de, n_di, N), np.nan)
    psth_per_neuron = np.full((n_de, n_di, N), np.nan)

    for i, De in enumerate(De_vals):
        for j, Di in enumerate(Di_vals):
            k = i * n_di + j + 1
            print(f'[{k}/{n_total}] fitting De={De}, Di={Di}', flush=True)
            model = fit_model(obs_train_jax, De, Di, Ne, Ni)
            LL_train_final[i, j] = float(model.ll_history[-1])

            test_ll = evaluate_on_test(model, obs_test_jax)
            LL_test[i, j] = test_ll

            m_test = np.asarray(model.m)
            C_np = np.asarray(model.C)
            d_bias_np = np.asarray(model.d_bias)

            r2 = compute_r2(m_test, C_np, d_bias_np, obs_test_np)
            r2_per_neuron[i, j] = r2
            R2_E[i, j] = float(r2[:Ne].mean())
            R2_I[i, j] = float(r2[Ne:].mean())

            psth_r = compute_psth_r(m_test, C_np, d_bias_np, obs_test_np, conditions_test)
            psth_per_neuron[i, j] = psth_r
            PSTH_E[i, j] = float(np.nanmean(psth_r[:Ne]))
            PSTH_I[i, j] = float(np.nanmean(psth_r[Ne:]))

            save_model(model, RESULTS_DIR / f'De{De}_Di{Di}.npz', m_test)
            print(
                f'    test_LL/step={test_ll:.4f}  '
                f'R2_E={R2_E[i, j]:.3f}  R2_I={R2_I[i, j]:.3f}  '
                f'PSTH_r_E={PSTH_E[i, j]:.3f}  PSTH_r_I={PSTH_I[i, j]:.3f}',
                flush=True,
            )

    np.savez(
        RESULTS_DIR / 'grid_summary.npz',
        De=np.array(De_vals),
        Di=np.array(Di_vals),
        LL_test=LL_test,
        LL_train_final=LL_train_final,
        R2_E=R2_E,
        R2_I=R2_I,
        PSTH_E=PSTH_E,
        PSTH_I=PSTH_I,
        r2_per_neuron=r2_per_neuron,
        psth_per_neuron=psth_per_neuron,
        ceiling_per_neuron=ceiling_per_neuron,
        ceiling_E=ceiling_E,
        ceiling_I=ceiling_I,
        train_idx=train_idx,
        test_idx=test_idx,
        Ne=Ne,
        Ni=Ni,
    )

    grid_fig = FIGURES_DIR / 'n_latents_grid.png'
    hist_fig = FIGURES_DIR / 'n_latents_r2_histograms.png'
    plot_grids(
        LL_test, LL_train_final, R2_E, R2_I, PSTH_E, PSTH_I,
        ceiling_E, ceiling_I, De_vals, Di_vals, grid_fig,
    )
    plot_r2_histograms(r2_per_neuron, Ne, De_vals, Di_vals, hist_fig)
    print(f'done. figures: {grid_fig}, {hist_fig}')
