'''
@ Jeremy Schroeter, Sep 2025

This file contains a bunch of utility functions for my rotation project in
the Brody Lab. Some of these functions are from Carlos Brody, but I have
rewritten them for didactic purposes, and also to make them more readable
(for me).

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import ast
import operator
import re
import functools



@functools.lru_cache(maxsize=8)
def gaussian_kernel(sigma: float) -> tuple[np.ndarray, int]:
    c = int(np.ceil(4 * sigma))
    x = np.arange(-c, c + 1)
    f = np.exp(-x ** 2 / (2 * sigma ** 2))
    return tuple(f / f.sum()), c  # tuple for hashability


def min_max_time(
        spike_times: np.ndarray
) -> tuple[float, float]:
    '''
    
    Returns the min and max spike time in the dataset. This is equivalent
    to concatenating all the spike times and calling .min()/.max() but it
    is faster!

    '''

    min_time = np.inf
    max_time = -np.inf
    for cell in spike_times:
        if len(cell) > 0:
            min_time = min(min_time, cell[0])
            max_time = max(max_time, cell[-1])
    
    return min_time, max_time


def get_session_time_points(
        raw_spike_times: np.ndarray,
        dt: float
) -> np.ndarray:

    min_time, max_time = min_max_time(raw_spike_times)
    return np.arange(np.floor(min_time), np.ceil(max_time), dt)



def bin_spikes(
        event_times: np.ndarray,
        t: np.ndarray,
        dt: float
) -> np.ndarray:
    '''
    
    Given a list of event (spike) times, bin them into bins of size dt, and return
    a list of the number of events in each bin. The first bin is centered at
    t[0] + dt / 2, and the last bin is centered at t[-1] - dt / 2
    
    '''
    start_time = t[0] - dt / 2
    stop_time = t[-1] + 3 * dt / 4
    bins = np.arange(start_time, stop_time, dt)

    return np.histogram(event_times, bins)[0]


def smooth(
        signal: np.ndarray,
        sigma: float
) -> np.ndarray:
    '''
    
    Smooth a 1d signal with a gaussian kernel, controlling for edge effects.
    
    '''
    
    kernel, center_bin = gaussian_kernel(sigma)
    kernel = np.array(kernel)  # Convert back from tuple (needed for lru_cache)
    smoothed_signal = np.convolve(signal, kernel, mode='full')[center_bin : -center_bin]
    
    edge_compensation = np.ceil(sigma * 2.5).astype(int)
    for i in range(edge_compensation):
        smoothed_signal[i] = kernel[center_bin - i:][::-1] @ signal[:i + center_bin + 1]
        smoothed_signal[i] /= kernel[center_bin - i:][::-1].sum()

        smoothed_signal[-i - 1] = kernel[:center_bin + i + 1] @ signal[-i - center_bin - 1:]
        smoothed_signal[-i - 1] /= kernel[:center_bin + i + 1].sum()
    
    return smoothed_signal


def smooth_all_spike_trains(
        spike_times: np.ndarray,
        dt: float,
        sigma: float,
        session_timepoints: np.ndarray,
        down_sample_factor: int
) -> np.ndarray:
    '''
    
    Takes an array of arrays containing spike times and first bins them
    and then smooths them with a gaussian filter to get firing rates before
    downsampling them.
    
    '''
    
    # Initialize array as #neurons x #timepoints
    N = len(spike_times)
    T = len(session_timepoints[0 : -1 : down_sample_factor])
    firing_rates = np.zeros((N, T))

    # smooooooooth
    for n in tqdm(range(N), total=N):
        binned_spikes = bin_spikes(spike_times[n], session_timepoints, dt)
        smoothed = smooth(binned_spikes, sigma / dt)
        down_sampled = smoothed[:-1:down_sample_factor] / dt
        firing_rates[n] = down_sampled
    
    return firing_rates


def organize_by_trials(
        trial_info: pd.DataFrame,
        firing_rates: np.ndarray,
        session_timepoints: np.ndarray,
        dt: float,
        down_sample_factor: int,
        start: str,
        end: str
) -> np.ndarray:

    n_trials = len(trial_info)
    n_cells = firing_rates.shape[0]
    bin_size = dt * down_sample_factor
    timepoint_arr = session_timepoints[:-1:down_sample_factor]

    # Parse expressions ONCE (not in loop)
    start_times = parseIt(start, trial_info)
    end_times = parseIt(end, trial_info)

    # Pre-compute max duration to avoid dynamic array growth
    durations = end_times - start_times
    max_duration = np.nanpercentile(durations, 99)
    max_bins = int(np.ceil(max_duration / bin_size)) + 5  # small buffer

    # Allocate full array upfront
    trial_rates = np.full((n_trials, n_cells, max_bins), np.nan)
    trial_T = np.arange(max_bins) * bin_size

    for trial in range(n_trials):
        start_idx = start_times[trial]
        end_idx = end_times[trial]

        # Get firing rates for the trial
        mask = (timepoint_arr > start_idx) & (timepoint_arr < end_idx)
        trial_frs = firing_rates[:, mask]

        # Copy into pre-allocated array (truncate if somehow longer)
        n_bins = min(trial_frs.shape[1], max_bins)
        trial_rates[trial, :, :n_bins] = trial_frs[:, :n_bins]

    return trial_rates, trial_T


def concat_non_nan_trials(
        trial_rates: np.ndarray
) -> np.ndarray:
    n_trials, n_cells, n_timebins = trial_rates.shape

    bins = np.zeros(n_trials, dtype=int)

    for trial in range(n_trials):
        trial_vector = trial_rates[trial, 1, :]

        # Find how many non-nan bins this trial has
        if np.any(np.isnan(trial_vector)):
            idx = np.where(np.isnan(trial_vector))[0][0]
            bins[trial] = idx
        else:
            idx = None
            bins[trial] = trial_rates.shape[-1]

    concatenated = np.zeros((n_cells, bins.sum()))

    for trial in range(n_trials):
        start = np.sum(bins[:trial])
        stop = np.sum(bins[:trial + 1])
        concatenated[:, start : stop] = trial_rates[trial, :, :bins[trial]]

    return concatenated
    



def parseIt(expr, dataframe):
    """
    parseIt(expr, dataframe) --> value

    Parse an expression and evaluate it in the context of a dataframe. For example,
    parseIt('stereo_click+clicks_on-cpoke_in-0.5', dataframe) will return the sum of 
    columns stereo_click and clicks_on, minus column cpoke_in, in the dataframe, minus 0.5.
    In the context of a PBupsTrials dataframe, this would return the time of 500 ms before
    the stereo click, relative to cpoke_in. 
    Addition, subtraction, multiplication, division, exponentiation, modulo operations, and
    negation are supported. The expression must only contain column names, constants and the 
    operators mentioned above. The function is safe to use with untrusted input.

    Parameters
    ----------
    expr : str
        The expression to parse and evaluate.
    dataframe : pandas.DataFrame
        The dataframe in which to evaluate the expression.

    Returns
    -------
    value : float or np.ndarray
        The result of evaluating the expression in the context of the dataframe.
    
    """
    # Define supported operators for safety
    allowed_operators = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg
    }

    def evaluate_expression(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Num):  # For Python 3.8 and earlier, use ast.Num
            return node.n
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](evaluate_expression(node.left), evaluate_expression(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](evaluate_expression(node.operand))
        elif isinstance(node, ast.Name):
            return dataframe[node.id].values
        else:
            raise TypeError("Unsupported operation")

    # Parse the expression into an AST
    tree = ast.parse(expr, mode='eval')
    
    # Ensure the AST only contains safe operations
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                 ast.Num, ast.operator, ast.Load, ast.Name)):
            raise ValueError("Unsupported expression")
    
    # Evaluate the expression
    return evaluate_expression(tree.body)



def get_region_mask(
        brain_regions: list[str],
        dfCells: pd.DataFrame
) -> np.ndarray:
    
    regex = "|".join(re.escape(r) for r in brain_regions)
    myCells = dfCells['brain_region'].str.contains(regex).values
    return myCells


def get_unique_brain_regions(
        cell_info: pd.DataFrame
) -> list:
    regions = cell_info['brain_region'].unique()
    suffixes = {r.split('_')[-1] for r in regions}
    return sorted(suffixes)


def get_colors(
        cell_info: pd.DataFrame,
        selection_info: np.ndarray
) -> list:
    regions = get_unique_brain_regions(cell_info)
    region_to_color = {
        region : plt.colormaps['tab20'](i)
        for i, region in enumerate(regions)
    }
    return [region_to_color[r.split('_')[-1]]
            for r in selection_info['brain_region']]


def left_right_mask(
        cells: pd.DataFrame
) -> tuple[np.ndarray]:
    left_mask = cells['brain_region'].str.contains('left')
    right_mask = cells['brain_region'].str.contains('right')
    return left_mask, right_mask


def _raised_cos_func(
        x: np.ndarray,
        center: np.ndarray,
        d_center: float
) -> np.ndarray:
    """Single raised cosine basis function."""
    return 0.5 * (np.cos(np.maximum(-np.pi, np.minimum(np.pi, (x - center) * np.pi / d_center / 2))) + 1)


def make_raised_cos_basis(
        n_basis: int,
        peak_range: tuple[float, float],
        dt: float = 1.0,
        log_scaling: str = 'linear',
        log_offset: float = 1.0,
        time_range: tuple[float, float] = None,
        initial_ones_flag: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Make a basis of raised cosines with a logarithmic or linear time axis.

    Based on makeRaisedCosBasis.m by J. Pillow.

    Parameters
    ----------
    n_basis : int
        Number of basis vectors.
    peak_range : tuple[float, float]
        Position of 1st and last cosine peaks.
    dt : float
        Time bin size of bins representing basis (default=1).
    log_scaling : str
        'log' or 'linear' scaling of time axis (default='linear').
    log_offset : float
        Offset for nonlinear stretching of t axis: y = log(t + log_offset).
        Larger values give more nearly linear stretching (default=1).
    time_range : tuple[float, float] None
        Time range of basis. If None, computed automatically.
    initial_ones_flag : bool
        If True, sets first basis vector to all ones prior to the 1st peak.
        Useful for refractory effects in GLM (default=False).

    Returns
    -------
    cos_basis : np.ndarray
        Cosine basis vectors [n_timepoints x n_basis].
    t_grid : np.ndarray
        Time lattice on which basis is defined [n_timepoints].
    basis_peaks : np.ndarray
        Centers of each cosine basis function [n_basis].
    '''

    if log_scaling not in ('log', 'linear'):
        raise ValueError("log_scaling must be 'log' or 'linear'")

    if log_scaling == 'log':
        # Log scaling of x axis
        if log_offset <= 0:
            raise ValueError('log_offset must be > 0')

        # Nonlinear time axis stretching function and its inverse
        nlin = lambda x: np.log(x + 1e-20)
        invnl = lambda x: np.exp(x) - 1e-20

        # Compute location for cosine basis centers in stretched coordinates
        log_peak_range = nlin(np.array(peak_range) + log_offset)
        d_center = (log_peak_range[1] - log_peak_range[0]) / (n_basis - 1)
        b_centers = np.linspace(log_peak_range[0], log_peak_range[1], n_basis)
        basis_peaks = invnl(b_centers) - log_offset  # peaks in original time

        # Compute time grid points
        if time_range is None:
            min_t = 0
            max_t = invnl(log_peak_range[1] + 2 * d_center) - log_offset
        else:
            min_t, max_t = time_range

        t_grid = np.arange(min_t, max_t + dt, dt)

        # Make the basis
        t_stretched = nlin(t_grid + log_offset)
        cos_basis = _raised_cos_func(
            t_stretched[:, np.newaxis],
            b_centers[np.newaxis, :],
            d_center
        )

    else:
        # Linear scaling of x axis
        d_center = (peak_range[1] - peak_range[0]) / (n_basis - 1)
        b_centers = np.linspace(peak_range[0], peak_range[1], n_basis)
        basis_peaks = b_centers.copy()

        # Compute time grid points
        if time_range is None:
            min_t = peak_range[0] - 2 * d_center
            max_t = peak_range[1] + 2 * d_center
        else:
            min_t, max_t = time_range

        t_grid = np.arange(min_t, max_t + dt, dt)

        # Make the basis
        cos_basis = _raised_cos_func(
            t_grid[:, np.newaxis],
            b_centers[np.newaxis, :],
            d_center
        )

    # If necessary, set first basis vector to 1 before first peak
    if initial_ones_flag:
        ii = t_grid <= peak_range[0]
        cos_basis[ii, 0] = 1.0

    # Check condition number
    cond_num = np.linalg.cond(cos_basis)
    if cond_num > 1e12:
        import warnings
        warnings.warn(f'Raised cosine basis is poorly conditioned (cond # = {cond_num:.2f})')

    return cos_basis, t_grid, basis_peaks


def fit_basis_weights(
        signal: np.ndarray,
        basis: np.ndarray
) -> np.ndarray:
    """
    Fit weights for basis functions using least squares.

    Parameters
    ----------
    signal : np.ndarray
        Signal to fit [n_timepoints] or [n_timepoints x n_signals].
    basis : np.ndarray
        Basis matrix [n_timepoints x n_basis].

    Returns
    -------
    weights : np.ndarray
        Fitted weights [n_basis] or [n_basis x n_signals].
    """
    # Solve (B'B) w = B' signal using lstsq for numerical stability
    return np.linalg.lstsq(basis, signal, rcond=None)[0]


def smooth_with_basis(
        signal: np.ndarray,
        basis: np.ndarray
) -> np.ndarray:
    """
    Smooth a signal by projecting onto a raised cosine basis and reconstructing.

    Parameters
    ----------
    signal : np.ndarray
        Signal to smooth [n_timepoints] or [n_timepoints x n_signals].
    basis : np.ndarray
        Basis matrix [n_timepoints x n_basis].

    Returns
    -------
    smoothed : np.ndarray
        Smoothed signal, same shape as input.
    """
    weights = fit_basis_weights(signal, basis)
    return basis @ weights


def smooth_all_spike_trains_basis(
        spike_times: np.ndarray,
        dt: float,
        session_timepoints: np.ndarray,
        down_sample_factor: int,
        n_basis: int,
        peak_range: tuple[float, float] = None,
        log_scaling: str = 'linear',
        log_offset: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Smooth spike trains using raised cosine basis instead of Gaussian kernel.

    Parameters
    ----------
    spike_times : np.ndarray
        Array of arrays containing spike times for each neuron.
    dt : float
        Time bin size in seconds.
    session_timepoints : np.ndarray
        Time points for the session.
    down_sample_factor : int
        Factor to downsample by after smoothing.
    n_basis : int
        Number of raised cosine basis functions.
    peak_range : tuple[float, float] None
        Range for basis function peaks in bins. If None, spans the data.
    log_scaling : str
        'log' or 'linear' scaling of time axis.
    log_offset : float
        Offset for log scaling (larger = more linear).

    Returns
    -------
    firing_rates : np.ndarray
        Smoothed firing rates [n_neurons x n_timepoints_downsampled].
    basis : np.ndarray
        The basis matrix used (useful for inspection).
    '''
    N = len(spike_times)
    n_bins_full = len(session_timepoints) - 1
    T = len(session_timepoints[0:-1:down_sample_factor])

    # Create basis for full resolution data
    if peak_range is None:
        # Default: span most of the data with some padding
        peak_range = (n_bins_full * 0.05, n_bins_full * 0.95)

    basis, _, _ = make_raised_cos_basis(
        n_basis=n_basis,
        peak_range=peak_range,
        dt=1.0,  # basis in bin units
        log_scaling=log_scaling,
        log_offset=log_offset,
        time_range=(0, n_bins_full - 1)
    )

    firing_rates = np.zeros((N, T))

    for n in tqdm(range(N), total=N):
        binned_spikes = bin_spikes(spike_times[n], session_timepoints, dt)

        # Pad or truncate to match basis length
        if len(binned_spikes) > basis.shape[0]:
            binned_spikes = binned_spikes[:basis.shape[0]]
        elif len(binned_spikes) < basis.shape[0]:
            binned_spikes = np.pad(binned_spikes, (0, basis.shape[0] - len(binned_spikes)))

        # Smooth using basis projection
        smoothed = smooth_with_basis(binned_spikes, basis)

        # Truncate to original length if needed, downsample, convert to rate
        smoothed = smoothed[:n_bins_full]
        down_sampled = smoothed[::down_sample_factor] / dt
        firing_rates[n] = down_sampled[:T]

    return firing_rates, basis

