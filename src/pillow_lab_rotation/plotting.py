import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

def plot_confidence_ellipse(M, V):
    """
    Plot a one standard-deviation ellipse for a N-dimensional Gaussian distribution.
    Arguments:
        M: mean of the Gaussian (N x 1) 
        V: Covariance matrix of the Gaussian (N x N)

    We've provided this function for you. Feel free to use it as is to help plot 
    the one standard-deviation confidence ellipse for each position estimate, modify it
    to your liking, or write your own function.

    Code adapted from EM_GM.m by Patrick P. C. Tsui.
    """
    eigenvals, eigenvecs = np.linalg.eig(V)
    d = len(M)
    if not np.any(V): # if V is array of all zeros
        V[:, :] = np.ones((d,d))  * np.finfo(float).eps
    inv_V = np.linalg.inv(V)

    # find the larger projection
    P = np.array([[1,0], [0,0]]) # X-axis projection operator
    P1 = P @ np.reshape((2 * np.sqrt(eigenvals[0]) * eigenvecs[:,0]), (2,1))
    P2 = P @ np.reshape((2 * np.sqrt(eigenvals[1]) * eigenvecs[:,1]), (2,1))
    if (np.all(np.abs(P1) >= np.abs(P2))):
        P_len = P1[0, 0]
    else:
        P_len = P2[0, 0]
    
    count = 0
    step = 0.001 * P_len
    contour_1 = np.zeros((2001,2))
    contour_2 = np.zeros((2001,2))

    for x in np.arange(-P_len, P_len + step, step):
        a = inv_V[1,1]
        b = x * (inv_V[0, 1] + inv_V[1, 0])
        c = x**2 * inv_V[0,0] - 1
        disc = b**2 - 4*a*c 

        if disc >= 0:
            root_1 = (-b + np.sqrt(disc)) / (2*a)
            root_2 = (-b - np.sqrt(disc)) / (2*a)
            if np.isreal(root_1):
                contour_1[count, :] = [x, root_1] + M.T
                contour_2[count, :] = [x, root_2] + M.T
                count += 1
    
    contour_1 = contour_1[0:count-1, :]
    contour_2 = np.vstack((contour_1[0], contour_2[0:count, :], contour_1[count-2, :]))

    plt.plot(M[0], M[1], 'r+')
    plt.plot(contour_1[:,0], contour_1[:,1], 'b-', linewidth=0.5, c='k')
    plt.plot(contour_2[:,0], contour_2[:,1], 'b-', linewidth=0.5, c='k')


def _diverging_norm(data, vmin_override=None):
    vmin = vmin_override if vmin_override is not None else np.min(data)
    vmax = np.max(data)
    if vmin >= 0:
        vmin = -vmax if vmax > 0 else -1
    if vmax <= 0:
        vmax = -vmin if vmin < 0 else 1
    return mc.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin=vmin)


def plot_ctds_matrices(A: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray):
    fig = plt.figure(figsize=(10, 6))

    # Plot A
    axA = fig.add_subplot(221)
    axA.matshow(A, cmap='bwr', norm=_diverging_norm(A))
    axA.set(title='$A$', xlabel='$D$', ylabel='$D$')

    # Plot Q
    axQ = fig.add_subplot(223)
    axQ.matshow(Q, cmap='bwr', norm=_diverging_norm(Q))
    axQ.set(title='$Q$', xlabel='$D$', ylabel='$D$')

    # Plot C
    axC = fig.add_subplot(222)
    axC.matshow(C, cmap='bwr', norm=_diverging_norm(C, vmin_override=-1))
    axC.set(title='$C$', xlabel='$D$', ylabel='$N$')

    # Plot R
    axR = fig.add_subplot(224)
    axR.matshow(R, cmap='bwr', norm=_diverging_norm(R, vmin_override=-1))
    axR.set(title='$R$', xlabel='$N$', ylabel='$N$')

    for ax in [axA, axC, axR, axQ]:
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(True)

    fig.suptitle('Simulated Data Matrices')
    fig.tight_layout()
    plt.show()


def plot_trajectories(
        x_true: np.ndarray,
        x_fit: np.ndarray,
        y_true: np.ndarray,
        y_fit: np.ndarray,
        De: int,
        Ne: int,
        trial: int = 0
):
    """
    Compare ground truth and fitted trajectories for latents and observations.
    2 columns: latents on top, observations on bottom.
    Excitatory dimensions are red, inhibitory dimensions are blue.
    """
    D = x_true.shape[2]
    N = y_true.shape[2]
    ncols = 2
    lat_rows = int(np.ceil(D / ncols))
    obs_rows = int(np.ceil(N / ncols))
    nrows = lat_rows + obs_rows

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2 * nrows), sharex=True)

    # Latents
    for d in range(D):
        row = d // ncols
        col = d % ncols
        ax = axes[row, col]
        color = 'r' if d < De else 'b'
        ax.plot(x_true[trial, :, d, 0], label='True', alpha=0.7, c=color)
        ax.plot(x_fit[trial, :, d, 0], label='Fitted', linestyle='--', alpha=0.7, c=color)
        ax.set_ylabel(f'$x_{{{d+1}}}$')
        if d == 0:
            ax.legend(loc='upper right')
            ax.set_title('Latents')

    # Hide unused latent axes
    for d in range(D, lat_rows * ncols):
        axes[d // ncols, d % ncols].set_visible(False)

    # Observations
    for n in range(N):
        row = lat_rows + n // ncols
        col = n % ncols
        ax = axes[row, col]
        color = 'r' if n < Ne else 'b'
        ax.plot(y_true[trial, :, n, 0], label='True', alpha=0.7, c=color)
        ax.plot(y_fit[trial, :, n, 0], label='Fitted', linestyle='--', alpha=0.7, c=color)
        ax.set_ylabel(f'$y_{{{n+1}}}$')
        if n == 0:
            ax.legend(loc='upper right')
            ax.set_title('Observations')

    # Hide unused observation axes
    for n in range(N, obs_rows * ncols):
        axes[lat_rows + n // ncols, n % ncols].set_visible(False)

    fig.tight_layout()
    plt.show()