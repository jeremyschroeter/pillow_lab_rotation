import numpy as np
from pillow_lab_rotation import ctds, simulate
import matplotlib.pyplot as plt

def main():
    Ne, Ni = 5, 5
    N = Ne + Ni
    De, Di = 2, 3
    D = De + Di

    simulation = simulate.CTDSSim(De, Di, Ne, Ni)
    np.random.seed(42)
    simulation.create_params()

    trial_list = [50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000, 20000, 40000]
    time_points = 100
    X_all, Y_all = simulation.simulate(time_points, trial_list[-1])

    A_loss = []
    C_loss = []
    Q_loss = []
    R_loss = []
    A_true, C_true, Q_true, Q0_true, mu0_true, R_true = simulation.get_params()

    np.random.seed(0)
    model = ctds.CTDS(De, Di, Ne, Ni)

    for n_trials in trial_list:
        print(f'Fitting with {n_trials} trials')
        # Initialize at the ground truth parameters
        model.A = A_true.copy()
        model.C = C_true.copy()
        model.Q = Q_true.copy()
        model.R = R_true.copy()
        model.Q0 = Q0_true.copy()
        model.mu0 = mu0_true.copy()

        Y = Y_all[:n_trials]
        model.fit(Y)

        C_prime = model.C
        H = np.linalg.inv(C_prime.T @ C_prime) @ C_prime.T @ C_true
        H_inv = np.linalg.inv(H)
        
        A_rec = H_inv @ model.A @ H
        C_rec = model.C @ H
        Q_rec = H_inv @ model.Q @ H_inv.T

        A_loss.append(np.mean((A_rec - A_true)**2))
        C_loss.append(np.mean((C_rec - C_true)**2))
        Q_loss.append(np.mean((Q_rec - Q_true)**2))
        R_loss.append(np.mean((model.R - R_true)**2))

    # Plot
    fig, axes = plt.subplots(2, 2)
    titles = ['$A$ MSE (aligned)', '$C$ MSE (aligned)', '$Q$ MSE (aligned)', '$R$ MSE']
    losses = [A_loss, C_loss, Q_loss, R_loss]

    for ax, title, loss in zip(axes.flat, titles, losses):
        ax.plot(time_points * np.array(trial_list), loss)
        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('time points')

    fig.suptitle('Initializing at true parameters')
    fig.tight_layout()
    plt.savefig('../figures/ctds_param_recovery.png', dpi=300)


if __name__ == '__main__':
    main()