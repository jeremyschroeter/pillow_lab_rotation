import numpy as np
import matplotlib.pyplot as plt

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