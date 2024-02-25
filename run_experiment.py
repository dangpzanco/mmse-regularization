
import numpy as np
import scipy.linalg as la

import rir_generator as rir

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import wiener_utils as utils

# Set random seed
np.random.seed(0)

SNR = 20
ar1 = 0.9

N = 1000
Lstar = 600
L = 200

# h_star = utils.load_filter(mode='rir', L=Lstar)

rir_options = dict(
    c=340,                    # Sound velocity (m/s)
    fs=8e3,                   # Sample frequency (samples/s)
    r=[2, 1.5, 1],            # Receiver position(s) [x y z] (m)
    s=[2, 3.5, 2],            # Source position [x y z] (m)
    L=[5, 4, 6],              # Room dimensions [x y z] (m)
    reverberation_time=0.225,   # Reverberation time (s)
)
h_star = rir.generate(nsample=Lstar, **rir_options).ravel()

norm_star = la.norm(h_star)
assert Lstar == h_star.size

Ns = np.arange(Lstar, 5*Lstar + 1, Lstar // 10)
SNR = np.array([0, 10, 20])
num_samples = 100
error_grid, error_bayes, best_alpha = utils.run_experiment(
    h_star=h_star, L=L, Ns=Ns, SNR=SNR, num_samples=num_samples, ar1=0.9
)

np.savez_compressed(
    'best_l2reg_mismatch',
    error_grid=error_grid,
    error_bayes=error_bayes,
    best_alpha=best_alpha,
    Ns=Ns, SNR=SNR,
)

error_grid, error_bayes, best_alpha = utils.run_experiment(
    h_star=h_star, L=Lstar, Ns=Ns, SNR=SNR, num_samples=num_samples, ar1=0.9
)

np.savez_compressed(
    'best_l2reg_trueL',
    error_grid=error_grid,
    error_bayes=error_bayes,
    best_alpha=best_alpha,
    Ns=Ns, SNR=SNR,
)
