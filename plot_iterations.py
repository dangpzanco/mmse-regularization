
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import tqdm

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import wiener_utils as utils
import plot_utils as putils

# Set random seed
np.random.seed(0)


ar1 = 0.9
N = 1000
L = 600

h_star = utils.load_filter(mode='rir', L=L)
norm_star = la.norm(h_star)
assert L == h_star.size

num_iters = 10
iters = np.arange(num_iters+1)
num_samples = 100
snr_vec = np.array([0, 10, 20])
best_alpha = np.zeros((num_samples, snr_vec.size, num_iters+1))
alpha1 = np.zeros((num_samples, snr_vec.size, num_iters+1))

for i in tqdm.trange(num_samples):
    for s, SNR in enumerate(snr_vec):
        X_train, d_train = utils.generate_signals(h_star, L, N, SNR, alpha=ar1)
        wiener = utils.Wiener(X_train, d_train, h_star=h_star)

        alpha0 = 10 ** (-SNR/10)
        alpha1[i,s,:], _, _ = wiener.alpha_mckay(num_iters=num_iters, alpha0=0.5)
        best_alpha[i,s,:] = wiener.best_alpha(mode='mis')

fig, ax = plt.subplots()
for i, SNR in enumerate(snr_vec):
    ax.axhline(best_alpha[:,i,:].mean(), label=f'${SNR}$ dB', color=color_list[i])
    ax.plot(alpha1[:,i,].mean(axis=0), ls='--', color=color_list[i])
    putils.plot_with_fill(
        ax, iters, best_alpha[:,i,:].T,
        ls='-',
        color=color_list[i], alpha=1/num_samples)
    putils.plot_with_fill(
        ax, iters, alpha1[:,i,].T,
        ls='--',
        color=color_list[i], alpha=1/num_samples)
ax.set_yscale('log')
ax.set_xlabel('$i$')
ax.set_ylabel(r'$\hat\alpha, \alpha^{(i)}$')
ax.axis([0, num_iters, None, None])
ax.legend(loc='upper right')
ax.grid()
fig.tight_layout()

putils.save_fig(fig, 'mckay_iteration', format='pdf')


print()

