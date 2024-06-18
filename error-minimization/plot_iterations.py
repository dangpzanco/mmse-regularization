
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import tqdm

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import wiener_utils as utils
import plot_utils as putils
import rir_generator as rir

# Set random seed
np.random.seed(0)

SNR = 20
ar1 = 0.9

N = 1000
M = 600

rir_options = dict(
    c=340,                    # Sound velocity (m/s)
    fs=8e3,                   # Sample frequency (samples/s)
    r=[2, 1.5, 1],            # Receiver position(s) [x y z] (m)
    s=[2, 3.5, 2],            # Source position [x y z] (m)
    L=[5, 4, 6],              # Room dimensions [x y z] (m)
    reverberation_time=0.225,   # Reverberation time (s)
)
w_true = rir.generate(nsample=M, **rir_options).ravel()

norm_star = la.norm(w_true)
assert M == w_true.size

Ns = np.arange(M // 10, 5*M + 1, M // 10)
SNR = np.array([0, 20])
num_samples = 100

np.seterr(divide='ignore', invalid='ignore')



num_iters = 10
iters = np.arange(num_iters+1)
num_samples = 100
snr_vec = np.array([0, 10, 20])

results = dict(
    grid=np.zeros((snr_vec.size, num_samples)),
    mackay=np.zeros((snr_vec.size, num_samples, num_iters+1)),
    barber=np.zeros((snr_vec.size, num_samples, num_iters+1)),
    fixpoint=np.zeros((snr_vec.size, num_samples, num_iters+1)),
)

for s, SNR in enumerate(tqdm.tqdm(snr_vec)):
    for i in tqdm.trange(num_samples, leave=False):
        X, d = utils.generate_signals(w_true, M, N, snr_vec[s], alpha=ar1)
        wiener = utils.Wiener(X, d, w_true=w_true)

        for key in results.keys():
            if key == 'grid':
                results[key][s, i] = wiener.alpha_grid()
            elif key == 'mackay':
                results[key][s, i, :], _, _ = wiener.alpha_mackay(num_iters=num_iters, alpha0=0.5)
            elif key == 'barber':
                results[key][s, i, :], _, _ = wiener.alpha_barber(num_iters=num_iters, alpha0=0.5)
            elif key == 'fixpoint':
                results[key][s, i, :] = wiener.alpha_fixpoint(num_iters=num_iters, alpha0=0.5)

fig, ax = plt.subplots()
for i, SNR in enumerate(snr_vec):
    ax.axhline(results['grid'][i].mean(), label=f'${SNR}$ dB', color=color_list[i])
    ax.plot(results['mackay'][i].mean(axis=0), ls='--', color=color_list[i])
    ax.plot(results['fixpoint'][i].mean(axis=0), ls='-.', color=color_list[i])
    putils.plot_with_fill(
        ax, iters, (results['grid'][i, :, None] * np.ones(num_iters+1)[None, :]).T,
        ls='-',
        color=color_list[i], alpha=1/num_samples
    )
    # putils.plot_with_fill(
    #     ax, iters, results['mackay'][i].T,
    #     ls='--',
    #     color=color_list[i], alpha=1/num_samples
    # )
    # putils.plot_with_fill(
    #     ax, iters, results['fixpoint'][i].T,
    #     ls='-.',
    #     color=color_list[i], alpha=1/num_samples
    # )
ax.set_yscale('log')
ax.set_xlabel('$i$')
ax.set_ylabel(r'$\hat\alpha, \alpha^{(i)}$')
ax.axis([0, num_iters, None, None])
ax.legend(loc='upper right')
ax.grid()
fig.tight_layout()

putils.save_fig(fig, 'fixedpoint_iteration', format='pdf')


print()

