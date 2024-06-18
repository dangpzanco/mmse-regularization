
import numpy as np
import scipy.linalg as la
import tqdm

import rir_generator as rir

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import wiener_utils as utils


def run_experiment(w_true, M=None, Ns=None, SNR=None, num_samples=1, ar1=0.9):

    if Ns is None:
        Ns = np.logspace(np.log10(M), np.log10(M)+1, 50).astype(int)

    if SNR is None:
        SNR = np.array([0, 5, 10, 15, 20, 25])

    misalignment = dict(
        mackay=np.zeros([Ns.size, SNR.size, num_samples]),
        # fixpoint=np.zeros([Ns.size, SNR.size, num_samples]),
        # barber=np.zeros([Ns.size, SNR.size, num_samples]),
        ledoit=np.zeros([Ns.size, SNR.size, num_samples]),
        hkb=np.zeros([Ns.size, SNR.size, num_samples]),
        grid=np.zeros([Ns.size, SNR.size, num_samples]),
        no_alpha=np.zeros([Ns.size, SNR.size, num_samples]),
    )

    best_alpha = dict(
        mackay=np.zeros([Ns.size, SNR.size, num_samples]),
        # fixpoint=np.zeros([Ns.size, SNR.size, num_samples]),
        # barber=np.zeros([Ns.size, SNR.size, num_samples]),
        ledoit=np.zeros([Ns.size, SNR.size, num_samples]),
        hkb=np.zeros([Ns.size, SNR.size, num_samples]),
        grid=np.zeros([Ns.size, SNR.size, num_samples]),
    )

    for n, N in enumerate(tqdm.tqdm(Ns)):
        for s in tqdm.trange(SNR.size, leave=False):
            for r in tqdm.trange(num_samples, leave=False):
                X, d = utils.generate_signals(w_true, M, N, SNR[s], alpha=ar1)

                wiener = utils.Wiener(X, d, w_true=w_true)
                # options = dict(alpha0=0.5, num_iters=20)
                options = dict(alpha0=0.5, num_iters=5)
                for key in misalignment.keys():
                    if key == 'no_alpha':
                        misalignment[key][n, s, r] = wiener.misalignment(0)
                        continue

                    alpha = wiener.best_alpha(mode=key, **options)
                    best_alpha[key][n, s, r] = alpha
                    misalignment[key][n, s, r] = wiener.misalignment(alpha)


    for key in misalignment.keys():
        misalignment[key] = 10 * np.log10(misalignment[key])

    return misalignment, best_alpha

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

# Ns = np.arange(M // 10, 5*M + 1, M // 10)
Ns = np.arange(M, 5*M + 1, M // 10)
SNR = np.array([0, 20])
num_samples = 100

np.seterr(divide='ignore', invalid='ignore')

misalignment, best_alpha = run_experiment(
    w_true=w_true, M=M, Ns=Ns, SNR=SNR, num_samples=num_samples, ar1=0.9
)

np.savez_compressed(
    'wiener_misalignment',
    misalignment=misalignment,
    best_alpha=best_alpha,
    Ns=Ns, SNR=SNR,
)


