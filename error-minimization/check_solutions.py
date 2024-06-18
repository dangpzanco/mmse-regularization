
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import tqdm

import warnings
warnings.filterwarnings('ignore')

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
# w_true = rir.generate(nsample=2*M, **rir_options).ravel()
# w_true = np.trim_zeros(w_true)[:M]
w_true = rir.generate(nsample=M, **rir_options).ravel()

norm_star = la.norm(w_true)
assert M == w_true.size

# Ns = np.arange(M // 10, 5*M + 1, M // 10)

np.seterr(divide='ignore', invalid='ignore')



num_samples = 1000
snr_vec = np.array([0, 10, 20])
# Ns = np.arange(2, 2*M, 2)
# Ns = np.arange(4, 2*M+1, 4)
Ns = np.arange(2, 151, 2)


v_x = np.zeros([snr_vec.size, num_samples, Ns.size])
v_d = np.zeros([snr_vec.size, num_samples, Ns.size])
v_xd = np.zeros([snr_vec.size, num_samples, Ns.size])
cond = np.zeros([snr_vec.size, num_samples, Ns.size], dtype=bool)

# alpha0 = np.logspace(-10, 20, 31)
alpha0 = 10.0 ** np.arange(-5, 5)
num_inits = alpha0.size

roots = np.zeros([snr_vec.size, num_samples, Ns.size, num_inits])
loss = np.zeros([snr_vec.size, num_samples, Ns.size, num_inits])
loss_inf = np.zeros([snr_vec.size, num_samples, Ns.size])
num_roots = np.zeros([snr_vec.size, num_samples, Ns.size])


for s, SNR in enumerate(tqdm.tqdm(snr_vec)):
    for i in tqdm.trange(num_samples, leave=False):
        X, d = utils.generate_signals(w_true, M, Ns.max(), snr_vec[s], alpha=ar1)
        for n, N in enumerate(tqdm.tqdm(Ns, leave=False)):
            xx = X[:, :N]
            dd = d[:N]
            # Rx = (1 / N) * (xx @ xx.T)
            # rxd = (1 / N) * (xx @ dd)
            wiener = utils.Wiener(xx, dd, w_true=w_true)
            for j, a0 in enumerate(alpha0):
                # sol = wiener.alpha_fixpoint(num_iters=100, alpha0=a0)
                sol, _, _ = wiener.alpha_mackay(num_iters=100, alpha0=a0)
                roots[s, i, n, j] = sol[-1]

            # if np.log10(roots[s, i, n]).std() > 0.1:
            #     wiener = utils.Wiener(xx, dd, w_true=w_true)
            #     for j, a0 in enumerate(alpha0):
            #         sol, _, _ = wiener.alpha_mackay(num_iters=1000, alpha0=a0)
            #         roots[s, i, n, j] = sol[-1]
            
            # if roots[s, i, n].std() > 1:
            #     print()

            v_x[s, i, n] = np.trace(wiener.Rx)
            # v_d[s, i, n] = la.norm(dd) ** 2 / N
            # v_xd[s, i, n] = N * la.norm(rxd) ** 2
            v_d[s, i, n] = (dd @ dd.conj()) / N
            v_xd[s, i, n] = N * (wiener.rxd @ wiener.rxd.conj())

            cond[s, i, n] = v_xd[s, i, n] > v_d[s, i, n] * v_x[s, i, n]

            loss[s, i, n] = wiener.likelihood(roots[s, i, n])
            loss_inf[s, i, n] = wiener.N * np.log(wiener.v_d)

            alpha = np.logspace(-10, 20, 1000)
            dL = wiener.d_likelihood(alpha)
            num_roots[s, i, n] = np.abs(np.diff(0.5*np.sign(dL))).sum() + 1

            # # if (not cond[s, i, n]) or (np.log10(roots[s, i, n]).std() > 0.1):
            # # if (not cond[s, i, n]) or (np.log10(roots[s, i, n]).std() > 0.1):
            # # if (not cond[s, i, n]) and (loss_inf[s, i, n] - loss[s, i, n].min() > 0.01):
            # if (not cond[s, i, n]) or (loss_inf[s, i, n] - loss[s, i, n].min() > 0.01):
            # if True:
            #     wiener = utils.Wiener(xx, dd, w_true=w_true)
            #     alpha = np.logspace(-10, 20, 1000)
            #     L = wiener.likelihood(alpha)
            #     dL = wiener.d_likelihood(alpha)
                
            #     fig, ax = plt.subplots()
            #     ax.plot(alpha, L, label=r"$L(\alpha)$")
            #     ax.plot(alpha, dL, label=r"$L'(\alpha)$")
            #     ax.set_xscale('log')
            #     ax.set_yscale('symlog')
            #     ax.grid()
            #     ax.legend()
            #     ax.set_xlabel(r"$\alpha$")

            #     fig, ax = plt.subplots()
            #     ax.plot(alpha0, roots[s, i, n])
            #     ax.set_xscale('log')
            #     ax.set_yscale('log')
            #     ax.grid()
            #     ax.set_xlabel(r"$\alpha^{(0)}$")
            #     ax.set_ylabel(r"$\alpha^{(I)}$")

            #     print()


np.savez(
    'condition.npz',
    v_x=v_x,
    v_d=v_d,
    v_xd=v_xd,
    cond=cond,
    roots=roots,
    num_roots=num_roots,
    snr_vec=snr_vec,
    num_samples=num_samples,
    Ns=Ns,
    alpha0=alpha0,
    w_true=w_true,
    ar1=ar1
)


print()
