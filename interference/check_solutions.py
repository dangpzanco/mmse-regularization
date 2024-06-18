import numpy as np
import numpy.random as rnd
import scipy.linalg as la
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt
import tqdm
import opt_einsum as oe

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import capon_utils as capon
import plot_utils as putils
import rir_generator as rir

import warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

seed = 0
rng = rnd.default_rng(seed)


# Number of antennas
M = 10

# Max number of samples
# N = 500
# N = 1000

# Number of samples to check
# Ns = np.arange(M, N+M, M)
# Ns = np.arange(2, 10 * M + 2, 2)
# Ns = np.arange(M, 10000 + M, M)
Ns = np.logspace(1, 4, 20).astype(int)

# Index of SOI
soi_index = 0

# Angles
phi = np.pi * np.array([0.2, 0.4, 0.7])
K = phi.size

# Signal powers
Pk = 10.0 ** (np.array([30, 20, 10]) / 10)

# Number of Monte Carlo samples
num_samples = 5

# Angles
phi = np.pi * np.array([0.2, 0.3, 0.6])
K = phi.size

# Signal powers
Pk = 10.0 ** (np.array([20, 10, 5]) / 10)

# Create steering vector
a = capon.get_sterring_vector(phi, M)

# Get analytical covariance matrix
Rtrue = oe.contract('k,km,kn->mn', Pk, a, a.conj()) + np.eye(M)

# Input SINR
SINRin = M * Pk / (np.trace(Rtrue).real - M * Pk)

# # for soi_index in tqdm.trange(K, leave=False):
# # Optimal SINR
# Rtrue_inv = np.linalg.inv(Rtrue)
# a_soi = a_t[soi_index]
# w = Rtrue_inv @ a_soi / (a_soi.conj() @ Rtrue_inv @ a_soi)
# spec = (w.conj() @ Rtrue @ w).real

# w_a = Pk[soi_index] * np.abs(w @ a_soi.conj()) ** 2
# sinr = w_a / (spec - w_a)

# # Target signal information
# target = {
#     'a_t': a_t[soi_index],
#     'sigma_s': np.sqrt(Pk[soi_index]),
#     'R': Rtrue
# }

# # for i, N in enumerate(tqdm.tqdm(Ns, leave=False)):
# # Input signal
# xx = x[:, :N]

# # Target angle
# a_t = capon.get_sterring_vector(phi, M, noise=0)[soi]
# beamformer = capon.RegularizedCapon(a_t, xx, target=target)

# # Target signal information
# oracle = {
#     'v_s': Pk[soi_index],
#     'R': Rtrue
# }
# wiener = capon.Wiener(a_t, xx, oracle=oracle)



num_iters = 10
iters = np.arange(num_iters+1)
num_samples = 10000

v_x = np.zeros([K, num_samples, Ns.size])
v_d = np.zeros([K, num_samples, Ns.size])
v_xd = np.zeros([K, num_samples, Ns.size])
cond = np.ones([K, num_samples, Ns.size], dtype=bool)

alpha0 = np.logspace(-5, 6, 12)
num_inits = alpha0.size

roots = np.zeros([K, num_samples, Ns.size, num_inits])
loss = np.zeros([K, num_samples, Ns.size, num_inits])
loss_inf = np.zeros([K, num_samples, Ns.size])
num_roots = np.zeros([K, num_samples, Ns.size])

# Create steering vector
a = capon.get_sterring_vector(phi, M, noise=0)

# x_samples = np.zeros([K, num_samples, M, Ns.max()], dtype=complex)

for k in tqdm.trange(K):
    # Target angle
    a_t = a[k]

    for i in tqdm.trange(num_samples, leave=False):
        x = capon.generate_input(Ns.max(), M, a, Pk, seed=rng)
        # x_samples[k, i] = x

        for n, N in enumerate(tqdm.tqdm(Ns, leave=False)):

            # Target signal information
            oracle = {
                'v_s': Pk[k],
                'R': Rtrue
            }
            wiener = capon.Wiener(a_t, x[:, :N], oracle=oracle)

            xx = wiener.X
            dd = wiener.d

            v_x[k, i, n] = np.trace(wiener.Rx)
            v_d[k, i, n] = (dd @ dd.conj()) / N
            v_xd[k, i, n] = N * (wiener.rxd @ wiener.rxd.conj())

            cond[k, i, n] = v_xd[k, i, n] > v_d[k, i, n] * v_x[k, i, n]

            for j, a0 in enumerate(alpha0):
                sol, _, _ = wiener.alpha_mackay(num_iters=5, alpha0=a0)
                roots[k, i, n, j] = sol[-1]

            # if (np.log10(roots[k, i, n]).std() > 0.1) and cond[k, i, n]:
            #     for j, a0 in enumerate(alpha0):
            #         sol, _, _ = wiener.alpha_mackay(num_iters=1000, alpha0=a0)
            #         roots[k, i, n, j] = sol[-1]

            loss[k, i, n] = wiener.likelihood(roots[k, i, n])
            loss_inf[k, i, n] = wiener.N * np.log(wiener.v_d)
            # assert loss_inf[k, i, n] == wiener.likelihood(np.inf)

            alpha = np.logspace(-10, 20, 1000)
            dL = wiener.d_likelihood(alpha)
            num_roots[k, i, n] = np.abs(np.diff(0.5*np.sign(dL))).sum() + 1

            # if cond[k, i, n]:
            #     for j, a0 in enumerate(alpha0):
            #         sol, _, _ = wiener.alpha_mackay(num_iters=100, alpha0=a0)
            #         roots[k, i, n, j] = sol[-1]

            #     if np.log10(roots[k, i, n]).std() > 0.1:
            #         for j, a0 in enumerate(alpha0):
            #             sol, _, _ = wiener.alpha_mackay(num_iters=1000, alpha0=a0)
            #             roots[k, i, n, j] = sol[-1]


            # if (not cond[k, i, n]) or (np.log10(roots[k, i, n]).std() > 0.1):
            # if (np.log10(roots[k, i, n]).std() > 0.6) and cond[k, i, n]:
            # if (not cond[k, i, n]) and (loss[k, i, n].min() - loss_inf[k, i, n] < -0.01):
            # if (not cond[k, i, n]) and (loss[k, i, n].min() < loss_inf[k, i, n]):
            if (not cond[k, i, n]) and (loss_inf[k, i, n] - loss[k, i, n].min() > 0.01):
                alpha = np.logspace(-10, 20, 1000)
                L = wiener.likelihood(alpha)
                dL = wiener.d_likelihood(alpha)
                sinr = wiener.sinr(alpha)

                res = opt.minimize_scalar(wiener.likelihood, bracket=(-3, 13), method='brent').x

                fig, ax = plt.subplots()
                ax.plot(alpha, L, label=r"$L(\alpha)$")
                ax.plot(alpha, dL, label=r"$L'(\alpha)$")
                ax.plot(alpha, sinr, label=r"SINR", color='k')
                ax.set_xscale('log')
                ax.set_yscale('symlog')
                ax.grid()
                ax.set_xlabel(r"$\alpha$")
                ax.legend()

                # fig, ax = plt.subplots()
                # ax.plot(alpha0, roots[k, i, n])
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # ax.grid()
                # ax.set_xlabel(r"$\alpha^{(0)}$")
                # ax.set_ylabel(r"$\alpha^{(I)}$")

                fig, ax = plt.subplots()
                # ax.plot(alpha, np.log(np.abs(dL)) * np.sign(dL), label=r"$L'(\alpha)$")
                ax.plot(alpha, dL, label=r"$L'(\alpha)$")
                ax.plot(alpha, np.sign(dL), label=r"$L'(\alpha)$")
                ax.set_xscale('log')
                ax.set_yscale('symlog')
                ax.grid()
                ax.set_xlabel(r"$\alpha$")
                ax.legend()

            #     print()


np.savez(
    'condition.npz',
    # x_samples=x_samples,
    loss=loss,
    loss_inf=loss_inf,
    a=a,
    Rtrue=Rtrue,
    v_x=v_x,
    v_d=v_d,
    v_xd=v_xd,
    cond=cond,
    roots=roots,
    num_roots=num_roots,
    Pk=Pk,
    phi=phi,
    num_samples=num_samples,
    Ns=Ns,
    alpha0=alpha0,
)



print()
