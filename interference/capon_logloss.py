
import numpy as np
import numpy.random as rnd
from scipy import io
import scipy.signal as sig
import numpy.linalg as la
import matplotlib.pyplot as plt
import opt_einsum as oe
import tqdm
import librosa
import sympy as sp
import scipy.optimize as sci_opt

import rir_generator as rir

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import plot_utils as putils
import capon_utils as capon


def get_signals(N, num_signals, Pk=1, seed=None):
    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd

    if np.isscalar(Pk):
        Pk = Pk * np.ones(num_signals)

    s = np.empty((num_signals, N), dtype=np.complex128)
    for i in range(num_signals):
        s[i] = rng.normal(loc=0, scale=np.sqrt(Pk[i] / 2), size=N) \
        + 1j * rng.normal(loc=0, scale=np.sqrt(Pk[i] / 2), size=N)
    return s


def get_sterring_vector(angles, L, noise=0, noise_type='angle', seed=None):
    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd

    # noise = noise * rng.uniform(low=-1, high=1, size=L)
    if noise_type == 'angle':
        phi_l = np.outer(np.cos(angles + noise), np.arange(L))
        ak = np.exp(1j * np.pi * phi_l)
    elif noise_type == 'additive':
        noise = rng.normal(scale=np.sqrt(noise / 2), size=(angles.size, L)) \
         + 1j * rng.normal(scale=np.sqrt(noise / 2), size=(angles.size, L))
        phi_l = np.outer(np.cos(angles), np.arange(L))
        ak = np.exp(1j * np.pi * phi_l) + noise
    elif noise_type == 'phase':
        delta = rng.normal(scale=noise, size=L)
        phi_l = np.outer(np.cos(angles), np.arange(L)) + delta[None, :]
        ak = np.exp(1j * np.pi * phi_l)

    return ak

def run_experiment_k(
        Ns, L, num_samples=1000, phi=None,
        noise=0.1, noise_type='additive', Pk=None, alpha=None, seed=None):

    if phi is None:
        phi = np.array([0.2, 0.4, 0.7]) * np.pi

    if Pk is None:
        Pk = 10.0 ** (np.array([10, 20, 30]) / 10)

    if alpha is None:
        alpha = np.logspace(-3, 2, 6)
        alpha = np.hstack((alpha, 0))

    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd

    K = phi.size
    num_alpha = alpha.size

    SINR = dict(
        mackay=np.zeros((K, Ns.size)),
        barber=np.zeros((K, Ns.size)),
        grid=np.zeros((K, Ns.size)),
        hkb=np.zeros((K, Ns.size)),
        ledoit=np.zeros((K, Ns.size)),
        lawless=np.zeros((K, Ns.size)),
        choice=np.zeros((K, Ns.size, num_alpha)),
        optimal=np.zeros(K),
        input=np.zeros(K),
    )
    best_alpha = dict(
        mackay=np.zeros((K, num_samples, Ns.size)),
        barber=np.zeros((K, num_samples, Ns.size)),
        grid=np.zeros((K, num_samples, Ns.size)),
        hkb=np.zeros((K, num_samples, Ns.size)),
        ledoit=np.zeros((K, num_samples, Ns.size)),
        lawless=np.zeros((K, num_samples, Ns.size)),
    )
    for mc in tqdm.trange(num_samples):
        # Create steering vector
        a_t = get_sterring_vector(phi, L, noise=noise, noise_type=noise_type, seed=rng)

        # Create signals
        s = get_signals(Ns.max(), K, Pk=Pk, seed=rng)

        # Create noise
        z = rng.normal(scale=np.sqrt(0.5), size=(L, Ns.max())) \
        + 1j * rng.normal(scale=np.sqrt(0.5), size=(L, Ns.max()))

        # Create received signal
        x = a_t.T @ s + z

        # Get analytical covariance matrix
        Rtrue = oe.contract('k,km,kn->mn', Pk, a_t, a_t.conj()) + np.eye(L)

        # Input SINR
        SINRin = L * Pk / (np.trace(Rtrue).real - L * Pk)
        SINR['input'] += SINRin

        for soi_index in tqdm.trange(K, leave=False):
            # Optimal SINR
            Rtrue_inv = np.linalg.inv(Rtrue)
            a_soi = a_t[soi_index]
            w = Rtrue_inv @ a_soi / (a_soi.conj() @ Rtrue_inv @ a_soi)
            spec = (w.conj() @ Rtrue @ w).real

            w_a = Pk[soi_index] * np.abs(w @ a_soi.conj()) ** 2
            sinr = w_a / (spec - w_a)
            SINR['optimal'][soi_index] += sinr

            # Target signal information
            target = {
                'a_t': a_t[soi_index],
                'sigma_s': np.sqrt(Pk[soi_index]),
                'R': Rtrue
            }
            for i, N in enumerate(tqdm.tqdm(Ns, leave=False)):
                # Input signal
                xx = x[:, :N]

                # Target angle
                a_target = get_sterring_vector(phi, L, noise=0)[soi_index]
                beamformer = capon.RegularizedCapon(a_target, xx, target=target)

                # Debug
                a_opt = beamformer.alpha_grid(alpha0=1, num_iters=100)
                # spec, norm_w, gamma = beamformer._iter_alpha(0)
                # fig, ax = plt.subplots()
                # ax.plot(a_opt)
                # ax.set_yscale('log')

                a = np.logspace(-20, 20, 10000)
                vals = np.array(list(map(beamformer._iter_alpha, a)))
                spec, norm_w, gamma = vals.T
                fig, ax = plt.subplots()
                ax.plot(a, spec, label='spec')
                ax.plot(a, norm_w, label='norm w')
                ax.plot(a, gamma, label='gamma')
                ax.legend()
                ax.set_xscale('log')
                ax.set_yscale('log')

                a = np.logspace(-20, 20, 10000)
                w = beamformer.w_hat(a)
                sinr_w = beamformer.sinr(w)
                fig, ax = plt.subplots()
                ax.plot(a, sinr_w)
                ax.axvline(a_opt[-1], color='k', linestyle='--')
                ax.set_xscale('log')
                ax.set_yscale('log')

                # options = dict(num_iters=1000, alpha0=a_opt[-1])
                options = dict(num_iters=1000, alpha0=1)
                eps_list = [0, 1e-3, 1e-6]
                debug_list = []
                for eps in eps_list:
                    debug_list.append(beamformer.alpha_debug(eps=eps, **options))

                fig, ax = plt.subplots()
                for i, debug in enumerate(debug_list):
                    ax.plot(debug['mackay']['alpha'], label=f'Mackay, $\\epsilon={eps_list[i]:.0e}$')
                ax.plot(debug_list[0]['barber']['alpha'], label='Barber')
                ax.axhline(a_opt[-1], color='k', linestyle='--', label='Optimal')
                ax.set_yscale('log')
                ax.set_title(f'$N = {N}$')
                ax.legend()

                a = np.logspace(-20, 20, 10000)
                w = beamformer.w_hat(a)
                sinr_w = beamformer.sinr(w)
                fig, ax = plt.subplots()
                ax.plot(a, sinr_w)
                ax.axvline(a_opt[-1], color='k', linestyle='--')
                for i, debug in enumerate(debug_list):
                    w = beamformer.w_hat(debug['mackay']['alpha'][-1])
                    sinr_w = beamformer.sinr(w)
                    ax.axvline(sinr_w, color=color_list[i], label=f'Mackay, $\\epsilon={eps_list[i]:.0e}$')
                w = beamformer.w_hat(debug['barber']['alpha'][-1])
                sinr_w = beamformer.sinr(w)
                ax.axvline(sinr_w, color=color_list[i+1], label='Barber')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()


                print()

                for key in best_alpha.keys():
                    alpha_soi = beamformer.alpha_opt(method=key, num_iters=20)
                    best_alpha[key][soi_index, mc, i] = alpha_soi

                    w_hat = beamformer.w_hat(alpha_soi)
                    sinr = beamformer.sinr(w_hat)
                    spec = beamformer.spectrum(w_hat)
                    SINR[key][soi_index, i] += sinr

                # Chosen alpha (target angle)
                w_hat = beamformer.w_hat(alpha)
                sinr = beamformer.sinr(w_hat)
                spec = beamformer.spectrum(w_hat)
                SINR['choice'][soi_index, i] += sinr

    for key in SINR.keys():
        SINR[key] /= num_samples

    results = dict(
        Ns=Ns,
        SINR=SINR,
        best_alpha=best_alpha,
        L=L, num_samples=num_samples,
        Pk=Pk, alpha=alpha,
        soi_index=soi_index, seed=seed
    )

    return results


# Number of antennas
L = 10

# Max number of samples
# N = 500
N = 1000

# Number of samples to check
Ns = np.arange(L, N+L, L)

# Index of SOI
soi_index = 0

# Angles
phi = np.pi * np.array([0.2, 0.4, 0.7])
# phi = np.array([0, 20, 60]) * np.pi / 180
# phi = np.array([0, 90, 180]) * np.pi / 180
# phi = np.pi * np.array([0.2, 0.3, 0.5, 0.6, 0.8])
K = phi.size

# Signal powers
# Pk = 10.0 ** (np.array([30, 20, 10]) / 10)
Pk = 10.0 ** (np.array([10, 20, 20]) / 10)
# Pk = 10.0 ** (np.array([20, 10, 10, 20, 10]) / 10)

# l2 regularization
alpha = np.hstack((np.logspace(0, 5, 6), 0))
num_alpha = alpha.size

# Steerring vector noise
# noise = 0.01
noise = 0

# alpha = np.array([0, 1, 10])
alpha = np.array([0])
num_alpha = alpha.size

# Number of Monte Carlo samples
num_samples = 100
# num_samples = 5

# Angles
phi = np.pi * np.array([0.2, 0.3, 0.6])
# phi = np.pi * np.array([0.22, 0.31, 0.63])
# phi = np.pi * np.array([0.2, 0.1, 0.3, 0.4, 0.6, 0.8])
K = phi.size

# Signal powers
Pk = 10.0 ** (np.array([20, 10, 5]) / 10)
# Pk = 10.0 ** (np.array([15, 10, 5]) / 10)
# Pk = 10.0 ** (np.array([10, 10, 5]) / 10)
# Pk = 10.0 ** (np.array([20, 10, 10, 10, 10, 10]) / 10)


########################### DEBUG ###########################

N = 1000
soi_index = 0

seed = 0
rng = rnd.default_rng(seed)

# Create steering vector
a_t = get_sterring_vector(phi, L, noise=0)

# Create signals
s = get_signals(Ns.max(), K, Pk=Pk, seed=rng)

# Create noise
z = rng.normal(scale=np.sqrt(0.5), size=(L, Ns.max())) \
+ 1j * rng.normal(scale=np.sqrt(0.5), size=(L, Ns.max()))

# Create received signal
x = a_t.T @ s + z

# Get analytical covariance matrix
Rtrue = oe.contract('k,km,kn->mn', Pk, a_t, a_t.conj()) + np.eye(L)

# Input SINR
SINRin = L * Pk / (np.trace(Rtrue).real - L * Pk)

# for soi_index in tqdm.trange(K, leave=False):
# Optimal SINR
Rtrue_inv = np.linalg.inv(Rtrue)
a_soi = a_t[soi_index]
w = Rtrue_inv @ a_soi / (a_soi.conj() @ Rtrue_inv @ a_soi)
spec = (w.conj() @ Rtrue @ w).real

w_a = Pk[soi_index] * np.abs(w @ a_soi.conj()) ** 2
sinr = w_a / (spec - w_a)

# Target signal information
target = {
    'a_t': a_t[soi_index],
    'sigma_s': np.sqrt(Pk[soi_index]),
    'R': Rtrue
}

# for i, N in enumerate(tqdm.tqdm(Ns, leave=False)):
# Input signal
xx = x[:, :N]

# Target angle
a_t = get_sterring_vector(phi, L, noise=0)[soi_index]
beamformer = capon.RegularizedCapon(a_t, xx, target=target)


# Target signal information
oracle = {
    'v_s': Pk[soi_index],
    'R': Rtrue
}

wiener = capon.Wiener(a_t, xx, oracle=oracle)
w_hat = wiener.w_hat(1)
sinr = wiener.sinr(1)

a = np.logspace(-20, 20, 1000)
u = wiener.u_hat(a)
w = wiener.w_hat(a)
sinr = wiener.sinr(a)

fig, ax = plt.subplots()
# ax.plot(a, np.abs(u @ a_t.conj()))
# ax.plot(a, np.abs(w @ a_t.conj()))
# ax.plot(a, la.norm(u, axis=-1) ** 2)
# ax.plot(a, la.norm(w, axis=-1) ** 2)
ax.plot(a, sinr)
# ax.plot(a, beamformer.sinr(beamformer.w_hat(a)))
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()


a_opt = wiener.alpha_grid()
res = sci_opt.minimize_scalar(lambda x: -wiener.sinr(x))



# Debug
a_opt = beamformer.alpha_grid()



print()

def logloss(wiener: capon.Wiener, alpha):
    N = wiener.N
    M = wiener.M

    w_hat = wiener.w_hat(alpha)

    Ra_inv = wiener.Q @ np.diag(1 / (wiener.lamb + alpha)) @ wiener.Q.T
    # lambA = 1 / (wiener.lamb[:, None] + alpha[None, :])
    # Ra_inv = oe.contract('ij,ja,jk->aik', wiener.Q, lambA, wiener.Q.T)

    val = -la.slogdet(Ra_inv)[-1] - M * np.log(alpha) + N * np.log(wiener.v_d - w_hat.T @ wiener.rxd)
    return val


def dlogloss(wiener: capon.Wiener, alpha):
    N = wiener.N
    mse, norm_w, gamma = wiener._alpha_iter(alpha)
    val = N * norm_w / (mse + alpha * norm_w) - gamma / alpha

    return val


def root_finder(wiener: capon.Wiener):
    N = wiener.N
    M = wiener.M
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    alpha = sp.symbols('alpha')
    D = sp.prod(lamb + alpha)

    Dm = [None] * M
    for m in range(M):
        Dm[m] = sp.poly(D / (lamb[m] + alpha))
    Dm = np.array(Dm)

    A = -np.sum(lamb * Dm)
    B = np.sum(zxd2 * Dm ** 2)
    C = np.sum(zxd2 * Dm)

    L = A * (v_d * D - C) + N * B

    roots = []
    for root in L.real_roots():
        roots.append(root.evalf())
    roots = np.array(roots).astype(float)

    # Lfull = L / (D * (v_d * D - C))
    Lfull = A / D + N * (B / D ** 2) / (v_d - C / D)

    return Lfull, roots[roots > 0]


def poly_func(wiener: capon.Wiener):
    N = wiener.N
    M = wiener.M
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    alpha = sp.symbols('alpha')
    gamma = np.sum(lamb / (lamb + alpha))
    ww = np.sum(zxd2 / (lamb + alpha) ** 2)
    rw = np.sum(zxd2 / (lamb + alpha))

    L = -gamma / alpha + N * ww / (v_d - rw)
    return L

def poly_func_a(wiener: capon.Wiener, alpha):
    N = wiener.N
    M = wiener.M
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    gamma = np.sum(lamb / (lamb + alpha))
    ww = np.sum(zxd2 / (lamb + alpha) ** 2)
    rw = np.sum(zxd2 / (lamb + alpha))

    L = -gamma / alpha + N * ww / (v_d - rw)
    return L


# Lpoly, roots = root_finder(wiener)
# Lpoly.evalf(subs={'alpha': 1})
# # L2 = np.array(list(map(lambda x: Lpoly.eval(x), a)))
# L2 = np.array(list(map(lambda x: Lpoly.evalf(subs={'alpha': x}), a)))

Lpoly = poly_func(wiener)
L2 = np.array(list(map(lambda x: Lpoly.evalf(subs={'alpha': x}), a)))
L3 = np.array(list(map(lambda x: poly_func_a(wiener, x), a)))

L = np.array(list(map(lambda x: logloss(wiener, x), a)))
dL = np.array(list(map(lambda x: dlogloss(wiener, x), a)))
a_mackay, _, _ = wiener.alpha_mackay(num_iters=100, alpha0=1)
a_opt = wiener.alpha_grid()


fig, ax = plt.subplots()
# ax.plot(a, sinr)
# ax.plot(a, L)
ax.plot(a, L2)
ax.plot(a, L3)
ax.plot(a, dL)
# ax.axvline(a_mackay[-1], color='k', linestyle='--', label='Mackay')
# ax.axvline(a_opt, color='tab:purple', linestyle='--', label='Optimal')
ax.set_yscale('symlog')
ax.set_xscale('log')
ax.legend()
ax.grid()

print()