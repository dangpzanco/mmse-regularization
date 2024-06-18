
import numpy as np
import numpy.random as rnd
from scipy import io
import scipy.signal as sig
import scipy.linalg as la
import matplotlib.pyplot as plt
import opt_einsum as oe
import tqdm
import librosa
import sympy as sp

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

N = 10
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

fig, ax = plt.subplots()
# ax.plot(a, np.abs(u @ a_t.conj()))
# ax.plot(a, np.abs(w @ a_t.conj()))
ax.plot(a, la.norm(u, axis=-1) ** 2)
ax.plot(a, la.norm(w, axis=-1) ** 2)
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid()


# Debug
a_opt = beamformer.alpha_grid()

def dlog_loss(beamformer: capon.RegularizedCapon, alpha):
    spec, norm_w, gamma = np.array(list(map(beamformer._iter_alpha, alpha))).T
    N = beamformer.N
    M = beamformer.M

    norm_w -= 1 / M

    val = N * norm_w / (spec + alpha * norm_w) - gamma/alpha
    return val

def solve_poly(beamformer: capon.RegularizedCapon):
    N = beamformer.N
    M = beamformer.M
    lamb = beamformer.lamb
    b2 = beamformer.b2

    alpha = sp.symbols('alpha')
    D = sp.prod(lamb + alpha)

    Dm = [None] * M
    for m in range(M):
        Dm[m] = sp.poly(D / (lamb[m] + alpha))
    Dm = np.array(Dm)

    A = np.sum(lamb * b2 * Dm)
    B = np.sum(b2 * Dm ** 2)
    C = np.sum(b2 * Dm)
    E = np.sum(lamb * Dm)
    F = B - 1/M * C**2
    D = sp.poly(D)

    # L = N * F / (A + alpha*F) - (C*E - A) / (alpha*C*D)
    # L = N * F *(alpha*C*D) - (C*E - A) * (A + alpha*F)

    # L = sp.poly(sp.simplify(L))
    J = alpha * C * D * (A + alpha * F)
    L = N*alpha*C*D*F - (C*E-A)*(A+alpha*F)

    roots = []
    for root in L.real_roots():
        roots.append(root.evalf())
    roots = np.array(roots).astype(float)

    return L, J, roots[roots > 0]


def fixfun(beamformer: capon.RegularizedCapon, alpha, eps=0):
    spec, norm_w, gamma = np.array(list(map(beamformer._iter_alpha, alpha))).T
    N = beamformer.N
    M = beamformer.M

    gamma += eps
    norm_w += eps

    v_e = spec / (N - gamma)
    v_w = (norm_w - 1/M) / gamma

    alpha_hat = v_e / (N * v_w)

    return alpha_hat


def fixfun_v(beamformer: capon.RegularizedCapon, v):
    N = beamformer.N
    M = beamformer.M

    v_e, v_w = v
    alpha = v_e / (N * v_w)

    spec, norm_w, gamma = np.array(list(map(beamformer._iter_alpha, alpha))).T

    v_e = spec / (N - gamma)
    v_w = (norm_w - 1/M) / gamma

    return v_e, v_w

v = np.logspace(-5, 5, 11)




# spec, norm_w, gamma = beamformer._iter_alpha(0)
# fig, ax = plt.subplots()
# ax.plot(a_opt)
# ax.set_yscale('log')

# S1 = la.inv(np.eye(Ns.max()) - x.conj().T @ la.inv(x @ x.T.conj()s) @ x)


alpha0 = np.logspace(-5, 5, 11)
alpha_fix = np.zeros(alpha0.size)
fig, ax = plt.subplots()
for i, a0 in enumerate(alpha0):
    debug = beamformer.alpha_debug(alpha0=a0, num_iters=1000, eps=0)
    ax.plot(debug['barber']['alpha'], label=f'$\\alpha_0 = {a0:.2e}$')
    # ax.plot(debug['barber']['v_w'], label=f'$\\alpha_0 = {a0:.2e}$')
    # ax.plot(debug['mackay']['alpha'], label=f'$\\alpha_0 = {a0:.2e}$')
    alpha_fix[i] = debug['barber']['alpha'][-1]
ax.set_yscale('log')
ax.legend()

fig, ax = plt.subplots()
for i, a0 in enumerate(alpha0):
    debug = beamformer.alpha_debug(alpha0=a0, num_iters=1000, eps=0)
    # ax.plot(debug['barber']['alpha'], label=f'$\\alpha_0 = {a0:.2e}$')
    # ax.plot(debug['barber']['v_w'], label=f'$\\alpha_0 = {a0:.2e}$')
    ax.plot(debug['mackay']['alpha'], label=f'$\\alpha_0 = {a0:.2e}$')
    # alpha_fix[i] = debug['barber']['alpha'][-1]
ax.set_yscale('log')
ax.legend()


import scipy.optimize as opt
def fun(alpha):
    spec, norm_w, gamma = beamformer._iter_alpha(alpha)
    N = beamformer.N
    M = beamformer.M

    v_e = spec / (N - gamma)
    v_w = (norm_w - 1/M) / gamma

    alpha_hat = v_e / (N * v_w)

    return alpha_hat
fun2 = lambda alpha: np.abs(fun(alpha) - alpha)
res = opt.minimize_scalar(fun2)

a = np.logspace(-10, 20, 10000)
fa = fixfun(beamformer, a)
fig, ax = plt.subplots()
ax.plot(a, fa, label='$f(\\alpha)$')
ax.plot(a, fa - a, label='$f(\\alpha) - \\alpha$')
ax.plot(a, np.gradient(fa, a), label="$f'(\\alpha)$")
ax.plot(a, a, label='$\\alpha$')
ax.axvline(a_opt, color='k', linestyle='--', label='Optimal')
ax.axvline(alpha_fix[0], color='k', linestyle='-', label='Fixed point 1')
ax.axvline(alpha_fix[-1], color='k', linestyle='-.', label='Fixed point 2')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

A, B, roots = solve_poly(beamformer)

A.eval([1,2])

a = np.logspace(-10, 20, 10000)
val = dlog_loss(beamformer, a)
fig, ax = plt.subplots()
ax.plot(a, val, label='$L(\\alpha)$')
for i in range(roots.size):
    ax.axvline(roots[i], color='k', linestyle='--', label=f'Root {i+1}')
ax.set_xscale('log')
ax.grid()
ax.set_yscale('symlog')

# R = beamformer.R
# alpha = 1
# w = beamformer.w_hat(alpha)
# a_t

vals = np.array(list(map(beamformer._iter_alpha, a)))
spec, norm_w, gamma = vals.T
norm_w -= 1 / beamformer.M
fig, ax = plt.subplots()
ax.plot(a, spec, label='spec')
ax.plot(a, norm_w, label='norm w')
ax.plot(a, gamma, label='gamma')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

options = dict(num_iters=1000, alpha0=a_opt)
# options = dict(num_iters=1000, alpha0=0)
eps = 0
debug = beamformer.alpha_debug(eps=eps, **options)

fig, ax = plt.subplots()
# ax.plot(debug['mackay']['alpha'], label=f'Mackay, $\\epsilon={eps:.0e}$')
# ax.plot(debug['barber']['alpha'], label='Barber')
# ax.axhline(a_opt, color='k', linestyle='--', label='Optimal')
# ax.plot(debug['mackay']['v_e'], label=f'Mackay, $\\epsilon={eps:.0e}$')
ax.plot(debug['barber']['v_e'], label='Barber')
ax.set_yscale('log')
ax.set_title(f'$N = {N}$')
ax.legend()



print()

