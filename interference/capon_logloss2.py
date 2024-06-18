
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
import scipy.signal as sci_sig
import seaborn as sns

import rir_generator as rir

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

# import plot_utils as putils
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


def get_sterring_vector(angles, M, noise=0, noise_type='angle', seed=None):
    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd

    # noise = noise * rng.uniform(low=-1, high=1, size=M)
    if noise_type == 'angle':
        phi_l = np.outer(np.cos(angles + noise), np.arange(M))
        ak = np.exp(1j * np.pi * phi_l)
    elif noise_type == 'additive':
        noise = rng.normal(scale=np.sqrt(noise / 2), size=(angles.size, M)) \
         + 1j * rng.normal(scale=np.sqrt(noise / 2), size=(angles.size, M))
        phi_l = np.outer(np.cos(angles), np.arange(M))
        ak = np.exp(1j * np.pi * phi_l) + noise
    elif noise_type == 'phase':
        delta = rng.normal(scale=noise, size=M)
        phi_l = np.outer(np.cos(angles), np.arange(M)) + delta[None, :]
        ak = np.exp(1j * np.pi * phi_l)

    return ak


# Number of antennas
M = 10

# Max number of samples
N = 100

# Index of SOI
soi_index = 0

# Angles
phi = np.pi * np.array([0.2, 0.4, 0.7])
# phi = np.array([0, 20, 60]) * np.pi / 180
# phi = np.array([0, 90, 180]) * np.pi / 180
# phi = np.pi * np.array([0.2, 0.3, 0.5, 0.6, 0.8])
K = phi.size

# Signal powers
Pk = 10.0 ** (np.array([30, 20, 10]) / 10)
# Pk = 10.0 ** (np.array([10, 20, 20]) / 10)    
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

# Angles
# phi = np.pi * np.array([0.2])
phi = np.pi * np.array([0.2, 0.3, 0.6])
# phi = np.pi * np.array([0.22, 0.31, 0.63])
# phi = np.pi * np.array([0.2, 0.1, 0.3, 0.4, 0.6, 0.8])

# Signal powers
# Pk = 10.0 ** (np.array([20]) / 10)
Pk = 10.0 ** (np.array([20, 10, 5]) / 10)
# Pk = 10.0 ** (np.array([15, 10, 5]) / 10)
# Pk = 10.0 ** (np.array([10, 10, 5]) / 10)
# Pk = 10.0 ** (np.array([20, 10, 10, 10, 10, 10]) / 10)

K = phi.size
assert K == Pk.size

########################### DEBUG ###########################

seed = 0
# seed = 10
rng = rnd.default_rng(seed)

# Create steering vector
a_t = get_sterring_vector(phi, M, noise=0)

# Create signals
s = get_signals(N, K, Pk=Pk, seed=rng)

# Create noise
z = rng.normal(scale=np.sqrt(0.5), size=(M, N)) \
+ 1j * rng.normal(scale=np.sqrt(0.5), size=(M, N))

# Create received signal
x = a_t.T @ s + z

# Get analytical covariance matrix
Rtrue = oe.contract('...,...m,...n->mn', Pk, a_t, a_t.conj()) + np.eye(M)
# Rtrue = Pk[0] * np.outer(a_t, a_t.conj()) + np.eye(M)

R = (1 / N) * (x @ x.conj().T)

# Input SINR
SINRin = M * Pk / (np.trace(Rtrue).real - M * Pk)

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
a_t = get_sterring_vector(phi, M, noise=0)[soi_index]
beamformer = capon.RegularizedCapon(a_t, xx, target=target)


# Target signal information
oracle = {
    'v_s': Pk[soi_index],
    'R': Rtrue
}

wiener = capon.Wiener(a_t, xx, oracle=oracle)
# w_hat = wiener.w_hat(1)
# sinr = wiener.sinr(1)

# a = np.logspace(-20, 20, 1000)
# u = wiener.u_hat(a)
# w = wiener.w_hat(a)
# sinr = wiener.sinr(a)

# a_opt = wiener.alpha_grid()
# res = sci_opt.minimize_scalar(lambda x: -wiener.sinr(x))




def logloss(wiener: capon.Wiener, alpha):
    N = wiener.N
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    val = np.log(1 + lamb/alpha).sum() + N * np.log(v_d - zxd2 @ (1 / (lamb + alpha)))
    return val


def dlogloss(wiener: capon.Wiener, alpha):
    # N = wiener.N
    # mse, norm_w, gamma = wiener._alpha_iter(alpha)
    # val = N * norm_w / (mse + alpha * norm_w) - gamma / alpha

    N = wiener.N
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    val1 = (lamb / (lamb + alpha)).sum()
    val2 = zxd2 @ (1 / (lamb + alpha) ** 2)
    val3 = v_d - zxd2 @ (1 / (lamb + alpha))
    val = -val1 / alpha + N * val2 / val3

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

    A = np.sum(lamb * Dm)
    B = np.sum(zxd2 * Dm ** 2)
    C = np.sum(zxd2 * Dm)

    L = A * (v_d * D - C) - N * B * alpha
    Lfull = -A / (alpha * D) + N * B / (D * (v_d - C*D))

    roots = []
    for root in L.real_roots():
        roots.append(root.evalf())
    roots = np.array(roots).astype(float)

    return L, Lfull, roots[roots > 0]


def polyprod(lamb):
    c = np.zeros([2, lamb.size])
    c[0] = 1
    c[1] = lamb

    poly = c[:, 0]
    for i in range(1, lamb.size):
        poly = np.polymul(poly, c[:, i])

    # C = fft.rfft(c, n=2 * lamb.size, axis=0)
    # poly = fft.irfft(np.prod(C, axis=-1))[:lamb.size+1]
    # C = fft.fft(c, n=2 * lamb.size, axis=0)
    # poly = fft.ifft(np.prod(C, axis=-1))[:lamb.size+1].real
    return poly

def polymul(a, b):
    return np.polymul(a, b)

def root_finder_fft(wiener: capon.Wiener):
    N = wiener.N
    M = wiener.M - 1
    v_d = wiener.v_d
    lamb = wiener.lamb[1:]
    zxd2 = wiener.zxd2[1:]

    D = polyprod(lamb)

    Dm = np.zeros([M, M])
    Dm2 = np.zeros([M, 2*M - 1])
    for m in range(M):
        Dm[m] = polyprod(lamb[np.arange(M) != m])
        Dm2[m] = polymul(Dm[m], Dm[m])

    A = np.sum(lamb[:, None] * Dm, axis=0)
    B = np.sum(zxd2[:, None] * Dm2, axis=0)
    C = np.sum(zxd2[:, None] * Dm, axis=0)

    # #### Debug ####
    # alpha = sp.symbols('alpha')
    # Dpoly = sp.prod(lamb + alpha)

    # Dmpoly = [None] * M
    # Dmpoly2 = [None] * M
    # for m in range(M):
    #     Dmpoly[m] = sp.poly(Dpoly / (lamb[m] + alpha))
    #     Dmpoly2[m] = Dmpoly[m] ** 2
    # Dmpoly = np.array(Dmpoly)
    # Dmpoly2 = np.array(Dmpoly2)

    # Apoly = np.sum(lamb * Dmpoly)
    # Bpoly = np.sum(zxd2 * Dmpoly2)
    # Cpoly = np.sum(zxd2 * Dmpoly)

    # vDCpoly = v_d * Dpoly - Cpoly
    # AvDCpoly = Apoly * vDCpoly
    # NaBDpoly = N * alpha * Bpoly * Dpoly
    # Lpoly = AvDCpoly - NaBDpoly
    # #### Debug ####

    # v * D(alpha) - C(alpha)
    vDC = v_d * D - np.hstack([np.zeros(D.size - C.size), C])

    # A(alpha) [v * D(alpha) - C(alpha)]
    AvDC = polymul(A, vDC)

    # N * alpha * B(alpha)
    NaB = np.hstack([N * B, 0])

    # L = np.hstack([np.zeros(NaB.size - AvDC.size), AvDC]) + NaB

    # -A(alpha) [v * D(alpha) - C(alpha)] + N * alpha * B(alpha)
    L = -AvDC + NaB

    # alpha * D(alpha) * [v * D(alpha) - C(alpha)]
    J = np.hstack([polymul(D, vDC), 0])

    roots = []
    for root in sp.Poly(L, sp.symbols('x')).real_roots():
        val = root.evalf()
        if val > 0:
            print(val)
            roots.append(val)
    roots = np.array(roots).astype(float)

    return L, J, roots[roots > 0]


def fg_gamma(wiener: capon.Wiener, alpha):
    N = wiener.N
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    gamma = np.sum(lamb / (lamb + alpha[:, None]), axis=-1)
    f = np.sum(zxd2 / (lamb + alpha[:, None]) ** 2, axis=-1)
    g = v_d - np.sum(zxd2 / (lamb + alpha[:, None]), axis=-1)

    return f, g, gamma


a = np.logspace(-20, 20, 1000)

v_d = wiener.v_d
lamb = wiener.lamb[1:]
norm_rxd = la.norm(wiener.rxd) ** 2
trR = lamb.sum()
lamb_min = lamb.min()
# lamb_min = lamb[1:].min()
lamb_max = lamb.max()
N = wiener.N

A = v_d - N * norm_rxd / trR
B = 2 * v_d * lamb_min - norm_rxd * (1 + N * lamb_max / trR)
C = lamb_min * (v_d * lamb_min - norm_rxd)

sol = np.roots([A, B, C])
sol = sol[sol > 0]
if sol.size == 1:
    sol = sol[0]

cond = N * norm_rxd > v_d * lamb.sum()
f, g, gamma = fg_gamma(wiener, a)

gamma_min = trR / (lamb_max + a)
f_max = norm_rxd / (lamb_min + a) ** 2
g_min = v_d - norm_rxd / (lamb_min + a)


fig, ax = plt.subplots()
ax.plot(a, f, label=r'$f(\alpha)$')
ax.plot(a, g, label=r'$g(\alpha)$')
ax.plot(a, f_max, label=r'$f_{\max}(\alpha)$')
ax.plot(a, g_min, label=r'$g_{\min}(\alpha)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()


fig, ax = plt.subplots()
ax.plot(a, np.abs(N*f_max/g_min), label=r'$N f_{\max}(\alpha) / g_{\min}(\alpha)$')
ax.plot(a, gamma_min/a, label=r'$\gamma_{\min}(\alpha) / \alpha$')
# ax.axvline(norm_rxd / v_d - lamb_min, color='black')
# ax.axvline(sol, color='black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()



gamma_sol = np.sum(lamb / (lamb + sol))
dgamma_sol = -np.sum(lamb / (lamb + sol) ** 2)
dgamma_a = dgamma_sol / sol - gamma_sol / sol ** 2
gamma_line = gamma_sol / sol + dgamma_a * (a - sol)

fig, ax = plt.subplots()
ax.plot(a, N*f/g, label=r'$N f(\alpha) / g(\alpha)$')
ax.plot(a, gamma/a, label=r'$\gamma(\alpha) / \alpha$')
ax.plot(a, gamma/a - N*f/g, label=r"$L'(\alpha)$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()

fig, ax = plt.subplots()
ax.plot(a, N*f/g, label=r'$N f(\alpha) / g(\alpha)$')
ax.plot(a, gamma/a, label=r'$\gamma(\alpha) / \alpha$')
ax.plot(a, N*f_max/g_min, label=r'$N f_{\max}(\alpha) / g_{\min}(\alpha)$')
ax.plot(a, gamma_min/a, label=r'$\gamma_{\min}(\alpha) / \alpha$')
# ax.plot(a, A*a**2 + B*a + C, label='quadratic')
ax.axvline(sol, color='black', label='quadratic solution')
ax.plot(a, gamma_line, label='tangent line')
# ax.plot(a, -N*f/g + gamma/a, label="$L'(\\alpha)$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()


fig, ax = plt.subplots()
ax.plot(a, g_min, label=r'$g_{\min}(\alpha)$')
ax.plot(a, g, label=r'$g(\alpha)$')
ax.plot(a, v_d - norm_rxd / (lamb_max + a), label=r'$g_{\max}(\alpha)$')
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()



fig, ax = plt.subplots()
ax.plot(a, A*a**2 + B*a + C)
ax.plot(a, wiener.sinr(a))
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()




alpha_logloss = sci_opt.minimize_scalar(lambda x: logloss(wiener, x)).x
alpha_sinr = sci_opt.minimize_scalar(lambda x: -wiener.sinr(x)).x



alpha0 = 1
res, _, _ = wiener.alpha_mackay(num_iters=100, alpha0=alpha0)
res2 = wiener.alpha_fixpoint(num_iters=100, alpha0=alpha0)
res3 = wiener.alpha_fixpoint(num_iters=100, alpha0=alpha0, eps=1e-6)

check = wiener.N * wiener.zxd2.sum() > wiener.v_d * wiener.lamb.sum()

fig, ax = plt.subplots()
ax.plot(res, label='Mackay')
ax.plot(res2, label='Proposed')
ax.plot(res3, label='Proposed 2')
ax.set_xlabel('iteration')
ax.set_yscale('log')
ax.grid()
ax.legend()






alpha0 = 1
res, _, _ = wiener.alpha_mackay(num_iters=100, alpha0=alpha0)
alpha_mackay = res[-1]

res2 = wiener.alpha_fixpoint(num_iters=100, alpha0=alpha0)
alpha_fixpoint = res2[-1]


a = np.logspace(-10, 60, 10000)

L = np.array(list(map(lambda x: logloss(wiener, x), a)))
Lpoly, Jpoly, roots = root_finder_fft(wiener)

fig, ax = plt.subplots()
ax.plot(a, L, label='$L(\\alpha)$')
ax.plot(a, wiener.sinr(a), label='$SINR(\\alpha)$')
# ax.plot(a, np.gradient(L, a))
for i in range(roots.size):
    ax.axvline(roots[i], color=color_list[i+2 % len(color_list)], linestyle='-', label=f'Root #{i+1}')
ax.axvline(alpha_fixpoint, color='black', linestyle='--', label='Fixpoint $\\alpha$')
ax.axvline(alpha_sinr, color='grey', linestyle='--', label='Oracle $\\alpha$')
ax.set_xscale('log')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$L(\\alpha)$')
ax.grid()
ax.legend()




z, p, k = sci_sig.tf2zpk(Lpoly, Jpoly)
fig, ax = plt.subplots()
ax.plot(z.real, z.imag, 'o', label='Zeros')
ax.plot(p.real, p.imag, 'x', label='Poles')
ax.axhline(0, color='black', linestyle='--')
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.axis([0, None, None, None])
ax.grid()
ax.legend()



a = np.logspace(-10, 60, 10000)
f, g, gamma = fg_gamma(wiener, a)
dL = -N*f/g + gamma/a

fig, ax = plt.subplots()
# ax.plot(a, N*f/g, label='$N f(\\alpha) / g(\\alpha)$')
ax.plot(a, f, label='$f(\\alpha)$')
ax.plot(a, g, label='$g(\\alpha)$')
# ax.plot(a, gamma/a, label='$\\gamma(\\alpha) / \\alpha$')
# ax.plot(a, -N*f/g + gamma/a, label="$L'(\\alpha)$")
ax.plot()
ax.set_xscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()


fig, ax = plt.subplots()
ax.plot(a, N*f/g + 1e-6, label='$N f(\\alpha) / g(\\alpha)$')
ax.plot(a, gamma/a, label='$\\gamma(\\alpha) / \\alpha$')
ax.plot(a, -N*f/g + gamma/a - 1e-6, label="$L'(\\alpha)$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()



la.norm(wiener.rxd)



U, S, Vh = la.svd(wiener.X, full_matrices=True)

d = wiener.d
c = Vh @ d
lamb = S ** 2 / N

M = wiener.M
N = wiener.N
rxd = wiener.rxd
zxd = wiener.zxd


np.sum(np.abs(c) ** 2 * lamb)
np.sum(np.abs(d)**2) * np.sum(lamb)




M = wiener.M
Rn = np.zeros([N, M, M], dtype=np.complex128)
for n in tqdm.trange(N):
    X = wiener.x[:, n]
    Rn[n] = (1 / (n+1)) * (X @ X.conj().T)

Rtrue = wiener.Rtrue
norm_true = la.norm(Rtrue)

error_R = la.norm(Rn - wiener.Rtrue, axis=(1,2)) / norm_true

fig, ax = plt.subplots()
ax.plot(error_R)
ax.grid()




w = wiener.w_hat(a)

v = wiener.v_s
# R = wiener.Rtrue

R = (1 / wiener.N) * (wiener.x @ wiener.x.conj().T)
# w0 = wiener.w_hat(alpha_fixpoint)
# v = oe.contract('...m,mn,...n->...', w0.conj(), R, w0).real

spec = oe.contract('...m,mn,...n->...', w.conj(), R, w).real
sinr_debug = spec / (spec - v)

fig, ax = plt.subplots()
ax.plot(a, wiener.sinr(a), label='$SINR(\\alpha)$')
ax.plot(a, sinr_debug, label='$SINR_d(\\alpha)$')
ax.set_xscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()

print()


