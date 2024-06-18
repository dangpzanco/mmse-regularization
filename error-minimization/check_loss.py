
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import tqdm
import opt_einsum as oe
import matplotlib.pyplot as plt
import scipy.optimize as sci_opt
import scipy.signal as sci_sig

import rir_generator as rir

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import wiener_utils as utils
import sympy as sp
import numpy.fft as fft


# Set random seed
np.random.seed(0)

# Number of samples
N = 1000

# Number of parameters
M = 100

# Signal-to-noise ratio (dB)
SNR = 0

rir_options = dict(
    c=340,                    # Sound velocity (m/s)
    fs=8e3,                   # Sample frequency (samples/s)
    r=[2, 1.5, 1],            # Receiver position(s) [x y z] (m)
    s=[2, 3.5, 2],            # Source position [x y z] (m)
    L=[5, 4, 6],              # Room dimensions [x y z] (m)
    reverberation_time=0.225,   # Reverberation time (s)
)
w_true = rir.generate(nsample=M, **rir_options).ravel()
w_true = rnd.randn(M)
X, d = utils.generate_signals(w_true, M, N, SNR, alpha=0.9)

wiener = utils.Wiener(X, d, w_true=w_true)

# def logloss(wiener: utils.Wiener, alpha):
#     N = wiener.N
#     M = wiener.M

#     w_hat = wiener.w_hat(alpha)

#     Ra_inv = wiener.Q @ np.diag(1 / (wiener.lamb + alpha)) @ wiener.Q.T
#     # lambA = 1 / (wiener.lamb[:, None] + alpha[None, :])
#     # Ra_inv = oe.contract('ij,ja,jk->aik', wiener.Q, lambA, wiener.Q.T)

#     val = -la.slogdet(Ra_inv)[-1] - M * np.log(alpha) + N * np.log(wiener.v_d - w_hat.T @ wiener.rxd)
#     return val


def logloss(wiener: utils.Wiener, alpha):
    N = wiener.N
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    val = np.log(1 + lamb/alpha).sum() + N * np.log(v_d - zxd2 @ (1 / (lamb + alpha)))
    return val


def dlogloss(wiener: utils.Wiener, alpha):
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


def root_finder(wiener: utils.Wiener):
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


def polyprod_fft(lamb):
    c = np.zeros([2, lamb.size])
    c[0] = 1
    c[1] = lamb

    C = fft.fft(c, n=2 * lamb.size, axis=0)
    poly = fft.ifft(np.prod(C, axis=-1))[:lamb.size+1].real
    return poly


def polymul(a, b):
    return np.polymul(a, b)


def polymul_fft(a, b):
    L = a.size + b.size - 1
    A = fft.fft(a, L)
    B = fft.fft(b, L)
    return fft.ifft(A * B).real


def root_finder_fft(wiener: utils.Wiener):
    N = wiener.N
    M = wiener.M
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

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



a = np.logspace(-10, 10, 1000)

alpha_logloss = sci_opt.minimize_scalar(lambda x: logloss(wiener, x)).x
alpha_mis = sci_opt.minimize_scalar(lambda x: wiener.misalignment(x)).x

alpha0 = 1
res, _, _ = wiener.alpha_mackay(num_iters=100, alpha0=alpha0)
alpha_mackay = res[-1]

res2 = wiener.alpha_fixpoint(num_iters=100, alpha0=alpha0)
alpha_fixpoint = res2[-1]

fig, ax = plt.subplots()
ax.plot(res)
ax.plot(res2)

mse, norm_w, gamma = wiener._alpha_iter(1)

w_hat = wiener.w_hat(a)
mis = la.norm(w_hat.T - wiener.w_true, axis=-1) ** 2


L = np.array(list(map(lambda x: logloss(wiener, x), a)))
Lpoly, Jpoly, roots = root_finder_fft(wiener)

# ax.plot(a, mis)
# ax.axvline(alpha_mis, color='black', linestyle='--', label='Optimal $\\alpha$')

fig, ax = plt.subplots()
ax.plot(a, L, color='black')
# ax.plot(a, np.gradient(L, a))
for i in range(roots.size):
    ax.axvline(roots[i], color=color_list[i % len(color_list)], linestyle='-', label=f'Root #{i+1}')
ax.axvline(alpha_mackay, color='grey', linestyle='--', label='Mackay $\\alpha$')
ax.set_xscale('log')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$L(\\alpha)$')
ax.grid()
ax.legend()

fig, ax = plt.subplots()
ax.plot(a, L - L.min())
ax.plot(a, np.gradient(L, a))
ax.plot(a, np.gradient(np.gradient(L, a), a))
ax.set_xscale('log')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$L(\\alpha)$')
ax.grid()
ax.legend()


dL = np.array(list(map(lambda x: dlogloss(wiener, x), a)))
dLpoly = np.polyval(Lpoly, a) / np.polyval(Jpoly, a)

fig, ax = plt.subplots()
ax.plot(a, dL, label='Eigenvalues')
ax.plot(a, dLpoly, label='Poly')
# ax.plot(a, (dL - dLpoly) / dL, label='Poly')
ax.set_xscale('log')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel("$L'(\\alpha)$")
ax.grid()
ax.legend()




# def eval_sos(sos, alpha):
#     sos = sos.copy()
#     val = 1
#     for i in range(sos.shape[0]):
#         b = sos[i, :3][::-1]
#         a = sos[i, 3:][::-1]
#         val *= np.polyval(b, alpha) / np.polyval(a, alpha)
#     return val

# sos = sci_sig.tf2sos(Lpoly, Jpoly)
# dLsos = eval_sos(sos, a)

# fig, ax = plt.subplots()
# ax.plot(a, dL, label='Eigenvalues')
# ax.plot(a, dLpoly, label='Poly')
# ax.plot(a, dLsos, label='Biquad')
# # ax.plot(a, (dL - dLpoly) / dL, label='Poly')
# ax.set_xscale('log')
# ax.set_xlabel('$\\alpha$')
# ax.set_ylabel("$L'(\\alpha)$")
# ax.grid()
# ax.legend()


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



# L = logloss(wiener, a[0])

# # L = np.array(list(map(lambda x: logloss(wiener, x), a)))
# dL = np.array(list(map(lambda x: dlogloss(wiener, x), a)))

# fig, ax = plt.subplots()
# # ax.plot(a, mis)
# # ax.plot(a, L)
# ax.plot(a, dL)
# # ax.plot(a, np.gradient(L, a))
# ax.plot(a, np.gradient(mis, a))
# # ax.plot(a, dL / np.gradient(L, a))
# ax.set_xscale('log')
# ax.set_yscale('symlog')
# ax.grid()




def fg_gamma(wiener: utils.Wiener, alpha):
    N = wiener.N
    v_d = wiener.v_d
    lamb = wiener.lamb
    zxd2 = wiener.zxd2

    gamma = np.sum(lamb / (lamb + alpha[:, None]), axis=-1)
    f = np.sum(zxd2 / (lamb + alpha[:, None]) ** 2, axis=-1)
    g = v_d - np.sum(zxd2 / (lamb + alpha[:, None]), axis=-1)

    return f, g, gamma

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


a = np.logspace(-4, 4, 10000)
f, g, gamma = fg_gamma(wiener, a)
dL = -N*f/g + gamma/a

fig, ax = plt.subplots()
ax.plot(a, N*f/g, label='$N f(\\alpha) / g(\\alpha)$')
# ax.plot(a, f, label='$f(\\alpha)$')
# ax.plot(a, g, label='$g(\\alpha)$')
ax.plot(a, gamma/a, label='$\\gamma(\\alpha) / \\alpha$')
ax.plot(a, -N*f/g + gamma/a, label="$L'(\\alpha)$")
ax.set_xscale('log')
ax.set_xlabel('$\\alpha$')
ax.grid()
ax.legend()





print()
