import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import opt_einsum as oe
from scipy import io
import scipy.linalg as sla
import rir_generator as rir
import scipy.optimize as sci_opt

import scipy.signal as sig
import librosa as rosa
import tqdm
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
import warnings


def generate_input(N, M, a, Pk, seed=None):
    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd

    K = Pk.size
    assert a.shape == (K, M)

    # Create signals
    s = np.empty((K, N), dtype=np.complex128)
    for i in range(K):
        s[i] = rng.normal(loc=0, scale=np.sqrt(Pk[i] / 2), size=N) \
        + 1j * rng.normal(loc=0, scale=np.sqrt(Pk[i] / 2), size=N)

    # Create noise
    z = rng.normal(scale=np.sqrt(0.5), size=(M, N)) \
    + 1j * rng.normal(scale=np.sqrt(0.5), size=(M, N))

    # Compute received signal
    x = a.T @ s + z

    return x


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



# class RegularizedCapon():
#     """Some Information about RegularizedCapon"""

#     def __repr__(self):
#         return f'RegularizedCapon(N={self.N}, M={self.M})'

#     def __init__(self, a, x, target=None):
#         super(RegularizedCapon, self).__init__()

#         self.M, self.N = x.shape
#         assert self.M == a.size
#         self.a = a
#         self.x = x

#         if target is None:
#             self.target = False
#         else:
#             self.target = True
#             self.a_t = target['a_t']
#             self.v_s = target['v_s'] ** 2
#             self.Rtrue = target['R']
#             assert self.a_t.size == self.M

#         self.R = (1 / self.N) * (self.x @ self.x.conj().T)
#         self.lamb, self.Q = la.eigh(self.R)

#         # b = Q^H a
#         self.b = self.Q.conj().T @ self.a
#         self.b2 = np.abs(self.b) ** 2

#     def w_hat(self, alpha):
#         if isinstance(alpha, (int, float)):
#             lamba = 1 / (self.lamb + alpha)
#             num = self.Q @ (lamba * self.b)
#             den = self.a.conj() @ num
#             w = num / den
#             return w

#         lamba = 1 / (self.lamb[None,] + alpha[:,None])
#         num = oe.contract(
#             self.Q, ['l', 'l*'],
#             lamba, ['a', 'l*'],
#             self.b, ['l*'],
#             ['a', 'l']
#         )
#         den = oe.contract(
#             self.a.conj(), ['l'],
#             num, ['a', 'l'],
#             ['a']
#         )
#         w = num / den[:, None]

#         return w

#     def spectrum(self, w):
#         if w.ndim == 1:
#             return (w.conj() @ self.R @ w).real
#         elif w.ndim > 1:
#             return oe.contract('...m,mn,...n->...', w.conj(), self.R, w).real

#     def _iter_alpha(self, alpha):
#         w = self.w_hat(alpha)
#         spec = self.spectrum(w)
#         norm_w = (w.conj() @ w).real
#         lamba = 1 / (self.lamb + alpha)
#         gamma = (self.lamb - spec * self.b2) @ lamba
#         return spec, norm_w, gamma.real

#     def alpha_mackay(self, num_iters=10, alpha0=1, eps=1e-6):

#         # R shape: (L, L)
#         # a shape: (L)

#         alpha = np.zeros(num_iters+1)
#         alpha[0] = alpha0

#         for i in range(num_iters):
#             spec, norm_w, gamma = self._iter_alpha(alpha[i])
#             # alpha[i+1] = spec / (norm_w - 1 / self.M + eps) * (gamma / (self.N - gamma + eps))
#             alpha[i+1] = spec / (norm_w - 1 / self.M + eps) / (self.N / gamma - 1)
#             alpha[i+1] = np.maximum(alpha[i+1], 0)

#         return alpha

#     def alpha_barber(self, num_iters=20, alpha0=1):

#         # R shape: (L, L)
#         # a shape: (L)

#         v_e = np.zeros(num_iters+1)
#         v_w = np.zeros(num_iters+1)
#         alpha = np.zeros(num_iters+1)

#         v_e[0], v_w[0] = (alpha0 * self.N, 1)
#         alpha[0] = v_e[0] / (self.N * v_w[0])

#         for i in range(num_iters):
#             spec, norm_w, gamma = self._iter_alpha(alpha[i])

#             v_e[i+1] = spec + v_e[i] * gamma / self.N
#             v_w[i+1] = (norm_w - 1/self.M + v_w[i] * (self.M - gamma)) / self.M
#             alpha[i+1] = v_e[i+1] / (self.N * v_w[i+1])
#             alpha[i+1] = np.maximum(alpha[i+1], 0)

#         return alpha

#     def alpha_debug(self, num_iters=10, alpha0=1, eps=0):
#         # R shape: (L, L)
#         # a shape: (L)

#         # Mackay
#         v_e = np.zeros(num_iters)
#         v_w = np.zeros(num_iters)
#         alpha = np.zeros(num_iters+1)
#         alpha[0] = alpha0
#         gamma_vec = np.zeros(num_iters)
#         spec_vec = np.zeros(num_iters)
#         norm_vec = np.zeros(num_iters)

#         for i in range(num_iters):
#             spec, norm_w, gamma = self._iter_alpha(alpha[i])

#             v_e[i] = spec / (self.N - gamma)
#             v_w[i] = (norm_w - 1/self.M) / gamma

#             # alpha[i+1] = spec / (norm_w - 1 / self.M + eps) / (self.N / gamma - 1 + eps)
#             alpha[i+1] = v_e[i] / (self.N * v_w[i])
#             gamma_vec[i] = gamma
#             spec_vec[i] = spec
#             norm_vec[i] = norm_w

#         mackay_dict = dict(
#             eps=eps,
#             alpha=alpha.copy(),
#             v_e=spec_vec.copy(),
#             v_w=norm_vec.copy(),
#             spec=spec_vec.copy(),
#             norm=norm_vec.copy(),
#             gamma=gamma_vec.copy()
#         )

#         # Barber
#         v_e = np.zeros(num_iters+1)
#         v_w = np.zeros(num_iters+1)
#         alpha = np.zeros(num_iters+1)
#         gamma_vec = np.zeros(num_iters)
#         spec_vec = np.zeros(num_iters)
#         norm_vec = np.zeros(num_iters)

#         v_e[0], v_w[0] = (alpha0 * self.N, 1)
#         alpha[0] = v_e[0] / (self.N * v_w[0])

#         for i in range(num_iters):
#             spec, norm_w, gamma = self._iter_alpha(alpha[i])

#             v_e[i+1] = spec + v_e[i] * gamma / self.N
#             # v_w[i+1] = (norm_w - 1/self.M + v_w[i] * (self.M - gamma)) / self.M
#             v_w[i+1] = (norm_w - 1/self.M) / self.M + v_w[i] * (1 - gamma / self.M)
#             alpha[i+1] = v_e[i+1] / (self.N * v_w[i+1])
#             gamma_vec[i] = gamma
#             spec_vec[i] = spec
#             norm_vec[i] = norm_w

#         barber_dict = dict(
#             alpha=alpha.copy(),
#             v_e=spec_vec.copy(),
#             v_w=norm_vec.copy(),
#             spec=spec_vec.copy(),
#             norm=norm_vec.copy(),
#             gamma=gamma_vec.copy()
#         )

#         debug_dict = dict(
#             mackay=mackay_dict,
#             barber=barber_dict
#         )

#         return debug_dict

#     def alpha_ledoit(self):
#         """
#         Ledoit-Wolf shrinkage
#         """

#         rho = np.sum(la.norm(self.x, axis=0) ** 4) / self.N**2 - la.norm(self.R, 'fro')**2 / self.N
#         nu = np.trace(self.R).real / self.M

#         alpha = np.minimum(nu * rho / la.norm(self.R - nu * np.eye(self.M), 'fro')**2, nu)
#         beta = 1 - alpha / nu

#         alpha = np.maximum(alpha, 0)
#         beta = np.maximum(beta, 0)

#         return alpha / beta

#     def alpha_hkb(self):
#         """
#         Compute the optimal alpha using Hoerl, Kennard, and Baldwin's method.
#         Requires N >= M.
#         """
#         w = self.w_hat(0)
#         # spec = self.spectrum(w)
#         spec = self.spectrum(w) / (self.N)
#         norm_w = (w.conj() @ w).real

#         alpha = (self.M - 1) * spec / (norm_w - 1/self.M)
#         alpha = np.maximum(alpha, 0)

#         return alpha

#     def alpha_lawless(self):
#         """
#         Compute the optimal alpha using Lawless and Wang's method.
#         Requires N >= M.
#         """
#         w = self.w_hat(0)
#         spec = self.spectrum(w) / (self.N)
#         u = self.a / self.M - w
#         norm_pred = (u.conj() @ self.R @ u).real

#         alpha = (self.M - 1) * spec / norm_pred
#         alpha = np.maximum(alpha, 0)

#         return alpha

#     def sinr(self, w):
#         if not self.target:
#             raise ValueError('No information given about the target signal')

#         # spec = (w.conj() @ self.Rtrue @ w).real
#         spec = oe.contract('...m,mn,...n->...', w.conj(), self.Rtrue, w).real
#         # w_a = self.v_s * np.abs(w @ self.a_t.conj()) ** 2
#         # w_a = self.v_s * np.abs(w @ self.a_t.conj()) ** 2

#         # return w_a / (spec - w_a)
#         # return spec / (spec - self.v_s)
#         return self.v_s / (spec - self.v_s)

#     def _alpha_grid(self, alpha):
#         w = self.w_hat(alpha)
#         alpha_opt = alpha[self.sinr(w).argmax()]
#         return alpha_opt

#     def alpha_grid(self):

#         import scipy.optimize as opt
#         fun = lambda alpha: -self.sinr(self.w_hat(alpha))
#         res = opt.minimize_scalar(fun)

#         return res.x

#     def alpha_opt(self, num_iters=100, alpha0=1, method='barber'):
#         if method == 'mackay':
#             alpha = self.alpha_mackay(num_iters=num_iters, alpha0=alpha0)[-1]
#         elif method == 'barber':
#             alpha = self.alpha_barber(num_iters=num_iters, alpha0=alpha0)[-1]
#         elif method == 'hkb':
#             alpha = self.alpha_hkb()
#         elif method == 'lawless':
#             alpha = self.alpha_lawless()
#         elif method == 'ledoit':
#             alpha = self.alpha_ledoit()
#         elif method == 'grid':
#             # alpha0 = np.log10(self.alpha_barber(num_iters=200, alpha0=alpha0)[-1].mean())
#             # alpha_vec = np.logspace(alpha0-1, alpha0+1, 1000)
#             # alpha_vec = np.logspace(alpha0-3, alpha0+3, 10000)

#             # alpha_vec = np.logspace(-3, 3, 10000)
#             # alpha = self.alpha_grid(alpha_vec)
#             # alpha = self.alpha_grid(alpha0=1, num_iters=100)[-1]
#             alpha = self.alpha_grid()
#         else:
#             raise ValueError(f'Method `{method}` not recognized.')

#         return alpha



class Wiener():
    """Some Information about Wiener"""

    def __repr__(self):
        return f'Wiener(N={self.N}, L={self.M})'

    def __init__(self, a, x, oracle=None):
        super(Wiener, self).__init__()

        self.M, self.N = x.shape
        assert self.M == a.size
        self.a = a
        self.x = x
        self.Rxx = (1 / self.N) * (self.x @ self.x.conj().T)

        if oracle is None:
            self.oracle = False
            self.v_s = None
            self.Rtrue = None
        else:
            self.oracle = oracle
            self.v_s = oracle['v_s']
            self.Rtrue = oracle['R']
            assert self.Rtrue.shape == (self.M, self.M)

        self.A = np.eye(self.M) - np.outer(self.a, self.a.conj()) / self.M

        self.X = self.A @ self.x
        self.d = 1/self.M * self.a @ self.x.conj()

        self.Rx = (1 / self.N) * (self.X @ self.X.conj().T)
        self.rxd = (1 / self.N) * (self.X @ self.d)

        self.lamb, self.Q = la.eigh(self.Rx)
        self.lamb[0] = 0
        self.zxd = self.Q.conj().T @ self.rxd
        self.zxd2 = np.abs(self.zxd) ** 2
        self.v_d = la.norm(self.d) ** 2 / self.N

    def u_hat(self, alpha):
        if isinstance(alpha, np.ndarray):
            # u = self.Q @ (self.zxd[:, None] / (self.lamb[:, None] + alpha[None, :]))
            zlamba = self.zxd / (self.lamb + alpha[:, None])
            u = oe.contract('mn,an->am', self.Q, zlamba)
        else:
            if alpha == 0:
                u = la.lstsq(self.Rx, self.rxd, rcond=None)[0]
            else:
                u = self.Q @ (self.zxd / (self.lamb + alpha))
        return u

    def w_hat(self, alpha):
        u = self.u_hat(alpha)

        if isinstance(alpha, np.ndarray):
            w = 1/self.M * self.a - u @ self.A.T
        else:
            w = 1/self.M * self.a - self.A @ u
        return w

    def sinr(self, alpha):
        w = self.w_hat(alpha)
        spec = oe.contract('...m,mn,...n->...', w.conj(), self.Rtrue, w).real
        return self.v_s / (spec - self.v_s)
        # return spec / (spec - self.v_s)

    def best_alpha(self, mode='mackay', num_iters=5, alpha0=None):
        if mode == 'fixpoint':
            alpha = self.alpha_fixpoint(num_iters=num_iters, alpha0=alpha0)[-1]
        elif mode == 'mackay':
            alpha = self.alpha_mackay(num_iters=num_iters, alpha0=alpha0)[0][-1]
        elif mode == 'barber':
            alpha = self.alpha_barber(num_iters=num_iters, alpha0=alpha0)[0][-1]
        elif mode == 'ledoit':
            alpha = self.alpha_ledoit()
        elif mode == 'hkb':
            alpha = self.alpha_hkb()
        elif mode == 'grid':
            alpha = self.alpha_grid()
        
        if mode == 'grid':
            S = -self.sinr(alpha)
            Sinf = -self.sinr(np.inf)

            if S > Sinf:
                alpha = np.inf

        if mode in ['mackay', 'barber', 'fixpoint']:
            # Check if the iterative solution is better than α = ∞
            L = self.likelihood(alpha)
            # Linf = self.likelihood(np.inf)
            Linf = self.N * np.log(self.v_d)

            # TODO: check nan and use inf
            if np.isnan(alpha) or (L > Linf):
                alpha = np.inf

        return alpha

    def _alpha_iter(self, alpha):
        u = self.u_hat(alpha)
        norm_u = la.norm(u) ** 2

        lamb_alpha = 1 / (self.lamb + alpha)
        # mse = la.norm(self.d - self.X.conj().T @ u) ** 2 / self.N
        # mse = self.v_d - alpha * norm_u - (self.zxd2 * lamb_alpha).sum()
        mse = self.v_d - alpha * norm_u - self.zxd2 @ lamb_alpha
        # mse = self.v_d - alpha * norm_u - (self.rxd.conj() @ u).real
        # mse = self.v_d + (u.conj() @ self.Rx @ u).real - 2*(self.rxd.conj() @ u).real

        # gamma = (self.lamb * lamb_alpha).sum()
        gamma = self.lamb @ lamb_alpha
        return mse, norm_u, gamma
    
    def _fixpoint_fun(self, alpha):
        f = self.zxd2 @ (1 / (self.lamb + alpha) ** 2)
        g = self.v_d - self.zxd2 @ (1 / (self.lamb + alpha))
        gamma = (self.lamb / (self.lamb + alpha)).sum()

    def alpha_fixpoint(self, num_iters=5, alpha0=1, eps=0):

        alpha = np.zeros(num_iters+1)
        alpha[0] = alpha0

        # # Check if there is at least one finite solution
        # if not (self.N * self.zxd2.sum() > self.v_d * self.lamb.sum()):
        #     warnings.warn('No finite solution')
        #     # alpha[:] = np.inf
        #     # return alpha

        for i in range(num_iters):

            f = self.zxd2 @ (1 / (self.lamb + alpha[i]) ** 2)
            g = self.v_d - self.zxd2 @ (1 / (self.lamb + alpha[i]))
            gamma = (self.lamb / (self.lamb + alpha[i])).sum()

            # alpha[i+1] = (g / f) / self.N * gamma
            # alpha[i+1] = gamma / (self.N * f / g + eps)

            alpha[i+1] = gamma / (self.N * f / g)

        try:
            assert alpha[-1] > 0
        except AssertionError:
            warnings.warn(f'Warning: Negative alpha ({alpha[-1]})')
            alpha = np.maximum(alpha, 0)

        return alpha

    def alpha_mackay(self, num_iters=5, alpha0=1):

        alpha = np.zeros(num_iters+1)
        v_e = np.ones(num_iters+1)
        v_w = np.ones(num_iters+1)
        alpha[0] = alpha0

        for i in range(num_iters):
            mse, norm_u, gamma = self._alpha_iter(alpha[i])

            v_e[i+1] = self.N * mse / (self.N - gamma)
            v_w[i+1] = norm_u / gamma

            alpha[i+1] = v_e[i+1] / (v_w[i+1] * self.N)

        try:
            assert alpha[-1] > 0
        except AssertionError:
            warnings.warn(f'Warning: Negative alpha ({alpha[-1]})')
            alpha = np.maximum(alpha, 0)

        return alpha, v_e, v_w

    def alpha_barber(self, num_iters=5, alpha0=None, v0=(1, 1)):

        v_e = np.zeros(num_iters+1)
        v_w = np.zeros(num_iters+1)
        alpha = np.zeros(num_iters+1)

        if alpha0 is None:
            v_e[0], v_w[0] = v0
        else:
            v_w[0] = 1
            v_e[0] = v_w[0] * alpha0 * self.N
        alpha[0] = v_e[0] / (v_w[0] * self.N)

        for i in range(num_iters):
            mse, norm_u, gamma = self._alpha_iter(alpha[i])

            v_e[i+1] = mse + gamma * v_e[i]
            v_w[i+1] = norm_u + (self.M - gamma) * v_w[i]

            alpha[i+1] = v_e[i+1] / (v_w[i+1] * self.N)

        return alpha, v_e, v_w

    def alpha_ledoit(self):
        """
        Ledoit-Wolf shrinkage
        """
        # from sklearn.covariance import LedoitWolf
        # ledoit = LedoitWolf()
        # ledoit.fit(self.X.conj().T)

        # # (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
        # mu = np.trace(self.Rx) / self.M
        # nu = ledoit.shrinkage_
        # beta = 1 - nu
        # eta = nu * mu
        # alpha = eta / beta

        rho = np.sum(la.norm(self.x, axis=0) ** 4) / self.N**2 - la.norm(self.Rxx, 'fro')**2 / self.N
        nu = np.trace(self.Rxx).real / self.M

        alpha = np.minimum(nu * rho / la.norm(self.Rxx - nu * np.eye(self.M), 'fro')**2, nu)
        beta = 1 - alpha / nu

        # try:
        #     assert alpha > 0
        #     assert beta > 0
        # except AssertionError:
        #     print('alpha =', alpha)
        #     print('beta =', beta)

        return alpha / beta


    def alpha_hkb(self):
        """
        Compute the optimal alpha using Hoerl, Kennard, and Baldwin's method.
        Requires N >= M.
        """
        u = self.u_hat(0)
        # u = la.lstsq(self.X.conj().T, self.d, rcond=None)[0]
        # u = la.lstsq(self.Rx, self.rxd, rcond=None)[0]

        mse = la.norm(self.d - self.X.conj().T @ u) ** 2 / self.N
        norm_u = la.norm(u) ** 2 / (self.M - 1)
        # mse, norm_u, gamma = self._alpha_iter(0)

        alpha = mse / (norm_u * self.N)

        # TODO: if N < M, alpha is 0
        if self.M > self.N:
            alpha = 0


        # try:
        #     assert alpha > 0
        # except AssertionError:
        #     print('alpha =', alpha)
        #     # raise

        return alpha

    def alpha_grid(self):
        # alpha0 = self.alpha_fixpoint(num_iters=20, alpha0=1)[-1]
        # options = dict(bracket=(alpha0 * 1e-5, alpha0 * 1e5), method='brent')
        # options = dict(bracket=(1e-3, 1e13), method='brent')
        # res = sci_opt.minimize_scalar(lambda x: -self.sinr(x), **options)
        # alpha = res.x
        options = dict(bracket=(-3, 13), method='brent')
        res = sci_opt.minimize_scalar(lambda x: -self.sinr(np.exp(x)), **options)
        alpha = np.exp(res.x)

        return alpha

    def likelihood(self, alpha):
        if isinstance(alpha, np.ndarray):
            w = self.u_hat(alpha)
            rxdw = (w @ self.rxd.conj()).real
            L = np.log(1 + self.lamb[None, :] / alpha[:, None]).sum(axis=-1) + self.N * np.log(self.v_d - rxdw)
        else:
            w = self.u_hat(alpha)
            L = np.log(1 + self.lamb / alpha).sum() + self.N * np.log(self.v_d - (self.rxd.conj() @ w).real)

        return L

    def d_likelihood(self, alpha):
        if isinstance(alpha, np.ndarray):
            w = self.u_hat(alpha)
            rxdw = (w @ self.rxd.conj()).real
            norm_w = la.norm(w, axis=-1) ** 2
            gamma = (self.lamb[None, :] / (self.lamb[None, :] + alpha[:, None])).sum(axis=-1)
            dL = self.N * norm_w / (self.v_d - rxdw) - gamma / alpha
        else:
            w = self.u_hat(alpha)
            norm_w = la.norm(w) ** 2
            gamma = (self.lamb / (self.lamb + alpha)).sum()
            dL = self.N * norm_w / (self.v_d - (self.rxd.conj() @ w).real) - gamma / alpha

        return dL
