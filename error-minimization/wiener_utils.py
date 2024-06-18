import numpy as np
import numpy.random as rnd
import numpy.linalg as la
from scipy import io
import scipy.optimize as sci_opt
import matplotlib.pyplot as plt

import rir_generator as rir

import scipy.signal as sig
import librosa as rosa
import tqdm
import warnings


def get_signal(N, amp=1, alpha=0.9):
    b = [1, 0]
    a = [1, -alpha]
    x = amp * rnd.randn(N)
    y = sig.lfilter(b, a, x)
    return y


def get_noise(SNR, x, seed=None):
    """
    Get noise with a desired SNR (dB) relative to signal x.
    """

    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd

    P_signal = np.var(x)
    P_noise = P_signal / 10 ** (SNR / 10)
    noise = np.sqrt(P_noise) * rng.normal(size=x.size)
    return noise


def get_frames(x, L):
    N = x.size
    if N < L:
        xx = np.pad(x, (0, L-N), mode='constant')
    else:
        xx = x.copy()

    x_pad = np.pad(xx, (L-1, 0), mode='constant')
    X = rosa.util.frame(x_pad, frame_length=L, hop_length=1)[::-1, :]

    return X


def generate_signals(w, L, N, SNR, alpha=0.9, seed=None, return_noise=False):
    x = get_signal(N, alpha=alpha)
    s = sig.convolve(w, x, mode='full')[:N]

    e = get_noise(SNR, s, seed=seed)
    d = s + e

    X = get_frames(x, L)[:,:N]
    assert X.shape == (L, N)

    if return_noise:
        return X, d, e
    else:
        return X, d


def load_filter(L=600, rir_options=None):
    if rir_options is None:
        rir_options = dict(
            c=340,                    # Sound velocity (m/s)
            fs=8e3,                   # Sample frequency (samples/s)
            r=[2, 1.5, 1],            # Receiver position(s) [x y z] (m)
            s=[2, 3.5, 2],            # Source position [x y z] (m)
            L=[5, 4, 6],              # Room dimensions [x y z] (m)
            reverberation_time=0.225,  # Reverberation time (s)
            # nsample=L,                # Number of output samples
        )
    h = rir.generate(nsample=L, **rir_options).ravel()
    return h


class Wiener():
    """Some Information about Wiener"""

    def __repr__(self):
        return f'Wiener(N={self.N}, L={self.M})'

    def __init__(self, X, d, w_true=None):
        super(Wiener, self).__init__()

        self.N = d.size
        self.M = X.shape[0]

        self.X = X.copy()
        self.d = d.copy()

        self.Rx = (1 / self.N) * (self.X @ self.X.T)
        self.rxd = (1 / self.N) * (self.X @ self.d)

        self.lamb, self.Q = la.eigh(self.Rx)
        self.zxd = self.Q.T @ self.rxd
        self.zxd2 = self.zxd ** 2
        self.v_d = la.norm(self.d) ** 2 / self.N

        if w_true is not None:
            self.w_true = w_true.copy()
            self.norm_star = la.norm(self.w_true) ** 2

    def w_hat(self, alpha):
        if isinstance(alpha, np.ndarray):
            w = self.Q @ (self.zxd[:, None] / (self.lamb[:, None] + alpha[None, :]))
        else:
            if alpha == 0:
                w = la.lstsq(self.Rx, self.rxd, rcond=None)[0]
            w = self.Q @ (self.zxd / (self.lamb + alpha))
        return w

    def misalignment(self, alpha):
        w_hat = self.w_hat(alpha)
        return la.norm(w_hat - self.w_true) ** 2 / self.norm_star

    def _alpha_iter(self, alpha):
        w = self.w_hat(alpha)
        norm_w = la.norm(w) ** 2

        # mse = la.norm(self.d - self.X.T @ w) ** 2 / self.N
        lamb_alpha = 1 / (self.lamb + alpha)
        # mse = self.v_d - alpha * norm_w - (self.zxd2 * lamb_alpha).sum()
        mse = self.v_d - alpha * norm_w - self.zxd2 @ lamb_alpha

        # gamma = (self.lamb * lamb_alpha).sum()
        gamma = self.lamb @ lamb_alpha
        return mse, norm_w, gamma

    def alpha_fixpoint(self, num_iters=5, alpha0=1):

        alpha = np.zeros(num_iters+1)
        alpha[0] = alpha0

        for i in range(num_iters):

            val2 = self.zxd2 @ (1 / (self.lamb + alpha[i]) ** 2)
            val1 = self.v_d - self.zxd2 @ (1 / (self.lamb + alpha[i]))
            val3 = (self.lamb / (self.lamb + alpha[i])).sum()
            val = (val1 / val2) / self.N * val3

            alpha[i+1] = val

        try:
            assert alpha[-1] > 0
        except AssertionError:
            warnings.warn(f'Warning: Negative alpha ({alpha[-1]})')

        return alpha

    def alpha_mackay(self, num_iters=5, alpha0=1):

        alpha = np.zeros(num_iters+1)
        v_e = np.ones(num_iters+1)
        v_w = np.ones(num_iters+1)
        alpha[0] = alpha0

        for i in range(num_iters):
            mse, norm_w, gamma = self._alpha_iter(alpha[i])

            v_e[i+1] = self.N * mse / (self.N - gamma)
            v_w[i+1] = norm_w / gamma

            alpha[i+1] = v_e[i+1] / (v_w[i+1] * self.N)

        try:
            assert alpha[-1] > 0
        except AssertionError:
            warnings.warn(f'Warning: Negative alpha ({alpha[-1]})')

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
            mse, norm_w, gamma = self._alpha_iter(alpha[i])

            v_e[i+1] = mse + gamma * v_e[i]
            v_w[i+1] = norm_w + (self.M - gamma) * v_w[i]

            alpha[i+1] = v_e[i+1] / (v_w[i+1] * self.N)


        return alpha, v_e, v_w

    def alpha_ledoit(self):
        """
        Ledoit-Wolf shrinkage
        """
        # from sklearn.covariance import LedoitWolf
        # ledoit = LedoitWolf()
        # ledoit.fit(self.X.T)

        # # (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
        # mu = np.trace(self.Rx) / self.M
        # nu = ledoit.shrinkage_
        # beta1 = 1 - nu
        # eta = nu * mu
        # alpha = eta / beta1

        rho = np.sum(la.norm(self.X, axis=0) ** 4) / self.N**2 - la.norm(self.Rx, 'fro')**2 / self.N
        nu = np.trace(self.Rx) / self.M

        alpha = np.minimum(nu * rho / la.norm(self.Rx - nu * np.eye(self.M), 'fro')**2, nu)
        beta = 1 - alpha / nu

        try:
            assert alpha > 0
            assert beta > 0
        except AssertionError:
            print('alpha =', alpha)
            print('alpha =', beta)

        return alpha / beta


    def alpha_hkb(self):
        """
        Compute the optimal alpha using Hoerl, Kennard, and Baldwin's method.
        Requires N >= M.
        """
        w = self.w_hat(0)
        # w = la.lstsq(self.Rx, self.rxd, rcond=None)[0]
        v_e = la.norm(self.d - self.X.T @ w) ** 2 / self.N
        v_w = la.norm(w) ** 2 / self.M

        alpha = v_e / (self.N * v_w)

        if self.M > self.N:
            alpha = 0

        # try:
        #     assert alpha > 0
        # except AssertionError:
        #     print('alpha =', alpha)
        #     # raise

        return alpha

    def alpha_grid(self):
        alpha0 = self.alpha_fixpoint(num_iters=20, alpha0=1)[-1]
        res = sci_opt.minimize_scalar(self.misalignment, bracket=(alpha0 * 1e-5, alpha0 * 1e5), method='brent')

        return res.x

    def best_alpha(self, mode='mackay', num_iters=5, alpha0=None):
        if mode == 'mackay':
            alpha = self.alpha_mackay(num_iters=num_iters, alpha0=alpha0)[0][-1]
        elif mode == 'fixpoint':
            alpha = self.alpha_fixpoint(num_iters=num_iters, alpha0=alpha0)[-1]
        elif mode == 'barber':
            alpha = self.alpha_barber(num_iters=num_iters, alpha0=alpha0)[0][-1]
        elif mode == 'ledoit':
            alpha = self.alpha_ledoit()
        elif mode == 'hkb':
            alpha = self.alpha_hkb()
        elif mode == 'grid':
            alpha = self.alpha_grid()
        return alpha

    def likelihood(self, alpha):
        if isinstance(alpha, np.ndarray):
            w = self.w_hat(alpha)
            rxdw = w.T @ self.rxd.conj()
            L = np.log(1 + self.lamb[None, :] / alpha[:, None]).sum(axis=-1) + self.N * np.log(self.v_d - rxdw)
        else:
            w = self.w_hat(alpha)
            L = np.log(1 + self.lamb / alpha).sum() + self.N * np.log(self.v_d - self.rxd.conj() @ w)

        return L

    def d_likelihood(self, alpha):
        if isinstance(alpha, np.ndarray):
            w = self.w_hat(alpha)
            norm_w = la.norm(w, axis=0) ** 2
            gamma = (self.lamb[None, :] / (self.lamb[None, :] + alpha[:, None])).sum(axis=-1)
            mse = la.norm(self.d[:, None] - self.X.conj().T @ w, axis=0) ** 2 / self.N

            # dL = self.N * norm_w / (self.v_d - rxdw) - gamma / alpha
            # rxdw = w.T @ self.rxd.conj()
            # g = self.v_d - rxdw

            f = norm_w
            g = mse + alpha * norm_w
            dL = self.N * f / g - gamma / alpha
        else:
            w = self.w_hat(alpha)
            norm_w = la.norm(w) ** 2
            gamma = (self.lamb / (self.lamb + alpha)).sum()
            mse = la.norm(self.d - self.X.conj().T @ w, axis=0) ** 2 / self.N
            # dL = self.N * norm_w / (self.v_d - self.rxd.conj() @ w) - gamma / alpha

            f = norm_w
            g = mse + alpha * norm_w
            dL = self.N * f / g - gamma / alpha

        return dL

    def roots(self, alpha0=1):
        # alpha0 = self.alpha_fixpoint(num_iters=20, alpha0=1)[-1]
        val = sci_opt.fsolve(self.d_likelihood, x0=alpha0)

        return val


