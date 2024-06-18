import numpy as np
import numpy.random as rnd
import scipy.linalg as la
from scipy import io
import rir_generator as rir

import scipy.signal as sig
import librosa as rosa
import tqdm


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


def generate_signals(h_star, L, N, SNR, alpha=0.9, seed=None, return_noise=False):
    x = get_signal(N, alpha=alpha)
    s = sig.convolve(h_star, x, mode='full')[:N]

    e = get_noise(SNR, s, seed=seed)
    d = s + e

    X = get_frames(x, L)[:,:N]
    assert X.shape == (L, N)

    if return_noise:
        return X, d, e
    else:
        return X, d


def load_filter(mode='paper', L=600, rir_options=None):
    if mode in ['paper', 'small']:
        mat = io.loadmat('g168path.mat')
        h_small = mat['B1'].ravel()
        if mode == 'paper':
            h_star = np.hstack([np.zeros(80), h_small, np.zeros(L - 80 - h_small.size)])
        else:
            h_star = h_small
    elif mode == 'rir':
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
        h_star = rir.generate(nsample=L, **rir_options)
        h_star = h_star.ravel()
    return h_star


def run_experiment(h_star, L=None, Ns=None, SNR=None, num_samples=1, ar1=0.9):

    if Ns is None:
        Ns = np.logspace(np.log10(L), np.log10(L)+1, 50).astype(int)

    if SNR is None:
        SNR = np.array([0, 5, 10, 15, 20, 25])

    error_grid = dict(
        mis=np.zeros([Ns.size, SNR.size, num_samples]),
        valid=np.zeros([Ns.size, SNR.size, num_samples]),
    )
    error_bayes = dict(
        mis=np.zeros([Ns.size, SNR.size, num_samples]),
        valid=np.zeros([Ns.size, SNR.size, num_samples]),
    )

    best_alpha = dict(
        mackay=np.zeros([Ns.size, SNR.size, num_samples]),
        barber=np.zeros([Ns.size, SNR.size, num_samples]),
        mis=np.zeros([Ns.size, SNR.size, num_samples]),
        valid=np.zeros([Ns.size, SNR.size, num_samples]),
    )

    for n, N in enumerate(tqdm.tqdm(Ns)):
        for s in tqdm.trange(SNR.size, leave=False):
            for r in tqdm.trange(num_samples, leave=False):
                X_train, d_train = generate_signals(h_star, L, N, SNR[s], alpha=ar1)
                X_valid, d_valid = generate_signals(h_star, L, N, SNR[s], alpha=ar1)

                wiener = Wiener(X_train, d_train, h_star=h_star, X_valid=X_valid, d_valid=d_valid)
                options = dict(alpha0=0.5, num_iters=5)
                best_alpha['mackay'][n, s, r] = wiener.best_alpha(mode='mackay', **options)
                best_alpha['barber'][n, s, r] = wiener.best_alpha(mode='barber', **options)

                best_alpha['mis'][n, s, r] = wiener.best_alpha(mode='mis')
                best_alpha['valid'][n, s, r] = wiener.best_alpha(mode='valid')

                wiener.alpha = best_alpha['mackay'][n, s, r]
                error_bayes['mis'][n, s, r] = wiener.misalignment
                error_bayes['valid'][n, s, r] = wiener.mse_valid

                wiener.alpha = best_alpha['mis'][n, s, r]
                error_grid['mis'][n, s, r] = wiener.misalignment
                error_grid['valid'][n, s, r] = wiener.mse_valid


    for key in error_grid.keys():
        # error_grid[key] /= num_samples
        # error_bayes[key] /= num_samples
        error_grid[key] = 10 * np.log10(error_grid[key])
        error_bayes[key] = 10 * np.log10(error_bayes[key])

    return error_grid, error_bayes, best_alpha



def run_experiment2(h_star, L=None, Ns=None, SNR=None, num_samples=1, ar1=0.9):

    if Ns is None:
        Ns = np.logspace(np.log10(L), np.log10(L)+1, 50).astype(int)

    if SNR is None:
        SNR = np.array([0, 5, 10, 15, 20, 25])

    misalignment = dict(
        mackay=np.zeros([Ns.size, SNR.size, num_samples]),
        barber=np.zeros([Ns.size, SNR.size, num_samples]),
        ledoit=np.zeros([Ns.size, SNR.size, num_samples]),
        hkb=np.zeros([Ns.size, SNR.size, num_samples]),
        lawless=np.zeros([Ns.size, SNR.size, num_samples]),
        grid=np.zeros([Ns.size, SNR.size, num_samples]),
    )

    best_alpha = dict(
        mackay=np.zeros([Ns.size, SNR.size, num_samples]),
        barber=np.zeros([Ns.size, SNR.size, num_samples]),
        ledoit=np.zeros([Ns.size, SNR.size, num_samples]),
        hkb=np.zeros([Ns.size, SNR.size, num_samples]),
        lawless=np.zeros([Ns.size, SNR.size, num_samples]),
        grid=np.zeros([Ns.size, SNR.size, num_samples]),
    )

    for n, N in enumerate(tqdm.tqdm(Ns)):
        for s in tqdm.trange(SNR.size, leave=False):
            for r in tqdm.trange(num_samples, leave=False):
                X_train, d_train = generate_signals(h_star, L, N, SNR[s], alpha=ar1)
                X_valid, d_valid = generate_signals(h_star, L, N, SNR[s], alpha=ar1)

                wiener = Wiener(X_train, d_train, h_star=h_star, X_valid=X_valid, d_valid=d_valid)
                options = dict(alpha0=0.5, num_iters=5)
                for key in best_alpha.keys():
                    alpha = wiener.best_alpha(mode=key, **options)
                    best_alpha[key][n, s, r] = alpha
                    wiener.alpha = alpha
                    misalignment[key][n, s, r] = wiener.misalignment


    for key in misalignment.keys():
        misalignment[key] = 10 * np.log10(misalignment[key])

    return misalignment, best_alpha


class Wiener():
    """Some Information about Wiener"""

    def __repr__(self):
        return f'Wiener(N={self.N}, L={self.L})'

    def __init__(self, X_train, d_train, h_star=None, X_valid=None, d_valid=None, alpha=1e-5):
        super(Wiener, self).__init__()

        self.N = d_train.size
        self.L = X_train.shape[0]

        self._h_hat = None
        self.init_train(X_train, d_train)

        if h_star is not None:
            self.h_star = h_star.copy()
            self.norm_star = la.norm(self.h_star) ** 2

        if X_valid is not None:
            self.init_valid(X_valid, d_valid)

    def init_train(self, X_train, d_train):
        self.X_train = X_train.copy()
        self.d_train = d_train.copy()

        self.Rx = (1 / self.N) * (self.X_train @ self.X_train.T)
        self.rxd = (1 / self.N) * (self.X_train @ self.d_train)

        self.lamb, self.Q = la.eigh(self.Rx)
        self.zxd = self.Q.T @ self.rxd
        self.norm_d = la.norm(self.d_train) ** 2 / self.N

    def init_valid(self, X_valid, d_valid):
        self.X_valid = X_valid.copy()
        self.d_valid = d_valid.copy()
        self.Rx_valid = (1 / self.N) * (self.X_valid @ self.X_valid.T)
        self.rxd_valid = (1 / self.N) * (self.X_valid @ self.d_valid)
        self.lamb_valid, self.Q_valid = la.eigh(self.Rx_valid)
        assert self.Rx_valid.shape == self.Rx.shape

    @property
    def h_hat(self):
        return self.h_alpha(self.alpha)

    @property
    def misalignment(self):
        # return la.norm(self.h_hat - self.h_star[:self.L]) ** 2 / self.norm_star
        h_hat = np.pad(self.h_hat, (0, self.h_star.size-self.L), mode='constant')
        try:
            return la.norm(h_hat - self.h_star) ** 2 / self.norm_star
        except Exception as e:
            print(e)
            print(h_hat.shape, self.h_star.shape)
            raise


    @property
    def mse_valid(self):
        return la.norm(self.d_valid - self.X_valid.T @ self.h_hat) ** 2 / self.N

    def h_alpha(self, alpha):
        if isinstance(alpha, np.ndarray):
            value = self.Q @ (self.zxd[:, None] / (self.lamb[:, None] + alpha[None, :]))
        else:
            value = self.Q @ (self.zxd / (self.lamb + alpha))
        return value

    def misalignment_alpha(self, alpha):
        h_alpha = self.h_alpha(alpha)
        return la.norm(h_alpha - self.h_star[:self.L]) ** 2 / self.norm_star

    def mse_valid_alpha(self, alpha):
        h_alpha = self.h_alpha(alpha)
        return la.norm(self.d_valid - self.X_valid.T @ h_alpha) ** 2 / self.N

    def alpha_mackay(self, num_iters=5, alpha0=1, v0=None):

        alpha = np.zeros(num_iters+1)
        v_e = np.ones(num_iters+1)
        v_h = np.ones(num_iters+1)

        if v0 is None:
            alpha[0] = alpha0
        else:
            v_e[0], v_h[0] = v0
            alpha[0] = v_e[0] / (v_h[0] * self.N)

        for i in range(num_iters):
            
            h_alpha = self.h_alpha(alpha[i])
            mse = la.norm(self.d_train - self.X_train.T @ h_alpha) ** 2 / self.N
            norm_h = la.norm(h_alpha) ** 2
            lambA = self.L - alpha[i] * (1 / (self.lamb + alpha[i])).sum()

            v_e[i+1] = self.N * mse / (self.N - lambA)
            v_h[i+1] = norm_h / lambA

            alpha[i+1] = v_e[i+1] / (v_h[i+1] * self.N)

        try:
            assert alpha[-1] > 0
        except AssertionError:
            print('alpha[-1] =', alpha[-1])
            raise

        return alpha, v_e, v_h

    def alpha_barber(self, num_iters=5, v0=(1, 1), alpha0=None):

        v_e = np.zeros(num_iters+1)
        v_h = np.zeros(num_iters+1)

        if alpha0 is None:
            v_e[0], v_h[0] = v0
        else:
            v_h[0] = 1
            v_e[0] = v_h[0] * alpha0 * self.N

        zxd2 = self.zxd ** 2
        for i in range(num_iters):
            alpha = v_e[i] / (v_h[i] * self.N)

            lambA = 1 / (self.lamb + alpha)
            lambA2 = lambA ** 2

            trK = (v_e[i] / self.N) * lambA.sum()
            norm_h = np.sum(zxd2 * lambA2)
            mse = self.norm_d - np.sum(zxd2 * lambA2 * (self.lamb + 2*alpha))

            v_e[i+1] = mse + alpha * (self.L * v_h[i] - trK)
            v_h[i+1] = (norm_h + trK) / self.L
        alpha = v_e / (v_h * self.N)

        return alpha, v_e, v_h

    def alpha_ledoit(self):
        """
        Ledoit-Wolf shrinkage
        """
        from sklearn.covariance import LedoitWolf
        ledoit = LedoitWolf()
        ledoit.fit(self.X_train.T)

        # (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
        mu = np.trace(self.Rx) / self.L
        nu = ledoit.shrinkage_
        beta = 1 - nu
        eta = nu * mu
        alpha = eta / beta

        # rho = np.sum(la.norm(self.x, axis=0) ** 4) / self.N**2 - la.norm(self.Rx, 'fro')**2 / self.N
        # nu = np.trace(self.Rx) / self.L

        # alpha = np.minimum(nu * rho / la.norm(self.Rx - nu * np.eye(self.L), 'fro')**2, nu)
        # beta = 1 - alpha / nu

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
        w = self.h_alpha(0)
        mse = la.norm(self.d_train - self.X_train.T @ w) ** 2 / self.N
        norm_w = la.norm(w) ** 2 / self.L

        alpha = mse / norm_w

        try:
            assert alpha > 0
        except AssertionError:
            print('alpha =', alpha)
            # raise

        return alpha

    def alpha_lawless(self):
        """
        Compute the optimal alpha using Lawless and Wang's method.
        Requires N >= M.
        """
        w = self.h_alpha(0)
        mse = la.norm(self.d_train - self.X_train.T @ w) ** 2 / self.N
        norm_pred = la.norm(self.X_train.T @ w) ** 2 / self.L

        alpha = mse / norm_pred

        try:
            assert alpha > 0
        except AssertionError:
            print('alpha =', alpha)
            # raise

        return alpha

    def alpha_grid(self, mode='mis'):
        import scipy.optimize as opt
        if mode == 'mis':
            fun = self.misalignment_alpha
        elif mode == 'valid':
            fun = self.mse_valid_alpha
        res = opt.minimize_scalar(fun)

        return res.x

    def best_alpha(self, mode='mackay', num_iters=5, alpha0=None):
        if mode == 'mackay':
            alpha = self.alpha_mackay(num_iters=num_iters, alpha0=alpha0)[0][-1]
        elif mode == 'barber':
            alpha = self.alpha_barber(num_iters=num_iters, alpha0=alpha0)[0][-1]
        elif mode == 'ledoit':
            alpha = self.alpha_ledoit()
        elif mode == 'hkb':
            alpha = self.alpha_hkb()
        elif mode == 'lawless':
            alpha = self.alpha_lawless()
        elif mode == 'mis' or mode == 'grid':
            alpha = self.alpha_grid(mode='mis')
            # alpha0 = np.log10(self.alpha_mackay()[0][-1])
            # alpha_vec = np.logspace(alpha0-1, alpha0+1, 1000)
            # mis = self.misalignment_alpha(alpha_vec)
            # # mis = la.norm(self.h_alpha(alpha_vec) - self.h_star[:self.L, None], axis=0)
            # alpha = alpha_vec[np.argmin(mis)]
        elif mode == 'valid':
            alpha = self.alpha_grid(mode='valid')
            # alpha0 = np.log10(self.alpha_mackay()[0][-1])
            # alpha_vec = np.logspace(alpha0-1, alpha0+1, 1000)
            # mse = self.mse_valid_alpha(alpha_vec)
            # # mse = la.norm(self.d_valid[:, None] - self.X_valid.T @ self.h_alpha(alpha_vec), axis=0)
            # alpha = alpha_vec[np.argmin(mse)]
        return alpha


