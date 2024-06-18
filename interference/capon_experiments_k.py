
import numpy as np
import numpy.random as rnd
from scipy import io
import scipy.signal as sig
import scipy.linalg as la
import matplotlib.pyplot as plt
import opt_einsum as oe
import tqdm
import librosa

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
        noise=0.1, noise_type='additive', Pk=None, alpha=0, seed=None):

    if phi is None:
        phi = np.array([0.2, 0.4, 0.7]) * np.pi

    if Pk is None:
        Pk = 10.0 ** (np.array([10, 20, 30]) / 10)

    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd

    K = phi.size

    SINR = dict(
        fixpoint=np.zeros((K, num_samples, Ns.size)),
        mackay=np.zeros((K, num_samples, Ns.size)),
        barber=np.zeros((K, num_samples, Ns.size)),
        grid=np.zeros((K, num_samples, Ns.size)),
        hkb=np.zeros((K, num_samples, Ns.size)),
        ledoit=np.zeros((K, num_samples, Ns.size)),
        choice=np.zeros((K, num_samples, Ns.size)),
        # lawless=np.zeros((K, Ns.size)),
        optimal=np.zeros(K),
        input=np.zeros(K),
    )
    best_alpha = dict(
        fixpoint=np.zeros((K, num_samples, Ns.size)),
        mackay=np.zeros((K, num_samples, Ns.size)),
        barber=np.zeros((K, num_samples, Ns.size)),
        grid=np.zeros((K, num_samples, Ns.size)),
        hkb=np.zeros((K, num_samples, Ns.size)),
        ledoit=np.zeros((K, num_samples, Ns.size)),
        # lawless=np.zeros((K, num_samples, Ns.size)),
    )
    for mc in tqdm.trange(num_samples):
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
        SINR['input'] += SINRin / num_samples
        for soi_index in tqdm.trange(K, leave=False):
            # Optimal SINR
            Rtrue_inv = np.linalg.inv(Rtrue)
            a_soi = a_t[soi_index]
            w = Rtrue_inv @ a_soi / (a_soi.conj() @ Rtrue_inv @ a_soi)
            spec = (w.conj() @ Rtrue @ w).real

            w_a = Pk[soi_index] * np.abs(w @ a_soi.conj()) ** 2
            sinr = w_a / (spec - w_a)
            SINR['optimal'][soi_index] += sinr / num_samples

            # Target signal information
            target = {
                'a_t': a_t[soi_index],
                'sigma_s': np.sqrt(Pk[soi_index]),
                'v_s': Pk[soi_index],
                'R': Rtrue
            }
            for i, N in enumerate(tqdm.tqdm(Ns, leave=False)):
                # Input signal
                xx = x[:, :N]

                # Target angle
                # beamformer = capon.RegularizedCapon(a_t[soi_index], xx, target=target)
                beamformer = capon.Wiener(a_t[soi_index], xx, oracle=target)

                options = dict(alpha0=0.5, num_iters=5)
                for key in best_alpha.keys():
                    # alpha_soi = beamformer.alpha_opt(method=key, num_iters=20)
                    # alpha_soi = beamformer.best_alpha(mode=key, num_iters=20, alpha0=1)
                    alpha_soi = beamformer.best_alpha(mode=key, **options)
                    best_alpha[key][soi_index, mc, i] = alpha_soi
                    sinr = beamformer.sinr(alpha_soi)
                    SINR[key][soi_index, mc, i] = sinr

                    # w_hat = beamformer.w_hat(alpha_soi)
                    # sinr = beamformer.sinr(w_hat)
                    # spec = beamformer.spectrum(w_hat)
                # w_hat = beamformer.w_hat(alpha)
                # sinr = beamformer.sinr(w_hat)
                # spec = beamformer.spectrum(w_hat)

                # Chosen alpha
                sinr = beamformer.sinr(alpha)
                SINR['choice'][soi_index, mc, i] = sinr

    # for key in SINR.keys():
    #     SINR[key] /= num_samples

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
# Ns = np.arange(L, N+L, L)
# Ns = np.arange(L, N+L, L)
# Ns = np.logspace(0, 3, 100).astype(int)
Ns = np.unique(np.logspace(np.log10(2), np.log10(N), 100).astype(int))

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

# Steerring vector noise
# noise = 0.01
noise = 0

# Check this specific alpha
alpha = 0

# Number of Monte Carlo samples
num_samples = 100
# num_samples = 5

# Angles
phi = np.pi * np.array([0.2, 0.3, 0.6])
# phi = np.pi * np.array([0.2, 0.1, 0.3, 0.4, 0.6, 0.8])
K = phi.size

# Signal powers
Pk = 10.0 ** (np.array([20, 10, 5]) / 10)
# Pk = 10.0 ** (np.array([20, 10, 10, 10, 10, 10]) / 10)

results = run_experiment_k(
    Ns, L, num_samples=num_samples, phi=phi, noise=noise,
    Pk=Pk, alpha=alpha, seed=0,
)
np.savez('capon_results_k.npz', **results)


