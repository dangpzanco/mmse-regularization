
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
marker_list = ['s', 'o', 'd', '^', 'v', '<', '>', 'p', 'h', 'x']

import plot_utils as putils
import capon_utils as capon


def plot_alpha(results, label_dict=None, keys=None, soi_index=0):
    best_alpha = results['best_alpha'].copy()
    Ns = results['Ns']

    if keys is None:
        keys = best_alpha.keys()

    if label_dict is None:
        label_dict = dict(zip(keys, keys))
    else:
        keys = label_dict.keys()

    fig, ax = plt.subplots()

    for i, key in enumerate(keys):
        _, num_samples, _ = best_alpha[key].shape
        alpha_vec = best_alpha[key][soi_index]
        if np.isnan(alpha_vec).any():
            print(f'Warning: best_alpha[{key}][{soi_index}] has NaN values')
        alpha_vec[alpha_vec == np.inf] = np.nan

        ax.plot(
            # Ns, np.mean(alpha_vec, axis=0),
            Ns, np.nanmedian(alpha_vec, axis=0),
            label=label_dict[key],
            color=color_list[i],
            marker=marker_list[i],
            # markevery=(3*i, 10),
            # markevery=10,
            markevery=(0.02*i, 0.1),
            fillstyle='none',
        )

        alpha_vec[np.isnan(alpha_vec)] = np.inf
        putils.plot_with_fill(
            ax, Ns, alpha_vec,
            color=color_list[i],
            alpha=1/num_samples
        )

    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    # ax.set_ylabel(r'$\alpha^{(I)}$')
    ax.set_ylabel(r'$\alpha$')
    # ax.legend(loc='lower right')
    ax.legend(loc='lower left')
    # ax.axis([Ns[0], Ns[-1], 1e-1, 2e4])
    ax.grid()

    return fig, ax



def plot_N(results, label_dict, soi_index=0):

    Ns = results['Ns']
    data = results['SINR']
    data_alpha = data['choice']
    data_optimal = data['optimal']
    alpha = results['alpha'].item()

    fig, ax = plt.subplots()
    ax.axhline(10*np.log10(data_optimal[soi_index]), color='k', linestyle='--', label='Optimal')

    for i, key in enumerate(label_dict.keys()):
        if np.isnan(data[key][soi_index]).any():
            print(f'Warning: SINR[{key}][{soi_index}] has NaN values')
        ax.plot(
                Ns,
                10*np.log10(np.nanmedian(data[key][soi_index], axis=0)),
                # 10*np.log10(np.nanmean(data[key][soi_index], axis=0)),
                # 10*np.log10(np.nanmedian(data[key][soi_index], axis=0)),
                label=label_dict[key],
                color=color_list[i],
                marker=marker_list[i],
                # markevery=10,
                # markevery=(3*i, 10),
                markevery=(0.02*i, 0.1),
                fillstyle='none',
            )
    ax.plot(
            Ns,
            10*np.log10(np.nanmedian(data_alpha[soi_index], axis=0)),
            label=f'$\\alpha = {alpha}$',
            color=color_list[i+1],
            marker=marker_list[i+1],
            # markevery=10,
            # markevery=(i+1, 10),
            markevery=(0, 0.1),
            fillstyle='none',
        )
    ax.legend(loc='lower right')
    ax.set_xlabel('$N$')
    ax.grid()
    ax.set_ylabel(r'$\mathsf{SINR}_' + f'{soi_index+1}$ [dB]')

    # ax.axis([Ns[0], Ns[-1], 0, 31])

    return fig, ax


def plot_phi(results, label_dict):
    phi = results['phi']
    data = results['SINR_phi']
    data_alpha = data['choice']
    alpha = results['alpha']
    num_alpha = alpha.size

    fig, ax = plt.subplots()
    for key in label_dict.keys():
        ax.plot(
            phi / np.pi,
            10*np.log10(data[key]),
            label=label_dict[key]
        )
    for i in range(num_alpha):
        ax.plot(
            phi / np.pi,
            10*np.log10(data_alpha[i]),
            label=f'$\\alpha = {alpha[i]}$',
        )
    ax.set_xlabel(r'$\phi$')
    ax.grid()

    ax.axis([0, 1, -40, 22])
    ax.set_ylabel('SINR (dB)')
    ax.legend(loc='lower right')

    return fig, ax


filename = 'capon_results_k.npz'
results = dict(np.load(filename, allow_pickle=True))
for key in results.keys():
    if results[key].dtype == 'object':
        results[key] = results[key].item()

label_dict = dict(
    # fixpoint='Proposed',
    # mackay='Proposed',
    # barber='Barber',
    # lawless='Lawless-Wang',
    grid='Oracle',
    mackay='Gull-MacKay',
    ledoit='Ledoit-Wolf',
    hkb='HKB',
)

K = results['SINR']['optimal'].size

for soi_index in range(K):
    fig, ax = plot_N(results, label_dict, soi_index=soi_index)
    ax.axis([None, None, -10, [40, 30, 20][soi_index]])
    ax.set_xscale('log')
    putils.save_fig(fig, f'capon_sinr{soi_index}', format='pdf')

    fig, ax = plot_alpha(results, label_dict, soi_index=soi_index)
    ax.axis([None, None, 1e-2, 1e5])
    ax.set_xscale('log')
    putils.save_fig(fig, f'capon_best_alpha{soi_index}', format='pdf')


# fig, ax = plot_phi(results, 'SINR_phi', label_dict)
# ax.axis([0, 1, -36, 30])
# putils.save_fig(fig, 'sinr_phi', format='pdf')


# fig, ax = plot_phi(results, 'SINR_phi', label_dict)
# ax.axis([0.195, 0.205, 0, 30])
# putils.save_fig(fig, 'sinr_phi_zoom', format='pdf', tight_scale=False)

plt.show()
print()



