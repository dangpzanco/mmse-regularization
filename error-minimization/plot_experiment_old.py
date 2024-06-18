
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import plot_utils as putils


def load_data(mode='trueL'):
    if mode == 'trueL':
        data = np.load('best_l2reg_trueL.npz', allow_pickle=True)
    elif mode == 'mismatch':
        data = np.load('best_l2reg_mismatch.npz', allow_pickle=True)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    error_grid = data['error_grid'].item()
    error_bayes = data['error_bayes'].item()
    best_alpha = data['best_alpha'].item()
    Ns = data['Ns']
    SNR = data['SNR']

    return error_grid, error_bayes, best_alpha, Ns, SNR


def plot_alpha(best_alpha, Ns, SNR, ax=None):
    _, _, num_samples = best_alpha['mis'].shape

    if ax is None:
        fig, ax = plt.subplots()

    # fig, ax = plt.subplots()
    for i, snr in enumerate(SNR):
        ax.plot(
            Ns, best_alpha['mis'][:,i,].mean(axis=-1),
            color=color_list[i],
            label=f'${snr}$ dB',
        )
        putils.plot_with_fill(
            ax, Ns, best_alpha['mis'][:,i,],
            color=color_list[i], alpha=1/num_samples)

        ax.plot(
            Ns, best_alpha['mackay'][:,i,].mean(axis=-1),
            color=color_list[i],
            ls='--',
        )
        putils.plot_with_fill(
            ax, Ns, best_alpha['mackay'][:,i,],
            ls='--',
            color=color_list[i], alpha=1/num_samples)
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\hat\alpha, \alpha^{(I)}$')
    ax.grid()

    return fig, ax


def plot_metric(error_grid, error_bayes, Ns, SNR, metric='mis', ax=None):
    _, _, num_samples = error_grid[metric].shape

    if ax is None:
        fig, ax = plt.subplots()

    for i, snr in enumerate(SNR):
        ax.plot(
            Ns, error_grid[metric][:,i].mean(axis=-1),
            label=f'${snr}$ dB', color=color_list[i]
        )
        putils.plot_with_fill(
            ax, Ns, error_grid[metric][:,i],
            ls='-',
            color=color_list[i], alpha=1/num_samples
        )

        ax.plot(
            Ns, error_bayes[metric][:,i].mean(axis=-1),
            color=color_list[i], ls='--',
        )
        putils.plot_with_fill(
            ax, Ns, error_bayes[metric][:,i],
            ls='--',
            color=color_list[i], alpha=1/num_samples
        )
    ax.legend()
    ax.set_xlabel('$N$')
    ax.axis([Ns[0], Ns[-1], -10, 0])
    if metric == 'mis':
        ax.set_ylabel(r'$\hat{\textrm{M}}, \textrm{M}(\alpha^{(I)})$')
    elif metric == 'valid':
        ax.set_ylabel(r'$\hat{\textrm{MSE}}_{\textrm{val}}, \textrm{MSE}_{\textrm{val}}(\alpha^{(I)})$')
    ax.grid()

    return fig, ax


for mode in ['trueL', 'mismatch']:
    error_grid, error_bayes, best_alpha, Ns, SNR = load_data(mode=mode)
    num_Ns, num_SNRs, num_samples = best_alpha['mis'].shape

    fig, ax = plot_alpha(best_alpha, Ns, SNR)
    if mode == 'trueL':
        ax.axis([Ns[0], Ns[-1], 1e-3, None])
        putils.save_fig(fig, f'best_alpha', format='pdf')
    elif mode == 'mismatch':
        ax.axis([Ns[0], Ns[-1], 1e-3, None])
        putils.save_fig(fig, f'best_alpha_mismatch', format='pdf')

    fig, ax = plot_metric(error_grid, error_bayes, Ns, SNR, metric='mis')
    if mode == 'trueL':
        ax.axis([Ns[0], Ns[-1], -20, 0])
        putils.save_fig(fig, f'best_misalign', format='pdf')
    elif mode == 'mismatch':
        ax.axis([Ns[0], Ns[-1], -10, 0])
        putils.save_fig(fig, f'best_misalign_mismatch', format='pdf')


print()

