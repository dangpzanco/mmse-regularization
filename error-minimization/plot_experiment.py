
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)
marker_list = ['s', 'o', 'd', '^', 'v', '<', '>', 'p', 'h', 'x']

import plot_utils as putils


def load_data(mode='trueL'):
    if mode == 'trueL':
        data = np.load('wiener_trueL.npz', allow_pickle=True)
    elif mode == 'mismatch':
        data = np.load('wiener_mismatch.npz', allow_pickle=True)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    misaligment = data['misaligment'].item()
    best_alpha = data['best_alpha'].item()
    Ns = data['Ns']
    SNR = data['SNR']

    return misaligment, best_alpha, Ns, SNR


def plot_alpha(results, label_dict, snr_index=0):
    best_alpha = results['best_alpha']
    Ns = results['Ns']
    keys = label_dict.keys()

    fig, ax = plt.subplots()

    for i, key in enumerate(keys):
        _, _, num_samples = best_alpha[key].shape
        ax.plot(
            Ns, np.nanmean(best_alpha[key][:, snr_index], axis=-1),
            label=label_dict[key],
            color=color_list[i],
            marker=marker_list[i],
            # markevery=(2*i, 10),
            markevery=(0.02*i, 0.1),
            fillstyle='none',
        )
        putils.plot_with_fill(
            ax, Ns, best_alpha[key][:, snr_index],
            color=color_list[i],
            alpha=1/num_samples
        )
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$\hat\alpha, \alpha^{(I)}$')
    ax.grid()

    return fig, ax


def plot_misalign(results, label_dict, snr_index=0):
    misalignment = results['misalignment']
    Ns = results['Ns']
    keys = label_dict.keys()

    fig, ax = plt.subplots()

    for i, key in enumerate(keys):
        _, _, num_samples = misalignment[key].shape
        ax.plot(
            Ns, np.nanmean(misalignment[key][:, snr_index], axis=-1),
            label=label_dict[key],
            color=color_list[i],
            marker=marker_list[i],
            # markevery=(2*i, 10),
            markevery=(0.02*i, 0.1),
            fillstyle='none',
        )
        putils.plot_with_fill(
            ax, Ns, misalignment[key][:, snr_index],
            ls='-',
            color=color_list[i], alpha=1/num_samples
        )
    ax.legend()
    ax.set_xlabel('$N$')
    ax.axis([Ns[0], Ns[-1], -10, None])
    ax.set_ylabel(r'$\hat{\mathsf{m}}, \mathsf{m}(\alpha^{(I)})$ [dB]')
    ax.grid()

    return fig, ax


label_dict = dict(
    # fixpoint='Proposed',
    # lawless='Lawless-Wang',
    grid='Oracle',
    mackay='Gull-MacKay',
    ledoit='Ledoit-Wolf',
    hkb='HKB',
)

label_dict_mis = label_dict.copy()
label_dict_mis['no_alpha'] = '$\\alpha = 0$'

print()

results = dict(np.load('wiener_misalignment.npz', allow_pickle=True))
for key in results.keys():
    if results[key].dtype == 'object':
        results[key] = results[key].item()

SNR = results['SNR']
Ns = results['Ns']

for i, snr in enumerate(SNR):
    fig, ax = plot_alpha(results, label_dict, snr_index=i)
    ax.axis([Ns[0], Ns[-1], [1e-2, 1e-3][i], [10, 1][i]])
    putils.save_fig(fig, f'best_alpha_snr={snr}', format='pdf')

    fig, ax = plot_misalign(results, label_dict_mis, snr_index=i)
    ax.axis([Ns[0], Ns[-1], [-5, -20][i], [10, 0][i]])
    putils.save_fig(fig, f'misalign_snr={snr}', format='pdf')


print()

