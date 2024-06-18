
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)
marker_list = ['o', 's', 'd', 'x', '+', 'v', '^', '<', '>', 'p', 'h']

import wiener_utils as utils
import plot_utils as putils
import rir_generator as rir

# Set random seed
np.random.seed(0)


# results = np.load('condition-trim_zeros.npz')
results = np.load('condition.npz')
roots = results['roots']
alpha0 = results['alpha0']
w_true = results['w_true']
snr_vec = results['snr_vec']
Ns = results['Ns']
cond = results['cond']







num_roots = results['num_roots']

false_num_roots = num_roots * ~cond
false_num_roots[false_num_roots == 0] = np.nan

true_num_roots = num_roots * cond
true_num_roots[true_num_roots == 0] = np.nan


data = pd.DataFrame(num_roots.reshape(3, -1).T, columns=[f'$k = {i+1}$' for i in range(3)])
data = pd.DataFrame(true_num_roots.reshape(3, -1).T, columns=[f'$k = {i+1}$' for i in range(3)])
data = pd.DataFrame(false_num_roots.reshape(3, -1).T, columns=[f'$k = {i+1}$' for i in range(3)])
data = pd.melt(data, var_name='Signal', value_name='Num. of roots')

fig, ax = plt.subplots()
sns.histplot(
    ax=ax,
    data=data,
    x='Num. of roots',
    hue='Signal',
    multiple="dodge",
    discrete=True,
    shrink=.8,
    stat='density',
)
ax.set_xlabel('Num. of roots')
ax.set_ylabel('Freq. num. of roots')
ax.set_yscale('log')
ax.grid()
ax.axis([None, None, 1e-4, 1])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()











prob = (1-cond).mean(axis=1)

fig, ax = plt.subplots()
for i, snr in enumerate(snr_vec):
    ax.plot(Ns, prob[i, :], color=color_list[i], label=f'SNR = {snr} dB', marker=marker_list[i], markevery=(2*i, 10))
ax.set_xlabel('$N$')
ax.set_ylabel('Prob. condition is false')
ax.plot(20*np.abs(w_true)[:Ns.max()], alpha=0.5, color='k', label=r'$|\boldsymbol{h}|$')
# ax.plot(20*w_true[:Ns.max()], alpha=0.5, color='k', label=r'$|\boldsymbol{h}|$')
ax.legend()
ax.grid()
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()

# putils.save_fig(fig, 'condition-trim_zeros', format='pdf')
putils.save_fig(fig, 'condition', format='pdf')
putils.save_fig(fig, 'condition', format='pgf')

mean_roots = np.nanmedian(roots, axis=1)
mean_roots = np.nanmedian(mean_roots, axis=-1)

# mean_roots = np.nanmean(np.log(roots), axis=1)
# mean_roots = np.exp(np.nanmean(mean_roots, axis=-1))



fig, ax = plt.subplots()
for i, snr in enumerate(snr_vec):
    ax.plot(Ns, mean_roots[i, :], color=color_list[i], label=f'SNR = {snr} dB', marker=marker_list[i], markevery=(2*i, 10))
ax.set_xlabel('$N$')
ax.set_ylabel(r'Median $\alpha^{(I)}$')
ax.set_yscale('log')
ax.legend()
ax.grid()
# ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()





plt.show()


print()
