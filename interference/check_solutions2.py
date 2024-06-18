
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

import capon_utils as capon
import plot_utils as putils
import rir_generator as rir

# Set random seed
np.random.seed(0)


results = np.load('condition.npz')
roots = results['roots']
alpha0 = results['alpha0']
# x_samples = results['x_samples']
loss = results['loss']
loss_inf = results['loss_inf']
a = results['a']
Pk = results['Pk']
Ns = results['Ns']
cond = results['cond']
Rtrue = results['Rtrue']

K, M = a.shape
SINRin = M * Pk / (np.trace(Rtrue).real - M * Pk)


# loss[~cond] > loss_inf[~cond][:, None]

# val = np.nanmin(np.nanmin(loss[~cond], axis=-1) - loss_inf[~cond])


prob = (1-cond).mean(axis=1)

fig, ax = plt.subplots()
for i, power in enumerate(Pk):
    ax.plot(Ns, prob[i, :],
            color=color_list[i],
            label=f'$k = {i+1}$',
            marker=marker_list[i],
            fillstyle='none',
    )
ax.set_xlabel('$N$')
ax.set_ylabel('Freq. (46) is false')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
ax.axis([None, None, 1e-3, 1])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()

# putils.save_fig(fig, 'capon-condition-trim_zeros', format='pdf')
putils.save_fig(fig, 'capon-condition', format='pdf')
putils.save_fig(fig, 'capon-condition', format='pgf')



cond = results['cond'].copy()
num_roots = results['num_roots'].copy()
K, num_samples, num_N = num_roots.shape

pd_list = []
for k in range(K):
    df = pd.DataFrame(num_roots[k], columns=Ns).astype('int')
    df = pd.melt(df, var_name='Num. samples', value_name='Num. of roots')
    pd_list.append(df)
data = pd.concat(pd_list, keys=np.arange(K)).reset_index().drop('level_1',axis=1)
data.columns = ['Signal', *data.columns[1:]]


fig, ax = plt.subplots()
sns.boxplot(
    ax=ax,
    data=data,
    y='Num. of roots',
    x='Num. samples',
    hue='Signal',
)
ax.set_xlabel('Num. of roots')
ax.set_ylabel('Freq. num. of roots')
ax.set_yscale('log')
ax.grid()
ax.axis([None, None, 1e-4, 1])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()









# ind_Ns = 6
ind_Ns = -7
ind_Ns = 0
num_roots = results['num_roots'].copy()
num_roots = num_roots[:, :, ind_Ns]

data = pd.DataFrame(num_roots.T, columns=[f'$k = {i+1}$' for i in range(3)])
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
    common_norm=False
)
ax.set_xlabel('Num. of roots')
ax.set_ylabel('Freq. num. of roots')
ax.set_yscale('log')
ax.grid()
ax.axis([None, None, 1e-3, 1])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()
putils.save_fig(fig, 'capon-num_roots', format='pdf', leg=False)


ind_Ns = -1
num_roots = results['num_roots'].copy()
num_roots = num_roots[:, :, ind_Ns]

data = pd.DataFrame(num_roots.T, columns=[f'$k = {i+1}$' for i in range(3)])
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
    common_norm=False
)
ax.set_xlabel('Num. of roots')
ax.set_ylabel('Freq. num. of roots')
ax.set_yscale('log')
ax.grid()
ax.axis([None, None, 1e-3, 1])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()
putils.save_fig(fig, 'capon-num_roots_largeN', format='pdf', leg=False)







cond = results['cond'].copy()
num_roots = results['num_roots'].copy()
prob = (1-(num_roots == 2)).mean(axis=1)
fig, ax = plt.subplots()
for i, power in enumerate(Pk):
    ax.plot(Ns, prob[i, :],
            color=color_list[i],
            label=f'$k = {i+1}$',
            marker=marker_list[i],
            fillstyle='none',
    )
ax.set_xlabel('$N$')
ax.set_ylabel('Freq. (50) is false')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
ax.axis([None, None, 1e-5, 1])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()









cond = results['cond']
num_roots = results['num_roots']
num_samples = num_roots.shape[1]
# mean_roots = np.mean(num_roots, axis=1)
mean_roots = np.max(num_roots, axis=1)



false_num_roots = num_roots * ~cond
false_num_roots[false_num_roots == 0] = np.nan

true_num_roots = num_roots * cond
true_num_roots[true_num_roots == 0] = np.nan


data = pd.DataFrame(true_num_roots.reshape(3, -1).T, columns=[f'$k = {i+1}$' for i in range(3)])
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
putils.save_fig(fig, 'capon-true_num_roots', format='pdf', leg=False)




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
putils.save_fig(fig, 'capon-false_num_roots', format='pdf', leg=False)


data = pd.DataFrame(num_roots.reshape(3, -1).T, columns=[f'$k = {i+1}$' for i in range(3)])
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
putils.save_fig(fig, 'capon-num_roots', format='pdf', leg=False)








fig, ax = plt.subplots()
ax.hist(num_roots.reshape(3, -1).T,
    bins=np.arange(7),
    density=True,
    label=[f'$k = {i+1}$' for i in range(3)],
)
ax.set_xlabel('Num. of roots')
ax.set_ylabel('Freq. num. of roots')
ax.set_yscale('log')
ax.legend()
ax.grid()
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()


fig, ax = plt.subplots()
for i, power in enumerate(Pk):
    # ax.plot(Ns, mean_roots[i, :],
    #         color=color_list[i],
    #         label=f'$k = {i+1}$',
    #         marker=marker_list[i],
    #         fillstyle='none',
    # )
    ax.hist(num_roots[i].ravel(),
            color=color_list[i],
            label=f'$k = {i+1}$',
            bins=np.arange(-1, 10),
            alpha=0.5,
            # marker=marker_list[i],
            # fillstyle='none',
    )
ax.set_xlabel('$N$')
ax.set_ylabel('Mean num. of roots')
ax.legend()
ax.set_xscale('log')
# ax.set_yscale('log')
ax.grid()
# ax.axis([None, None, 1e-2, 1])
ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()














mean_roots = np.nanmedian(roots, axis=1)
mean_roots = np.nanmedian(mean_roots, axis=-1)

# mean_roots = np.nanmean(np.log(roots), axis=1)
# mean_roots = np.exp(np.nanmean(mean_roots, axis=-1))




fig, ax = plt.subplots()
for i, power in enumerate(Pk):
    # ax.plot(Ns, mean_roots[i, :], color=color_list[i], label=f'Power = {10*np.log10(power)} dB', marker=marker_list[i], markevery=(2*i, 10))
    ax.plot(Ns, mean_roots[i, :], color=color_list[i], label=f'Power = {10*np.log10(power)} dB', marker=marker_list[i])
ax.set_xlabel('$N$')
ax.set_ylabel(r'Median $\alpha^{(I)}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid()
# ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()





plt.show()


exponent = 10 ** np.log10(roots).round()
coeff = (roots / exponent).round(decimals=3)
round_roots = coeff * exponent

np.unique(round_roots, axis=-1)

K, num_samples, num_N, num_inits = roots.shape

num_roots = np.zeros((K, num_samples, num_N), dtype=int)
for k in tqdm.trange(K):
    for i in tqdm.trange(num_samples, leave=False):
        for n in tqdm.trange(num_N, leave=False):
            if cond[k, i, n]:
                if np.log10(roots[k, i, n]).std() > 1:
                    num_roots[k, i, n] = np.unique(round_roots[k, i, n]).size
                else:
                    num_roots[k, i, n] = 1
            else:
                num_roots[k, i, n] = 0

mean_num_roots = np.mean(num_roots, axis=1)
max_num_roots = np.max(num_roots, axis=1)
median_num_roots = np.median(num_roots, axis=1)


fig, ax = plt.subplots()
for i, power in enumerate(Pk):
    # putils.plot_with_fill(ax, Ns, num_roots[i])
    ax.plot(Ns, max_num_roots[i, :], color=color_list[i], label=f'Power = {10*np.log10(power)} dB', marker=marker_list[i], markevery=(2*i, 10))
ax.set_xlabel('$N$')
ax.set_ylabel('Num. roots')
ax.set_xscale('log')
# ax.set_yscale('log')
ax.legend()
ax.grid()
# ax.autoscale(enable=True, axis='x', tight=True)
fig.tight_layout()




print()

x_samples = results['x_samples']
a = results['a']
Rtrue = results['Rtrue']

# k, i, n = 0, 43, 0
k, i, n = 2, 43, 0

x = x_samples[k, i][:, :Ns[n]]
wiener = capon.Wiener(a[k], x, oracle={'v_s': Pk[k], 'R': Rtrue})

alpha = np.logspace(-10, 20, 1000)
L = wiener.likelihood(alpha)
dL = wiener.d_likelihood(alpha)
sinr = wiener.sinr(alpha)

# res = opt.minimize_scalar(wiener.likelihood, bracket=(-3, 13), method='brent').x

fig, ax = plt.subplots()
ax.plot(alpha, L, label=r"$L(\alpha)$")
ax.plot(alpha, dL, label=r"$L'(\alpha)$")
ax.plot(alpha, sinr, label="SINR")
ax.set_xscale('log')
ax.set_yscale('symlog')
ax.grid()
ax.set_xlabel(r"$\alpha$")
ax.legend()


fig, ax = plt.subplots()
ax.plot(alpha0, roots[k, i, n])
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
ax.set_xlabel(r"$\alpha^{(0)}$")
ax.set_ylabel(r"$\alpha^{(I)}$")



# root
# for k in tqdm.trange(K):
#     for i in tqdm.trange(num_samples, leave=False):
#         for n in tqdm.trange(num_N, leave=False):
#             if not cond[k, i, n]:


