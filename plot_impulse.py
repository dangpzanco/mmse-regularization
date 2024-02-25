
import numpy as np
import rir_generator as rir
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
color_list = list(mcolors.TABLEAU_COLORS)

import plot_utils as putils

L = 600
rir_options = dict(
    c=340,                    # Sound velocity (m/s)
    fs=8e3,                   # Sample frequency (samples/s)
    r=[2, 1.5, 1],            # Receiver position(s) [x y z] (m)
    s=[2, 3.5, 2],            # Source position [x y z] (m)
    L=[5, 4, 6],              # Room dimensions [x y z] (m)
    reverberation_time=0.225,   # Reverberation time (s)
)
h = rir.generate(nsample=L, **rir_options).ravel()

fig, ax = plt.subplots()

ax.plot(h, color='k')
ax.set_ylabel('$h(t)$')
ax.set_xlabel('$t$')
ax.axis([0, L-1, None, None])
ax.grid()
fig.tight_layout()

putils.save_fig(fig, 'impulse_response', format='pdf')

print()

