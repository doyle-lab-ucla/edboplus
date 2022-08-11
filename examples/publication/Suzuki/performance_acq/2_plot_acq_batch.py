import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


n_steps = 30
colors = ['#DC143C', '#0343DF', '#FAC205']
feat = 'dft'
fig, ax = plt.subplots(figsize=(15., 4.), dpi=500, nrows=1, ncols=4)

batch_count = 0
for batch in [1, 2, 3, 5]:
    color_i = 0
    for acq in ['EHVI', 'MOUCB', 'MOGreedy']:
        avg = pd.read_csv(f"./{feat}_{acq}_{batch}_avg.csv")
        avg = avg.apply(pd.to_numeric, errors='coerce')
        max = pd.read_csv(f"./{feat}_{acq}_{batch}_max.csv")
        max = max.apply(pd.to_numeric, errors='coerce')
        min = pd.read_csv(f"./{feat}_{acq}_{batch}_min.csv")
        min = min.apply(pd.to_numeric, errors='coerce')
        n_exp = avg['n_experiments'].values[1:]

        hypervol_max = max['hypervolume completed (%)'].values[1:]
        hypervol_min = min['hypervolume completed (%)'].values[1:]
        hypervol_avg = avg['hypervolume completed (%)'].values[1:]
        # Plot performance for each acquisition function.
        ax[batch_count].plot(n_exp, hypervol_avg, color=colors[color_i], lw=2.5, label=acq.upper())
        ax[batch_count].fill_between(x=n_exp, y1=hypervol_avg, y2=hypervol_max, color=colors[color_i], alpha=0.3, lw=0.)
        ax[batch_count].fill_between(x=n_exp, y1=hypervol_min, y2=hypervol_avg, color=colors[color_i], alpha=0.3, lw=0.)
        ax[batch_count].plot(n_exp, hypervol_min, color=colors[color_i], alpha=1., lw=1., ls='--')
        ax[batch_count].plot(n_exp, hypervol_max, color=colors[color_i], alpha=1., lw=1., ls='--')
        ax[batch_count].plot(n_exp, np.ones_like(n_exp) * 100, dashes=[8, 4], color='black', linewidth=0.8)
        ax[batch_count].scatter(n_exp, hypervol_avg, marker='o', s=0., color=colors[color_i])

        ax[batch_count].set_xticks(np.arange(0, 120, 5))
        ax[batch_count].set_xlim(0, n_steps)
        ax[batch_count].set_ylim(0, 100)
        ax[batch_count].set_xlabel('Samples')
        ax[batch_count].set_ylabel('Hypervolume (%)')
        color_i += 1

    batch_count += 1
    plt.legend()

if not os.path.exists('figures'):
    os.mkdir('figures')

plt.tight_layout()
plt.savefig(f"figures/benchmark_acquisition_functions_batch.svg")
plt.show()

