
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# sns.set_style("ticks")
# sns.set_context("paper")
import matplotlib as mpl
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.1

objective_1 = 'conversion'
objective_2 = 'selectivity'

plt.rcParams['font.family'] = 'Helvetica'
# mpl.rc('font', **{'family':'sans-serif', 'sans-serif':['HelveticaLight']})

# Best objectives.
best_conversion_in_scope = 100.
best_selectivity_in_scope = 100.
n_steps = 60
n_experiments = 60
feat_iter = 0

if not os.path.exists('./results_plots'):
    os.mkdir('results_plots')

fig, ax = plt.subplots(figsize=(7., 2.5), dpi=500, nrows=1, ncols=3)

colors_sampling = ['#DC143C', '#0343DF', '#FAC205', '#15B01A']

alphas = [0.4, 0.6, 0.7, 1.0]
i = -1
for sampling_method in ['seed', 'lhs', 'cvtsampling']:

    i += 1
    j = -1
    for batch in [1, 2, 3, 5]:
        j += 1
        acq = 'EHVI'

        df_i = pd.read_csv(f'./results/results_benchmark_dft_acq_{acq}_batch_{batch}_seed_1_init_sampling_{sampling_method}.csv')
        df_i = df_i[df_i['n_experiments'] <= n_experiments]

        # Hypervolume.
        hypervol = df_i['hypervolume completed (%)'].values[:]

        # Plot performance for each acquisition function.
        n_exp = df_i['n_experiments'].values[:]

        ax[i].plot(n_exp, hypervol, color=colors_sampling[j], lw=2.5,
                      label=f"{batch}",
                   alpha=alphas[j])

        ax[i].set_title(f"{sampling_method}")
        ax[i].set_xlabel('Samples')
        ax[i].set_ylabel('Hypervolume (%)')
        ax[i].set_ylim(0, 100)

ax[i].legend()
plt.tight_layout()
plt.savefig(f"results_plots/benchmark_hypervol.svg")

plt.show()

