
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.despine()
import matplotlib as mpl
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.1
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 10
import pareto
from edbo.plus.benchmark.multiobjective_benchmark import is_pareto
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from sklearn.preprocessing import MinMaxScaler

# Hue: Color (ligand), shape (base), filling (solvent), alpha (ligand_eq).

import seaborn as sns

dataset = 'dft'
acq = 'EHVI'
batch = 1
total_restarts = 5
n_steps = 30
seed = 0


def get_pareto_points(objective_values):
    """ Get pareto for the ground truth function.
    NOTE: Assumes maximization."""
    pareto_ground = pareto.eps_sort(tables=objective_values,
                                    objectives=np.arange(2),
                                    maximize_all=True)
    idx_pareto = is_pareto(objectives=-objective_values)
    return np.array(pareto_ground), idx_pareto


def get_high_tradeoff_points(pareto_points):
    """ Pass a numpy array with the pareto points and returns a numpy
        array with the high tradeoff points."""

    scaler_pareto = MinMaxScaler()
    pareto_scaled = scaler_pareto.fit_transform(pareto_points)
    try:
        tradeoff = HighTradeoffPoints()

        tradeoff_args = tradeoff.do(-pareto_scaled)  # Always minimizing.
        tradeoff_points = pareto_points[tradeoff_args]
    except:
        tradeoff_points = []
        pass
    return tradeoff_points


df_exp = pd.read_csv('../data/dataset_B1.csv')
objective_vals = df_exp[['objective_conversion', 'objective_selectivity']].values
pareto_points, idx_pareto = get_pareto_points(objective_vals)
high_tradeoff_points = get_high_tradeoff_points(pareto_points)


df_benchmark = pd.read_csv(f'../results_{dataset}/results_benchmark_{dataset}_acq_{acq}_batch_{batch}_seed_{seed}.csv')

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 15))

palettes = [['tab10', 'viridis'], [None, 'Blues']]

hues = [['ligand', 'base'], ['solvent', 'ligand_equivalent']]
for i in range(0, 2):
    for j in range(0, 2):
        sns.scatterplot(x=df_exp['objective_conversion'], y=df_exp['objective_selectivity'],
                        hue=df_exp[hues[i][j]], s=40, lw=1., edgecolor='black', ax=ax[i][j], palette=palettes[i][j])
        sns.lineplot(x=pareto_points[:, 0], y=pareto_points[:, 1],
                     linewidth=1.2, color='grey', ls='dotted', ax=ax[i][j])
        ax[i][j].set_xlim(-5, 105)
        ax[i][j].set_ylim(-5, 105)
        ax[i][j].legend(loc=4)
        ax[i][j].set_title(hues[i][j])

plt.tight_layout()
plt.show()

palettes = ['tab10', None]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
hues = ['ligand', 'solvent']

for i in range(0, 2):
    sns.scatterplot(x=df_exp['objective_conversion'], y=df_exp['objective_selectivity'],
                    hue=df_exp[hues[i]], s=50, lw=1., edgecolor='black', ax=ax[i], palette=palettes[i])
    sns.lineplot(x=pareto_points[:, 0], y=pareto_points[:, 1],
                 linewidth=1.2, color='grey', ls='dotted', ax=ax[i])
    ax[i].set_xlim(-5, 105)
    ax[i].set_ylim(-5, 105)
    ax[i].legend(loc=4)
    ax[i].set_title(hues[i])

ax[0].legend(scatterpoints=1, loc='best', ncol=2, markerscale=1, fontsize=9)
ax[1].legend(scatterpoints=1, loc='best', ncol=2, markerscale=1, fontsize=9)

plt.tight_layout()
plt.savefig('Fig2_scope.svg', dpi=500, format='svg')
plt.show()
