import os.path

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


df_exp = pd.read_csv('../data/data.csv')
df_exp['I1'] = -df_exp['I1'].values
objective_vals = df_exp[['P', 'I1']].values
pareto_points, idx_pareto = get_pareto_points(objective_vals)
high_tradeoff_points = get_high_tradeoff_points(pareto_points)

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

print(df_exp.columns)

palettes = [['Reds', 'Reds', 'Blues'],
            ['Greens', 'Oranges', 'Reds'],
            ['Blues', 'Greens', 'Oranges']
            ]

hues = [['Temperature', 'Temperature', 'Volume'],
        ['D', 'SM2', 'W'],
        ['Mixing', 'Time', 'WB']
        ]

for i in range(0, 3):
    for j in range(0, 3):
        sns.scatterplot(x=df_exp['P'], y=df_exp['I1'],
                        hue=df_exp[hues[i][j]], s=40, lw=1., edgecolor='black', ax=ax[i][j], palette=palettes[i][j])
        sns.lineplot(x=pareto_points[:, 0], y=pareto_points[:, 1],
                     linewidth=1.2, color='grey', ls='dotted', ax=ax[i][j])
        # ax[i][j].set_xlim(-5, 105)
        # ax[i][j].set_ylim(-5, 105)
        ax[i][j].legend(loc=3)
        ax[i][j].set_title(hues[i][j])
fig.delaxes(ax[0][0])
plt.tight_layout()

if not os.path.exists('../plots'):
    os.mkdir('../plots')
plt.savefig('../plots/SI_ground_truth.svg', dpi=500, format='svg')
plt.show()
