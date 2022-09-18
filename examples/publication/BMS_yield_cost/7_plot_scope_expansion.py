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
import seaborn as sns


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


df_exp = pd.read_csv('./data/experiments_yield_and_cost.csv')

df_exp['cost'] = -df_exp['cost']
objective_vals = df_exp[['yield', 'cost']].values
pareto_points, idx_pareto = get_pareto_points(objective_vals)
high_tradeoff_points = get_high_tradeoff_points(pareto_points)

print(np.unique(df_exp['base'].values))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))


hues = ['ligand', 'base', 'solvent', 'concentration']

sns.scatterplot(x=df_exp['cost'], y=df_exp['yield'],
                hue=df_exp['ligand'], s=80,
                lw=0.01, edgecolor='black',
                ax=ax, palette='Spectral',
                style=df_exp['solvent'],
                )
sns.lineplot(x=pareto_points[:, 1], y=pareto_points[:, 0],
             linewidth=2, color='grey', ls='dotted', ax=ax)
ax.set_xlim(-0.5, 0.02)
ax.set_ylim(-10, 110)

if not os.path.exists('results_plots'):
    os.mkdir('results_plots')

plt.savefig(f'./results_plots/dataset.svg', format='svg', dpi=500)
# plt.show()

# Reduced space

df_exp = pd.read_csv('./data/experiments_yield_and_cost.csv')

# Removing a ligand.
df_exp = df_exp[df_exp["ligand"].str.contains("CgMe-PPh")==False]
df_exp = df_exp[df_exp["ligand"].str.contains("PPh3")==False]

df_exp['cost'] = -df_exp['cost']
objective_vals = df_exp[['yield', 'cost']].values
pareto_points, idx_pareto = get_pareto_points(objective_vals)
high_tradeoff_points = get_high_tradeoff_points(pareto_points)

print(np.unique(df_exp['base'].values))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

hues = ['ligand', 'base', 'solvent', 'concentration']

sns.scatterplot(x=df_exp['cost'], y=df_exp['yield'],
                hue=df_exp['ligand'], s=80,
                lw=0.01, edgecolor='black',
                ax=ax, palette='Spectral',                
                style=df_exp['solvent'],
                )
sns.lineplot(x=pareto_points[:, 1], y=pareto_points[:, 0],
             linewidth=2, color='grey', ls='dotted', ax=ax)
ax.set_xlim(-0.5, 0.02)
ax.set_ylim(-10, 110)

if not os.path.exists('results_plots'):
    os.mkdir('results_plots')
plt.savefig(f'./results_plots/dataset_reduced.svg', format='svg', dpi=500)
# plt.show()

