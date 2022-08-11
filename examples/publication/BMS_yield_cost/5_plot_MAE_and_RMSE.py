import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pareto
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from sklearn.preprocessing import MinMaxScaler

from edbo.plus.benchmark.multiobjective_benchmark import is_pareto

sns.set_style("ticks")
import matplotlib as mpl
# mpl.rcParams['grid.linestyle'] = ':'
# mpl.rcParams['grid.linewidth'] = 0.1
plt.rcParams['font.family'] = 'Helvetica'
import joypy
from matplotlib import cm

##############

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

######


samplings = ['seed', 'lhs', 'cvtsampling']
batch_sizes = [1, 2, 3, 5]
# colorpalettes = ['Blues', 'Reds', 'Greens', 'Oranges']
max_number_experiments = 45
objective_1 = 'yield'
objective_2 = 'cost'

colors = ['blue', 'green', 'red']

df_all = pd.read_csv(f'./results/results_benchmark_dft_acq_EHVI_batch_{batch_sizes[0]}_seed_1_init_sampling_{samplings[0]}.csv')
for i in batch_sizes:
    for j in samplings:
        df_i = pd.read_csv(f'./results/results_benchmark_dft_acq_EHVI_batch_{i}_seed_1_init_sampling_{j}.csv')
        df_i = df_i[df_i['n_experiments'] <= max_number_experiments]
        df_all = df_all.append(df_i, ignore_index=True)


df_all.drop_duplicates(inplace=True)

df_finish = df_all[(df_all['n_experiments'] < max_number_experiments+2) & (df_all['n_experiments'] > max_number_experiments-2)]

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 2.2))

sns.barplot(data=df_finish, x='init_method', y='MAE_yield',
            hue='batch', ax=ax[0], palette='Blues',
            lw=0.7, edgecolor='black', ci=None)
# ax[0].set_ylim((5, 18))

sns.barplot(data=df_finish, x='init_method', y='MAE_cost',
            hue='batch', ax=ax[1], palette='Reds',
            lw=0.7, edgecolor='black', ci=None)
# ax[1].set_ylim(0.01)


sns.barplot(data=df_finish, x='init_method', y='RMSE_yield',
            hue='batch', ax=ax[2], palette='Blues',
            lw=0.7, edgecolor='black', ci=None)
# ax[2].set_ylim(10, 25)

sns.barplot(data=df_finish, x='init_method', y='RMSE_cost',
            hue='batch', ax=ax[3], palette='Reds',
            lw=0.7, edgecolor='black', ci=None)
# ax[3].set_ylim(0.01, 0.06)


plt.savefig('./results_plots/fig2c.svg', format='svg', dpi=500)
plt.tight_layout()
plt.show()
