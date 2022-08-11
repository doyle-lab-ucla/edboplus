
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

objective_1 = 'P'
objective_2 = 'I1'

plt.rcParams['font.family'] = 'Helvetica'
# mpl.rc('font', **{'family':'sans-serif', 'sans-serif':['HelveticaLight']})

# Best objectives.
best_P_in_scope = 100.
best_I1_in_scope = 100.
n_steps = 30

if not os.path.exists('./figures'):
    os.mkdir('figures')


colors = ['#DC143C', '#0343DF', '#FAC205']
color_i = 0
fig, ax = plt.subplots(figsize=(8., 8.), dpi=500, nrows=2, ncols=2)

acq = 'EHVI'
for sampling in ['seed', 'lhs', 'cvtsampling']:

    avg = pd.read_csv(f"./{sampling}_avg.csv")

    avg = avg.apply(pd.to_numeric, errors='coerce')
    max = pd.read_csv(f"./{sampling}_max.csv")
    max = max.apply(pd.to_numeric, errors='coerce')
    min = pd.read_csv(f"./{sampling}_min.csv")
    min = min.apply(pd.to_numeric, errors='coerce')

    n_exp = avg['n_experiments'].values[1:]

    # Hypervolume.
    hypervol_max = max['hypervolume completed (%)'].values[1:]
    hypervol_min = min['hypervolume completed (%)'].values[1:]
    hypervol_avg = avg['hypervolume completed (%)'].values[1:]

        # Where hypervolume is 99% completed.
    try:
        hyper_complete_arg = np.argwhere(hypervol_avg > 99.0)[0]
        hyper_complete_y = [hypervol_avg[hyper_complete_arg]]
        hyper_complete_x = [n_exp[hyper_complete_arg]]
    except:
        P_complete_x = []
        P_complete_y = []

    # Distance pareto.
    dtradeoff_max = max['dmaximin_tradeoff'].values[1:]
    dtradeoff_min = min['dmaximin_tradeoff'].values[1:]
    dtradeoff_avg = avg['dmaximin_tradeoff'].values[1:]


    # Best samples at each run.
    bestP_max = max[f'{objective_1}_best'].values[1:]
    bestI1_max = max[f'{objective_2}_best'].values[1:]
    bestP_min = min[f'{objective_1}_best'].values[1:]
    bestI1_min = min[f'{objective_2}_best'].values[1:]
    bestP_avg = avg[f'{objective_1}_best'].values[1:]
    bestI1_avg = avg[f'{objective_2}_best'].values[1:]

    # Where best P is sampled.
    try:
        P_complete_arg = np.argwhere(bestP_max == best_P_in_scope)[0]
        P_complete_y = [bestP_max[P_complete_arg]]
        P_complete_x = [n_exp[P_complete_arg]]
    except:
        P_complete_x = []
        P_complete_y = []

    # Where best I1 is sampled.
    try:
        I1_complete_arg = np.argwhere(bestI1_min == best_I1_in_scope)[0]
        I1_complete_y = [bestI1_min[I1_complete_arg]]
        I1_complete_x = [n_exp[I1_complete_arg]]
    except:
        I1_complete_x = []
        I1_complete_y = []

    # Plot performance for each acquisition function.
    ax[0][0].plot(n_exp, hypervol_avg, color=colors[color_i], lw=2.5, label=sampling.upper())
    ax[0][0].fill_between(x=n_exp, y1=hypervol_avg, y2=hypervol_max, color=colors[color_i], alpha=0.3, lw=0.)
    ax[0][0].fill_between(x=n_exp, y1=hypervol_min, y2=hypervol_avg, color=colors[color_i], alpha=0.3, lw=0.)
    ax[0][0].plot(n_exp, hypervol_min, color=colors[color_i], alpha=1., lw=1., ls='--')
    ax[0][0].plot(n_exp, hypervol_max, color=colors[color_i], alpha=1., lw=1., ls='--')
    # ax[0][0].plot(n_exp, np.ones_like(n_exp)*100, dashes=[8, 4], color='black', linewidth=0.8)
    ax[0][0].scatter(n_exp, hypervol_avg, marker='o', s=0., color=colors[color_i])

    ax[0][0].set_xticks(np.arange(0, 120, 5))
    ax[0][0].set_xlim(0, n_steps)

    # ax[0][0].set_ylim(40, 100)
    ax[0][0].set_xlabel('Samples')
    ax[0][0].set_ylabel('Hypervolume (%)')
    # plt.tick_params(axis="x", direction="in")
    # plt.tick_params(axis="y", direction="in")

    # Plot distance tradeoff.
    ax[0][1].plot(n_exp, dtradeoff_avg, color=colors[color_i], lw=2.5, label=sampling.upper())
    ax[0][1].plot(n_exp, dtradeoff_min, color=colors[color_i], lw=1., ls='--', label=sampling.upper())
    ax[0][1].plot(n_exp, dtradeoff_max, color=colors[color_i], lw=1., ls='--', label=sampling.upper())
    ax[0][1].fill_between(x=n_exp, y1=dtradeoff_avg, y2=dtradeoff_max, color=colors[color_i], alpha=0.3)
    ax[0][1].fill_between(x=n_exp, y1=dtradeoff_min, y2=dtradeoff_avg, color=colors[color_i], alpha=0.3)
    # ax[0][1].plot(n_exp, np.ones_like(n_exp) * 0, dashes=[8, 4], color='black', linewidth=0.8)
    ax[0][1].scatter(n_exp, dtradeoff_avg, marker='o', s=0., color=colors[color_i])

    ax[0][1].set_xticks(np.arange(0, 120, 5))
    ax[0][1].set_xlim(0, n_steps)
    # ax[0][1].set_ylim(0, 80)
    ax[0][1].set_xlabel('Samples')
    ax[0][1].set_ylabel(r'$d_{(trade-off)}$')

    # Plot best P.
    ax[1][0].plot(n_exp, bestP_avg, color=colors[color_i], lw=2.5, label=sampling)
    ax[1][0].plot(n_exp, bestP_min, color=colors[color_i], lw=1, ls='--', label=sampling, alpha=1.)
    ax[1][0].plot(n_exp, bestP_max, color=colors[color_i], lw=1, ls='--', label=sampling, alpha=1.)
    # ax[1][0].fill_between(x=n_exp, y1=bestP_avg, y2=bestP_max, color=colors[color_i], alpha=0.3)
    # ax[1][0].fill_between(x=n_exp, y1=bestP_min, y2=bestP_avg, color=colors[color_i], alpha=0.3)

    # ax[1][0].plot(n_exp, np.ones_like(n_exp) * 0,
    #            dashes=[8, 4], color='black', linewidth=0.8)
    ax[1][0].scatter(n_exp, bestP_avg, marker='o', s=0.,
                  color=colors[color_i])

    ax[1][0].set_xticks(np.arange(0, 120, 5))
    ax[1][0].set_xlim(0, n_steps)
    # ax[1][0].set_ylim(0.8, 1.1)
    ax[1][0].set_xlabel('Samples')
    ax[1][0].set_ylabel('Best P')

    # Plot best I1.
    ax[1][1].plot(n_exp, bestI1_avg, color=colors[color_i], lw=2.5,
               label=sampling.upper())

    ax[1][1].plot(n_exp, bestI1_min, color=colors[color_i], lw=1.0, ls='--',
                  label=sampling.upper())
    ax[1][1].plot(n_exp, bestI1_max, color=colors[color_i], lw=1.0, ls='--',
                  label=sampling.upper())

    ax[1][1].fill_between(x=n_exp,
                       y1=bestI1_avg,
                       y2=bestI1_max, color=colors[color_i], alpha=0.3,
                       )
    ax[1][1].fill_between(x=n_exp,
                       y1=bestI1_min,
                       y2=bestI1_avg, color=colors[color_i], alpha=0.3,
                       )
    # ax[1][1].plot(n_exp, np.ones_like(n_exp) * 0,
    #            dashes=[8, 4], color='black', linewidth=0.8)
    ax[1][1].scatter(n_exp, bestI1_avg, marker='o', s=0.,
                  color=colors[color_i])


    ax[1][1].set_xticks(np.arange(0, 120, 5))
    ax[1][1].set_xlim(0, n_steps)
    ax[1][1].set_ylim(0.0, 0.005)
    ax[1][1].set_xlabel('Samples')
    ax[1][1].set_ylabel('Best I1')

    color_i += 1

ax[0][1].legend()
plt.tight_layout()
plt.savefig(f"figures/benchmark_sampling.svg")
plt.show()


