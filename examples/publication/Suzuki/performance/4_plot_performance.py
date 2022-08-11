
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
n_steps = 30
feat_iter = 0

if not os.path.exists('./figures'):
    os.mkdir('figures')


for acq in ['EHVI', 'MOUCB', 'MOGreedy']:
    colors = ['#DC143C', '#0343DF', '#FAC205', '#15B01A']
    color_i = 0
    fig, ax = plt.subplots(figsize=(8., 8.0), dpi=500, nrows=2, ncols=2)

    for feat in ['ohe', 'dft', 'mordred', 'random']:
        avg = pd.read_csv(f"./{feat}_{acq}_avg.csv")
        avg = avg.apply(pd.to_numeric, errors='coerce')
        max = pd.read_csv(f"./{feat}_{acq}_max.csv")
        max = max.apply(pd.to_numeric, errors='coerce')
        min = pd.read_csv(f"./{feat}_{acq}_min.csv")
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
            conversion_complete_x = []
            conversion_complete_y = []

        # Distance pareto.
        dtradeoff_max = max['dmaximin_tradeoff'].values[1:]
        dtradeoff_min = min['dmaximin_tradeoff'].values[1:]
        dtradeoff_avg = avg['dmaximin_tradeoff'].values[1:]


        # Best samples at each run.
        bestconversion_max = max['objective_conversion_best'].values[1:]
        bestselectivity_max = max['objective_selectivity_best'].values[1:]
        bestconversion_min = min['objective_conversion_best'].values[1:]
        bestselectivity_min = min['objective_selectivity_best'].values[1:]
        bestconversion_avg = avg['objective_conversion_best'].values[1:]
        bestselectivity_avg = avg['objective_selectivity_best'].values[1:]

        # Where best conversion is sampled.
        try:
            conversion_complete_arg = np.argwhere(bestconversion_max == best_conversion_in_scope)[0]
            conversion_complete_y = [bestconversion_max[conversion_complete_arg]]
            conversion_complete_x = [n_exp[conversion_complete_arg]]
        except:
            conversion_complete_x = []
            conversion_complete_y = []

        # Where best selectivity is sampled.
        try:
            selectivity_complete_arg = np.argwhere(bestselectivity_min == best_selectivity_in_scope)[0]
            selectivity_complete_y = [bestselectivity_min[selectivity_complete_arg]]
            selectivity_complete_x = [n_exp[selectivity_complete_arg]]
        except:
            selectivity_complete_x = []
            selectivity_complete_y = []

        # Plot performance for each acquisition function.
        ax[0][0].plot(n_exp, hypervol_avg, color=colors[color_i], lw=2.5,
                 label=feat.upper())
        ax[0][0].fill_between(x=n_exp,
                        y1=hypervol_avg,
                        y2=hypervol_max, color=colors[color_i], alpha=0.3, lw=0.)
        ax[0][0].fill_between(x=n_exp,
                        y1=hypervol_min,
                        y2=hypervol_avg, color=colors[color_i], alpha=0.3, lw=0.)
        ax[0][0].plot(n_exp, hypervol_min, color=colors[color_i], alpha=1., lw=1., ls='--')
        ax[0][0].plot(n_exp, hypervol_max, color=colors[color_i], alpha=1., lw=1., ls='--')
        ax[0][0].plot(n_exp, np.ones_like(n_exp)*100,
                 dashes=[8, 4], color='black', linewidth=0.8)
        ax[0][0].scatter(n_exp, hypervol_avg, marker='o', s=0., color=colors[color_i])

        ax[0][0].set_xticks(np.arange(0, 120, 10))
        ax[0][0].set_xlim(0, n_steps)
        ax[0][0].set_ylim(0, 100)
        ax[0][0].set_xlabel('Samples')
        ax[0][0].set_ylabel('Hypervolume (%)')

        # Plot distance tradeoff.
        ax[0][1].plot(n_exp, dtradeoff_avg, color=colors[color_i], lw=2.5,
                   label=feat.upper())
        ax[0][1].plot(n_exp, dtradeoff_min, color=colors[color_i], lw=1., ls='--',
                      label=feat.upper())
        ax[0][1].plot(n_exp, dtradeoff_max, color=colors[color_i], lw=1., ls='--',
                      label=feat.upper())


        ax[0][1].fill_between(x=n_exp,
                           y1=dtradeoff_avg,
                           y2=dtradeoff_max, color=colors[color_i], alpha=0.3,
                           )
        ax[0][1].fill_between(x=n_exp,
                           y1=dtradeoff_min,
                           y2=dtradeoff_avg, color=colors[color_i], alpha=0.3,
                           )
        ax[0][1].plot(n_exp, np.ones_like(n_exp) * 0,
                   dashes=[8, 4], color='black', linewidth=0.8)
        ax[0][1].scatter(n_exp, dtradeoff_avg, marker='o', s=0.,
                      color=colors[color_i])


        ax[0][1].set_xticks(np.arange(0, 120, 10))
        ax[0][1].set_xlim(0, n_steps)
        ax[0][1].set_ylim(0, 80)
        ax[0][1].set_xlabel('Samples')
        ax[0][1].set_ylabel(r'$d_{(trade-off)}$')

        # Plot best conversion.
        ax[1][0].plot(n_exp, bestconversion_avg, color=colors[color_i], lw=2.5,
                   label=feat)
        ax[1][0].plot(n_exp, bestconversion_min, color=colors[color_i], lw=1, ls='--',
                      label=feat, alpha=1.)
        ax[1][0].plot(n_exp, bestconversion_max, color=colors[color_i], lw=1, ls='--',
                      label=feat, alpha=1.)
        ax[1][0].fill_between(x=n_exp,
                           y1=bestconversion_avg,
                           y2=bestconversion_max, color=colors[color_i], alpha=0.3,
                           )
        ax[1][0].fill_between(x=n_exp,
                           y1=bestconversion_min,
                           y2=bestconversion_avg, color=colors[color_i], alpha=0.3,
                           )

        ax[1][0].plot(n_exp, np.ones_like(n_exp) * 0,
                   dashes=[8, 4], color='black', linewidth=0.8)
        ax[1][0].scatter(n_exp, bestconversion_avg, marker='o', s=0.,
                      color=colors[color_i])

        ax[1][0].set_xticks(np.arange(0, 120, 10))
        ax[1][0].set_xlim(0, n_steps)
        ax[1][0].set_ylim(20, 100)
        ax[1][0].set_xlabel('Samples')
        ax[1][0].set_ylabel('Best conversion')

        # Plot best selectivity.
        ax[1][1].plot(n_exp, bestselectivity_avg, color=colors[color_i], lw=2.5,
                   label=feat.upper())

        ax[1][1].plot(n_exp, bestselectivity_min, color=colors[color_i], lw=1.0, ls='--',
                      label=feat.upper())
        ax[1][1].plot(n_exp, bestselectivity_max, color=colors[color_i], lw=1.0, ls='--',
                      label=feat.upper())


        ax[1][1].fill_between(x=n_exp,
                           y1=bestselectivity_avg,
                           y2=bestselectivity_max, color=colors[color_i], alpha=0.3,
                           )
        ax[1][1].fill_between(x=n_exp,
                           y1=bestselectivity_min,
                           y2=bestselectivity_avg, color=colors[color_i], alpha=0.3,
                           )
        ax[1][1].plot(n_exp, np.ones_like(n_exp) * 0,
                   dashes=[8, 4], color='black', linewidth=0.8)
        ax[1][1].scatter(n_exp, bestselectivity_avg, marker='o', s=0.,
                      color=colors[color_i])


        ax[1][1].set_xticks(np.arange(0, 120, 10))
        ax[1][1].set_xlim(0, n_steps)
        ax[1][1].set_ylim(0, 100.)
        ax[1][1].set_xlabel('Samples')
        ax[1][1].set_ylabel('Best selectivity')

        color_i += 1
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/benchmark_{acq}.svg")
    plt.show()


