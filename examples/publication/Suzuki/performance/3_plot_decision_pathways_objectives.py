
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


datasets = ['ohe', 'dft', 'mordred', 'random']
acq = 'EHVI'
batch = 1
total_restarts = 5
n_steps = 30

color_paletes = [sns.color_palette("Blues", n_colors=total_restarts),
                 sns.color_palette("Reds", n_colors=total_restarts),
                 sns.color_palette("Greens", n_colors=total_restarts),
                 sns.color_palette("Oranges", n_colors=total_restarts)]

cp = 0
for dataset in datasets:
    objectives = ['objective_conversion', 'objective_selectivity']
    dict_ratios_plot = {'width_ratios': [0.5, 0.2, 0.5, 0.2], 'wspace': 0.4}
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3),
                           gridspec_kw=dict_ratios_plot)
    obj_counter = 0
    for obj in objectives:

        for seed in range(total_restarts):
            df_benchmark = pd.read_csv(f'../results_{dataset}/results_benchmark_{dataset}_acq_{acq}_batch_{batch}_seed_{seed}.csv')
            df_exp = pd.read_csv('../data/dataset_B1.csv')
            total_number_of_experiments = len(df_exp)

            trace_xy = []
            for i in range(0, n_steps):
                trace_xy.append([df_benchmark['step'][i], df_benchmark[f"{obj}_collected_values"][i]])
            trace_xy = np.reshape(trace_xy, (len(trace_xy), -2))
            ax[0+obj_counter].scatter(trace_xy[:, 0], trace_xy[:, 1],
                        facecolor='white', s=50,
                        edgecolors=color_paletes[cp][seed],
                        zorder=100)
            ax[0+obj_counter].plot(trace_xy[:, 0], trace_xy[:, 1],
                     linestyle='dotted', c=color_paletes[cp][seed],
                     lw=1.1, alpha=1.)
            ax[0+obj_counter].set_xlim(-1, n_steps+1)
            ax[0+obj_counter].set_ylim(-5, 100+10)
            # ax[0].set_title(f'Objective: {obj}')
            sns.despine(trim=True, offset=2, ax=ax[0+obj_counter])
            sns.distplot(a=df_benchmark, x=df_benchmark[f"{obj}_collected_values"],
                         ax=ax[1+obj_counter], vertical=True,
                         hist=False,
                         # bins=20
                         kde_kws={'shade': True,
                                  'color': color_paletes[cp][seed],
                                  'alpha': 0.1},
                         color='black'
                         )

            ax[1+obj_counter].set_xlim(0, 0.025)
            ax[1+obj_counter].set_ylim(-5, 100+10)
            ax[1+obj_counter].axvline(x=0.015, color='black', ls='dotted', alpha=0.5)

        ax[0+obj_counter].set_title(dataset)
        ax[0+obj_counter].set_xlabel('Number of samples collected')
        ax[0+obj_counter].set_ylabel(f"{obj} (in %)")
        hlinecolor = 'black'
        hlinestyle = 'dotted'
        hlinewidth = 0.5
        # plt.hlines(y=13, linestyles=hlinestyle, lw=hlinewidth, colors=hlinecolor, xmin=0, xmax=n_steps)
        # plt.hlines(y=14, linestyles=hlinestyle, lw=hlinewidth, colors=hlinecolor, xmin=0, xmax=n_steps)
        # plt.hlines(y=29, linestyles=hlinestyle, lw=hlinewidth, colors=hlinecolor, xmin=0, xmax=n_steps)
        # plt.hlines(y=9, linestyles=hlinestyle, lw=hlinewidth, colors=hlinecolor, xmin=0, xmax=n_steps)
        # plt.hlines(y=8, linestyles=hlinestyle, lw=hlinewidth, colors=hlinecolor, xmin=0, xmax=n_steps)
        obj_counter += 2
    plt.savefig(f"fig_3_{cp}.svg", format='svg', dpi=500)
    plt.show()
    plt.tight_layout()
    cp += 1
