import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
import matplotlib as mpl
# mpl.rcParams['grid.linestyle'] = ':'
# mpl.rcParams['grid.linewidth'] = 0.1
plt.rcParams['font.family'] = 'Helvetica'
import joypy
from matplotlib import cm

samplings = ['seed', 'lhs', 'cvtsampling']
objective_1 = 'yield'
objective_2 = 'cost'
max_num_experiments = 46

df_0 = pd.read_csv(f'./results/results_benchmark_dft_acq_EHVI_batch_3_seed_1_init_sampling_{samplings[0]}.csv')
df_0['step'] += 1
df_0 = df_0[df_0['n_experiments'] < max_num_experiments]

df_1 = pd.read_csv(f'./results/results_benchmark_dft_acq_EHVI_batch_3_seed_1_init_sampling_{samplings[1]}.csv')
df_1['step'] += 1
df_1 = df_1[df_1['n_experiments'] < max_num_experiments]

df_2 = pd.read_csv(f'./results/results_benchmark_dft_acq_EHVI_batch_3_seed_1_init_sampling_{samplings[2]}.csv')
df_2['step'] += 1
df_2 = df_2[df_2['n_experiments'] < max_num_experiments]

frames = [df_0, df_1, df_2]
colormaps_obj_1 = [cm.Blues] * 3
colormaps_obj_2 = [cm.Reds] * 3
# colormaps_obj_2 = [cm.PuRd] * 3
# colormaps = [cm.autumn_r, cm.autumn_r, cm.cool, cm.summer]
# pal = sns.cubehelix_palette(10, rot=-.25, light=.7, as_cmap=True)

for i in range(0, 3):
    df = pd.concat(frames)

    plt.figure()
    ax, fig = joypy.joyplot(
        data=eval(f"df_{i}")[['step', f"{objective_1}_collected_values"]],
        by='step',
        linecolor='black',
        linewidth=0.7,
        ylim='own',
        column=['yield_collected_values'],
        colormap=colormaps_obj_1[i],
        legend=False,
        alpha=0.95, #bins=10,
        normalize=False,
        grid=False,
        figsize=(3, 3), #x_range=(0, 100)
        x_range=(0, 100)
    )

    plt.savefig(f'./results_plots/subplot_{samplings[i]}_{objective_1}.svg', format='svg', dpi=500)
    plt.show()
    ax, fig = joypy.joyplot(
        data=eval(f"df_{i}")[['step', f"{objective_2}_collected_values"]],
        by='step',
        linecolor='black',
        linewidth=0.7,
        # hist=True,
        ylim='own',
        column=[f'{objective_2}_collected_values'],
        # color=['#686de0'],
        colormap=colormaps_obj_2[i],
        legend=False,
        alpha=0.95, #bins=10,
        normalize=False, grid=False,
        figsize=(3, 3),
        x_range=(0, 0.4)
    )
    plt.savefig(f'./results_plots/subplot_{samplings[i]}_{objective_2}.svg', format='svg', dpi=500)
    plt.show()
