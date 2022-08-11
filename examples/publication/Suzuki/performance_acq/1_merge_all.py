import pandas as pd
import numpy as np


objective_1 = 'objective_conversion'
objective_2 = 'objective_selectivity'
columns_to_keep = ['step', 'n_experiments', 'hypervolume completed (%)']

for batch in [1, 2, 3, 5]:
    for acq in ['EHVI', 'MOUCB', 'MOGreedy']:

        df_i = pd.read_csv(f"../results/results_benchmark_dft_acq_{acq}_batch_{batch}_seed_0.csv")
        columns_to_drop = list(set(df_i.columns.values) - set(columns_to_keep))
        df_i.drop(columns=columns_to_drop, inplace=True)
        for seed_i in range(0, 5):
            df_j = pd.read_csv(f"../results/results_benchmark_dft_acq_{acq}_batch_{batch}_seed_{seed_i}.csv")
            df_j.drop(columns=columns_to_drop, inplace=True)
            df_i = df_i.append(df_j)

        df_i.to_csv(f"./dft_{acq}_{batch}_all.csv", index=False)

        df_av = df_i.groupby(['step', 'n_experiments']).agg([np.average])
        df_av['step'] = np.unique(df_i.step.values)
        df_av['n_experiments'] = np.unique(df_i.n_experiments.values)
        df_av.to_csv(f"./dft_{acq}_{batch}_avg.csv", index=False)

        df_min = df_i.groupby(['step', 'n_experiments']).agg([np.min])
        df_min['step'] = np.unique(df_i.step.values)
        df_min['n_experiments'] = np.unique(df_i.n_experiments.values)
        df_min.to_csv(f"./dft_{acq}_{batch}_min.csv", index=False)

        df_max = df_i.groupby(['step', 'n_experiments']).agg([np.max])
        df_max['step'] = np.unique(df_i.step.values)
        df_max['n_experiments'] = np.unique(df_i.n_experiments.values)
        df_max.to_csv(f"./dft_{acq}_{batch}_max.csv", index=False)



