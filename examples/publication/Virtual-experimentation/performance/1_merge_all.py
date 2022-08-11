import pandas as pd
import numpy as np

batch = 1

objective_1 = 'P'
objective_2 = 'I1'

columns_to_keep = ['step', 'n_experiments',
                   'dmaximin_tradeoff', 'hypervolume completed (%)',
                   f'MAE_{objective_1}', f"MAE_{objective_2}",
                   f'RMSE_{objective_1}', f'RMSE_{objective_2}',
                   f'R2_{objective_1}', f'R2_{objective_2}',
                   f'{objective_1}_best', f'{objective_2}_best'
                   ]

acq = 'EHVI'
for sampling in ['seed', 'lhs', 'cvtsampling']:
    df_i = pd.read_csv(f"../results/results_benchmark_acq_{acq}_batch_{batch}_{sampling}.csv")
    columns_to_drop = list(set(df_i.columns.values) - set(columns_to_keep))
    df_i.drop(columns=columns_to_drop, inplace=True)

    df_i.to_csv(f"./{sampling}_all.csv", index=False)

    df_av = df_i.groupby(['step', 'n_experiments']).agg([np.average])
    df_av['step'] = np.unique(df_i.step.values)
    df_av['n_experiments'] = np.unique(df_i.n_experiments.values)
    df_av.to_csv(f"./{sampling}_avg.csv", index=False)

    df_min = df_i.groupby(['step', 'n_experiments']).agg([np.min])
    df_min['step'] = np.unique(df_i.step.values)
    df_min['n_experiments'] = np.unique(df_i.n_experiments.values)
    df_min.to_csv(f"./{sampling}_min.csv", index=False)

    df_max = df_i.groupby(['step', 'n_experiments']).agg([np.max])
    df_max['step'] = np.unique(df_i.step.values)
    df_max['n_experiments'] = np.unique(df_i.n_experiments.values)
    df_max.to_csv(f"./{sampling}_max.csv", index=False)



