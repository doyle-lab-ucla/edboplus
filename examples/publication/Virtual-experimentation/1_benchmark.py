
import shutil
from edbo.plus.benchmark.multiobjective_benchmark import Benchmark
import os
import numpy as np
import pandas as pd


#######################
# Benchmark inputs
budget = 30

acq = 'EHVI'
seed = 1
for sampling_method in ['seed', 'lhs', 'cvtsampling']:
    for batch in [1, 2, 3, 5]:
        df_exp = pd.read_csv('./data/data.csv')
        df_exp['new_index'] = np.arange(0, len(df_exp.values))
        sort_column = 'new_index'

        # Select the features for the model.
        columns_regression = ['Temperature', 'Volume', 'D',
                              'SM2',
                              'W',
                              'Mixing',
                              'Time',
                              'WB'
                              ]

        # Select objectives.
        objectives = ['P', 'I1']
        objective_modes = ['max', 'min']
        objective_thresholds = [None, None]
        print(f"Columns for regression: {columns_regression}")

        label_benchmark = f"benchmark_acq_{acq}_batch_{batch}_{sampling_method}.csv"

        # Remove previous files.
        if os.path.exists(label_benchmark):
            os.remove(label_benchmark)

        if os.path.exists(f'pred_{label_benchmark}'):
            os.remove(f'pred_{label_benchmark}')

        if os.path.exists(f'results_{label_benchmark}'):
            os.remove(f'results_{label_benchmark}')

        bench = Benchmark(
            df_ground=df_exp,
            features_regression=columns_regression,
            objective_names=objectives,
            objective_modes=objective_modes,
            objective_thresholds=objective_thresholds,
            filename=label_benchmark,
            filename_results=f'results_{label_benchmark}',
            index_column=sort_column,acquisition_function=acq
        )

        bench.run(
            steps=int(budget/batch), batch=batch, seed=seed,
            init_method=sampling_method,
            plot_train=False, plot_predictions=False
        )

        if not os.path.exists('results'):
            os.mkdir('results')

        shutil.move(label_benchmark, f'results/{label_benchmark}')
        shutil.move(f'pred_{label_benchmark}', f'results/pred_{label_benchmark}')
        shutil.move(f'results_{label_benchmark}', f'results/results_{label_benchmark}')
