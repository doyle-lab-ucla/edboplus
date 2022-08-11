
import shutil
import pandas as pd
import numpy as np
import os
from edbo.plus.benchmark.multiobjective_benchmark import Benchmark
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("darkgrid")
sns.set_context("talk")

# Benchmark filename
for batch in [1, 2, 3, 5]:
    for acq_i in ['EHVI']:
        for sampling_method in ['seed', 'lhs', 'cvtsampling']:
            budget = 60
            acq = acq_i
            seed = 1

            df_exp = pd.read_csv('./data/clean_dft.csv')
            sort_column = 'new_index'

            columns_regression = df_exp.columns
            columns_regression = columns_regression.drop([sort_column, 'yield', 'cost']).tolist()
            objectives = ['yield', 'cost']
            objective_modes = ['max', 'min']
            objective_thresholds = [None, None]
            print(f"Columns for regression: {columns_regression}")

            label_benchmark = f"benchmark_dft_acq_{acq}_batch_{batch}_seed_{seed}_init_sampling_{sampling_method}.csv"

            if not os.path.exists(f"./results/{label_benchmark}"):

                # Remove previous files
                if os.path.exists(label_benchmark):
                    os.remove(label_benchmark)

                if os.path.exists(f'pred_{label_benchmark}'):
                    os.remove(f'pred_{label_benchmark}')

                if os.path.exists(f'results_{label_benchmark}'):
                    os.remove(f'results_{label_benchmark}')

                bench = Benchmark(df_ground=df_exp,
                                  features_regression=columns_regression,
                                  objective_names=objectives,
                                  objective_modes=objective_modes,
                                  objective_thresholds=objective_thresholds,
                                  filename=label_benchmark,
                                  filename_results=f'results_{label_benchmark}',
                                  index_column=sort_column,
                                  acquisition_function=acq)

                bench.run(steps=int(budget/batch), batch=batch, seed=seed,
                          plot_ground=False,
                          plot_predictions=False, plot_train=False,
                          init_method=sampling_method)

                # Move results.
                if not os.path.exists('results'):
                    os.mkdir('results')
                shutil.move(label_benchmark, f'results/{label_benchmark}')
                shutil.move(f'pred_{label_benchmark}', f'results/pred_{label_benchmark}')
                shutil.move(f'results_{label_benchmark}', f'results/results_{label_benchmark}')

