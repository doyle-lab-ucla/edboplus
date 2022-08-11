
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


for batch in [1, 2, 3, 5]:
    for acq_i in [
        'EHVI',
        'MOUCB',
        'MOGreedy'
    ]:
        for seed_i in np.arange(0, 5):
            budget = 30
            acq = acq_i
            seed = seed_i

            df_exp = pd.read_csv('./data/dataset_B2_DFT_clean.csv')
            df_exp['new_index'] = np.arange(0, len(df_exp.values))
            sort_column = 'new_index'

            # Select the features for the model.
            columns_regression = df_exp.columns
            columns_regression = columns_regression.drop('solvent')
            columns_regression = columns_regression.drop('ligand')

            columns_regression = columns_regression.drop([sort_column, 'objective_conversion', 'objective_selectivity']).tolist()
            objectives = ['objective_conversion', 'objective_selectivity']
            objective_modes = ['max', 'max']
            objective_thresholds = [None, None]
            print(f"Columns for regression: {columns_regression}")
            ######################

            label_benchmark = f"benchmark_dft_acq_{acq}_batch_{batch}_seed_{seed}.csv"

            if not os.path.exists(f"./results_dft/{label_benchmark}"):
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
                          plot_predictions=False,
                          plot_ground=False,
                          plot_train=False)

                # Move results.
                if not os.path.exists('results_dft'):
                    os.mkdir('results_dft')
                shutil.move(label_benchmark, f'results_dft/{label_benchmark}')
                shutil.move(f'pred_{label_benchmark}', f'results_dft/pred_{label_benchmark}')
                shutil.move(f'results_{label_benchmark}', f'results_dft/results_{label_benchmark}')

