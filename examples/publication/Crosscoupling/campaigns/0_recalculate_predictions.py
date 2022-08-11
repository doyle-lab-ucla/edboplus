

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil

for campaign in ['challenging_campaign_cvt', 'challenging_campaign_random', 'easy_campaign']:
    for round in range(1, 8):
        df = pd.read_csv(f"{campaign}/edbo_crosscoupling_photoredox_yield_ee_round{round}.csv")
        df.to_csv('optimization.csv', index=False)

        from edbo.plus.optimizer_botorch import EDBOplus

        filename = 'optimization.csv'

        regression_columns = df.columns.drop(['Ligand', 'priority']).values.tolist()

        opt = EDBOplus()
        opt.run(
            filename=filename,
            objectives=['yield', 'ee'],
            objective_mode=['max', 'max'],
            objective_thresholds=[None, None],
            batch=3,
            init_sampling_method='cvtsampling',
            columns_features=regression_columns
        )

        shutil.copy('pred_optimization.csv', f"{campaign}/predictions_{round}.csv")