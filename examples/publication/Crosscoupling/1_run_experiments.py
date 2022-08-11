
# Cross-coupling photoredox.
import pandas as pd
from edbo.plus.optimizer_botorch import EDBOplus

filename = 'edbo_crosscoupling_photoredox_yield_ee.csv'

df_to_opt = pd.read_csv(filename)
regression_columns = df_to_opt.columns.drop(['Ligand', 'priority']).values.tolist()

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
