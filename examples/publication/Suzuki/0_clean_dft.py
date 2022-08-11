import os

import numpy as np
import pandas as pd


df_dft = pd.read_csv('data/dataset_B2.csv')

# # Remove correlated features.
corr_matrix = df_dft.corr().abs()
# Select upper triangle of correlation matrix.
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find features with correlation greater than 0.95.
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features
df_dft.drop(to_drop, axis=1, inplace=True)

# Remove columns that have only one or two unique values.
extra_columns_to_remove = []
for column in df_dft.columns.values:
    if len(np.unique(df_dft[column].values)) <= 1:
        extra_columns_to_remove.append(column)
df_dft.drop(extra_columns_to_remove, axis=1, inplace=True)

# Store SMILES.
solvent_ohe = df_dft['solvent'].values
base_ohe = df_dft['base'].values
ligand_ohe = df_dft['ligand'].values

# Remove non numerical.
df_edbo_numeric = df_dft.select_dtypes(include=np.number)

# Add back OHE features.
df_edbo_numeric.insert(1, "solvent", solvent_ohe, False)
df_edbo_numeric.insert(1, "base", base_ohe, False)
df_edbo_numeric.insert(1, "ligand", ligand_ohe, False)

df_edbo_numeric.to_csv('./data/dataset_B2_DFT_clean.csv', index=0)
