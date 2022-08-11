
import numpy as np
import pandas as pd

df_exp = pd.read_csv('./data/experiments_yield_and_cost.csv')


# Base features.
for i in ['base', 'ligand', 'solvent']:
    df_i = pd.read_csv(f"data/{i}_dft.csv")
    df_i.rename(columns={f"{i}_file_name": i}, inplace=True)
    df_exp = pd.merge(df_exp, df_i, on=i)

df_edbo = df_exp.copy(deep=True)
# Remove correlated features.
corr_matrix = df_edbo.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find features with correlation greater than 0.95.
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features
df_edbo.drop(to_drop, axis=1, inplace=True)

# Remove columns that have only one or two unique values.
extra_columns_to_remove = []
for column in df_edbo.columns.values:
    if len(np.unique(df_edbo[column].values)) <= 1:
        extra_columns_to_remove.append(column)
df_edbo.drop(extra_columns_to_remove, axis=1, inplace=True)

# Remove non numerical.
df_edbo_numeric = df_edbo.select_dtypes(include=np.number)
df_edbo_numeric.to_csv('./data/clean_dft.csv', index=0)
print(df_edbo_numeric)
