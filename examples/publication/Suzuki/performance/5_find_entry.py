
import pandas as pd


df = pd.read_csv('../data/dataset_B1.csv')

c_ligand, c_base, c_leq, c_solvent = 'SPhos', 'NaOH(aq.)', 0.0625, 'DMF'

c_ligand, c_base, c_leq, c_solvent = 'P(Ph)3', 'KOAc', 0.125, 'MeOH'

c_ligand, c_base, c_leq, c_solvent = 'P(Cy)3', 'Cs2CO3(aq.)', 0.125, 'MeOH'

c_ligand, c_base, c_leq, c_solvent = 'P(Ph)3', 'NaOH(aq.)', 0.125, 'MeOH'

c_ligand, c_base, c_leq, c_solvent = 'P(Ph)3', 'CsF(aq.)', 0.125, 'MeCN'

df_new = df[(df['ligand'] == c_ligand) & (df['base'] == c_base) & (df['solvent'] == c_solvent)]

print(df_new)
