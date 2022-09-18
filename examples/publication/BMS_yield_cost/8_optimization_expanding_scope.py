from edbo.plus.optimizer_botorch import EDBOplus
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_lookup = pd.read_csv('./data/experiments_yield_and_cost.csv')
df_large = pd.read_csv('./data/experiments_yield_and_cost.csv')

condition1 = df_large["ligand"].str.contains("CgMe-PPh")==False
condition2 = df_large["ligand"].str.contains("PPh3")==False
df_small = df_large[condition1 & condition2]

# Refereces for plots.
ref_best_yield_small_scope = np.max(df_small['yield'])
ref_best_cost_small_scope = np.min(df_small['cost'])

ref_best_yield_large_scope = np.max(df_large['yield'])
ref_best_cost_large_scope = np.min(df_large['cost'])

df_small.to_csv('./data/small_scope_lookup.csv', index=False)
df_large.to_csv('./data/large_scope_lookup.csv', index=False)

df_small.drop(columns=['yield', 'cost'], inplace=True)
df_large.drop(columns=['yield', 'cost'], inplace=True)

df_small.to_csv('./small_scope.csv', index=False)
df_large.to_csv('./large_scope.csv', index=False)

# Expand scope.
df_expand = df_large.copy()
condition1 = df_large["ligand"].str.contains("CgMe-PPh")==True
condition2 = df_large["ligand"].str.contains("PPh3")==True
df_expand = df_large[condition1 | condition2]
df_expand['priority'] = np.zeros(len(df_expand))
df_expand['yield'] = ['PENDING'] * len(df_expand)
df_expand['cost'] = ['PENDING'] * len(df_expand)

print('References:')
print('Small scope (best yield / best cost):', ref_best_yield_small_scope, ref_best_cost_small_scope)
print('Large scope (best yield / best cost):',ref_best_yield_large_scope, ref_best_cost_large_scope)

# Run optimization loops.
n_rounds_small = 6
n_round_large = 5
batch_size = 3
columns_regression = df_small.drop(columns=['new_index']).columns.tolist()

n_experiments = 0

track_results_dict = {
    'n_experiments': [],
    'best_yield': [],
    'best_cost': [],
    'max_ei_yield': [],
    'max_ei_cost': [],
    'max_uncertainty_yield': [],
    'max_uncertainty_cost': [],
    'avg_uncertainty_yield': [],
    'avg_uncertainty_cost': [],    
    }

collected_yields = []
collected_costs = []

for round in range(0, n_rounds_small):    
    EDBOplus().run(
        filename='small_scope.csv',  # Previously generated scope.
        objectives=['yield', 'cost'],  # Objectives to be optimized.
        objective_mode=['max', 'min'],  # Maximize yield and ee but minimize side_product.
        batch=batch_size,  # Number of experiments in parallel that we want to perform in this round.
        columns_features=columns_regression, # features to be included in the model.
        init_sampling_method='cvtsampling'  # initialization method.
    )
    
    n_experiments += batch_size
    # Update with experimental values (observations).
    df_results = pd.read_csv('small_scope.csv')    
    arg_lookup = df_results.loc[0:batch_size-1]['new_index'].values    
    
    for a in range(len(arg_lookup)):        
        df_results.at[a,'yield'] = df_lookup.loc[arg_lookup[a]]['yield']
        df_results.at[a,'cost'] = df_lookup.loc[arg_lookup[a]]['cost']
        collected_yields.append(df_lookup.loc[arg_lookup[a]]['yield'])
        collected_costs.append(df_lookup.loc[arg_lookup[a]]['cost'])
    
    df_results.to_csv('small_scope.csv', index=False)
    
    if round > 0:
        # Save all predicted values.
        df_pred = pd.read_csv('pred_small_scope.csv')
        max_ei_yield = np.max(df_pred['yield_expected_improvement'])
        max_ei_cost = np.max(df_pred['cost_expected_improvement'])
        max_uncertainty_yield = np.max((df_pred['yield_predicted_variance']))
        max_uncertainty_cost = np.max((df_pred['cost_predicted_variance']))
        avg_uncertainty_yield = np.average((df_pred['yield_predicted_variance']))
        avg_uncertainty_cost = np.average((df_pred['cost_predicted_variance']))                        
        best_yield = np.max(collected_yields)
        best_cost = np.min(collected_costs)        
        track_results_dict['n_experiments'].append(n_experiments)
        track_results_dict['best_yield'].append(best_yield)
        track_results_dict['best_cost'].append(best_cost)    
        track_results_dict['max_ei_yield'].append(max_ei_yield)                        
        track_results_dict['max_ei_cost'].append(max_ei_cost)                        
        track_results_dict['max_uncertainty_yield'].append(max_uncertainty_yield)                                
        track_results_dict['max_uncertainty_cost'].append(max_uncertainty_cost)                                
        track_results_dict['avg_uncertainty_yield'].append(avg_uncertainty_yield)                        
        track_results_dict['avg_uncertainty_cost'].append(avg_uncertainty_cost)                        

# Plot before expanding:
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

sns.scatterplot(
    x=np.array(track_results_dict['n_experiments']), 
    y=np.array(track_results_dict['max_ei_yield']), ax=ax[0][0], color='C1', s=100,
    zorder=100
    )
sns.scatterplot(
    x=track_results_dict['n_experiments'], 
    y=track_results_dict['max_ei_cost'], ax=ax[0][1], color='C1', s=100,
    zorder=100
    )
sns.scatterplot(
    x=track_results_dict['n_experiments'], 
    y=track_results_dict['best_yield'], ax=ax[1][0], color='C1',  s=100,
    zorder=100
    )
sns.scatterplot(    
    x=track_results_dict['n_experiments'], 
    y=track_results_dict['best_cost'], ax=ax[1][1], color='C1',s=100,
    zorder=100
    )

ax[0][0].set_xlabel('Number of experiments')
ax[0][1].set_xlabel('Number of experiments')
ax[1][0].set_xlabel('Number of experiments')
ax[1][1].set_xlabel('Number of experiments')
ax[0][0].set_ylabel('Max EI (yield)')
ax[0][1].set_ylabel('Max EI (cost)')
ax[1][0].set_ylabel('Highest yield found')
ax[1][1].set_ylabel('Lowest cost found')


# Expand scope:
df_small = pd.read_csv('small_scope.csv')
df_expand = df_expand.append(df_small)
df_expand.sort_values(by=['priority'], ascending=False, inplace=True)
df_expand.to_csv('expanded_scope.csv', index=False)

n_experiments -= batch_size

# Keep optimizing after expanding.
for round in range(0, n_round_large):    
    EDBOplus().run(
        filename='expanded_scope.csv',  # Previously generated scope.
        objectives=['yield', 'cost'],  # Objectives to be optimized.
        objective_mode=['max', 'min'],  # Maximize yield and ee but minimize side_product.
        batch=batch_size,  # Number of experiments in parallel that we want to perform in this round.
        columns_features=columns_regression, # features to be included in the model.
        init_sampling_method='cvtsampling'  # initialization method.
    )
    
    n_experiments += batch_size
    # Update with experimental values (observations).
    df_results = pd.read_csv('expanded_scope.csv')    
    arg_lookup = df_results.loc[0:batch_size-1]['new_index'].values    
    
    for a in range(len(arg_lookup)):        
        df_results.at[a,'yield'] = df_lookup.loc[arg_lookup[a]]['yield']
        df_results.at[a,'cost'] = df_lookup.loc[arg_lookup[a]]['cost']
        collected_yields.append(df_lookup.loc[arg_lookup[a]]['yield'])
        collected_costs.append(df_lookup.loc[arg_lookup[a]]['cost'])
    
    df_results.to_csv('expanded_scope.csv', index=False)
    
    if round > 0:
        # Save all predicted values.
        df_pred = pd.read_csv('pred_expanded_scope.csv')
        max_ei_yield = np.max(df_pred['yield_expected_improvement'])
        max_ei_cost = np.max(df_pred['cost_expected_improvement'])
        max_uncertainty_yield = np.max((df_pred['yield_predicted_variance']))
        max_uncertainty_cost = np.max((df_pred['cost_predicted_variance']))
        avg_uncertainty_yield = np.average((df_pred['yield_predicted_variance']))
        avg_uncertainty_cost = np.average((df_pred['cost_predicted_variance']))                        
        best_yield = np.max(collected_yields)
        best_cost = np.min(collected_costs)        
        track_results_dict['n_experiments'].append(n_experiments)
        track_results_dict['best_yield'].append(best_yield)
        track_results_dict['best_cost'].append(best_cost)    
        track_results_dict['max_ei_yield'].append(max_ei_yield)                        
        track_results_dict['max_ei_cost'].append(max_ei_cost)                        
        track_results_dict['max_uncertainty_yield'].append(max_uncertainty_yield)                                
        track_results_dict['avg_uncertainty_yield'].append(avg_uncertainty_yield)                        
        track_results_dict['avg_uncertainty_cost'].append(avg_uncertainty_cost)
        
        
sns.scatterplot(
    x=np.array(track_results_dict['n_experiments']), 
    y=np.array(track_results_dict['max_ei_yield']), ax=ax[0][0], color='C0', s=95,
    zorder=10
    )
sns.scatterplot(
    x=track_results_dict['n_experiments'], 
    y=track_results_dict['max_ei_cost'], ax=ax[0][1], color='C0', s=95,
    )
sns.scatterplot(
    x=track_results_dict['n_experiments'], 
    y=track_results_dict['best_yield'], ax=ax[1][0], color='C0',  s=95,
    zorder=10
    )
sns.scatterplot(    
    x=track_results_dict['n_experiments'], 
    y=track_results_dict['best_cost'], ax=ax[1][1], color='C0',s=95,
    zorder=10
    )

plt.tight_layout()
plt.savefig('./results_plots/expand_scope.svg', format='svg')
plt.show()

    
    
    
    

    


