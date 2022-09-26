
from edbo.plus.optimizer_botorch import EDBOplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pareto
from edbo.plus.benchmark.multiobjective_benchmark import is_pareto
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# # Metrics.
# def get_pareto_points(objective_values):
#     """ Get pareto for the ground truth function.
#     NOTE: Assumes maximization."""
#     pareto_ground = pareto.eps_sort(tables=objective_values,
#                                     objectives=np.arange(2),
#                                     maximize_all=True)
#     idx_pareto = is_pareto(objectives=-objective_values)
#     return np.array(pareto_ground), idx_pareto

# def get_hypervolume(pareto_points, ref_mins):
#     """
#     Calculate hypervolume.
#     """     
#     pareto_torch = torch.Tensor(pareto_points)
#     hv = Hypervolume(ref_point=torch.Tensor(ref_mins))
#     hypervolume = hv.compute(pareto_Y=pareto_torch)
#     return hypervolume


# # Combinations of constraints tested in this example.
# # Columns that remain constant after EDBO suggest the best sample using batch=1.
set_constraints = [
    ['ligand'],
    ['ligand', 'base'],
    ['solvent', 'concentration', 'temperature'],    
]

# df_results = pd.DataFrame(columns=['seed', 'constraints', 
#                                    'n_exp', 'hypervolume'])

# for columns_to_constrain in set_constraints:
#     # Parameters.
#     batch_size = 5
#     # columns_to_constrain = ['solvent', 'concentration', 'temperature']  
#     n_rounds = 7 
#     n_seeds = 5   
#     # Load lookup tables.
#     df_hte = pd.read_csv('./data/experiments_yield_and_cost.csv')
#     # Get targets for hypervolume indicator.
#     targets_hte = np.zeros((len(df_hte), 2))
#     targets_hte[:, 0] = df_hte['yield'].to_numpy()
#     targets_hte[:, 1] = -df_hte['cost'].to_numpy()
#     worst_targets = np.min(targets_hte, axis=0)
#     pareto_ref = get_pareto_points(objective_values=targets_hte)[0]
#     hypervolume_ref = get_hypervolume(pareto_points=pareto_ref, ref_mins=worst_targets)

#     # Get columns names for regression and search space.
#     columns_search_space = df_hte.drop(columns=['yield', 'cost']).columns.tolist()
#     columns_regression = df_hte.drop(columns=['new_index', 'yield', 'cost']).columns.tolist()
#     df_full_space = df_hte[columns_search_space]

#     # Initialize optimization campaing.    
#     for seed in range(0, n_seeds):
#         n_exp = 0
#         df_full_space.to_csv('optimization.csv', index=False)
#         for round in range(0, n_rounds):
#             EDBOplus().run(
#                 filename='optimization.csv',
#                 seed=seed, 
#                 objectives=['yield', 'cost'],  
#                 objective_mode=['max', 'min'],  # Maximize yield but minimize cost.
#                 batch=1,  
#                 columns_features=columns_regression, # features to be included in the model.
#                 init_sampling_method='cvtsampling'  # initialization method.
#             )
            
#             df_opt = pd.read_csv('optimization.csv')
                    
#             # Initial optimization to obtain the best sample in the entire search space.
#             best_suggested_sample = df_opt.loc[0]    
#             df_reduced_space = df_opt.copy()
#             for col in columns_to_constrain:
#                 df_reduced_space = df_reduced_space[df_reduced_space[col] == best_suggested_sample[col]]

#             df_reduced_space.drop(columns=['yield', 'cost', 'priority'], inplace=True)
#             df_reduced_space.to_csv('optimization_reduced.csv', index=False)
            
#             EDBOplus().run(
#                 filename='optimization_reduced.csv',  # Previously generated scope.
#                 objectives=['yield', 'cost'],  # Objectives to be optimized.
#                 objective_mode=['max', 'min'],  # Maximize yield and ee but minimize side_product.
#                 batch=batch_size,  
#                 seed=seed,
#                 columns_features=columns_regression, # features to be included in the model.
#                 init_sampling_method='cvtsampling'  # initialization method.
#             )
            
#             df_opt_reduced = pd.read_csv('optimization_reduced.csv')    
            
#             idx_best_samples = df_opt_reduced['new_index'].values.tolist()[:batch_size]
#             print('Index best samples:', idx_best_samples)
#             df_opt = df_opt.sort_values(by='new_index')    
#             df_opt.reset_index(inplace=True)
#             df_opt.drop(columns=['index'], inplace=True)
            
#             for a in range(len(idx_best_samples)):        
#                 df_opt.at[idx_best_samples[a],'yield'] = df_hte.loc[idx_best_samples[a]]['yield']
#                 df_opt.at[idx_best_samples[a],'cost'] = df_hte.loc[idx_best_samples[a]]['cost']
#                 df_opt.at[idx_best_samples[a],'priority'] = 1
            
#             df_opt = df_opt.sort_values(by='priority', ascending=False)        
#             df_opt.to_csv('optimization.csv', index=False)
            
#             # Monitoring hypervolume.
#             df_train = df_opt[df_opt['yield'] != 'PENDING']
#             df_train['yield'] = copy.deepcopy(pd.to_numeric(df_train['yield']))
#             df_train['cost'] = copy.deepcopy(pd.to_numeric(df_train['cost']))
            
#             targets_train = np.zeros((len(df_train), 2))
#             targets_train[:, 0] = df_train['yield'].to_numpy()
#             targets_train[:, 1] = -df_train['cost'].to_numpy()
#             pareto_train = get_pareto_points(objective_values=targets_train)[0]
#             hypervolume_train = get_hypervolume(pareto_points=pareto_train, 
#                                             ref_mins=worst_targets)
#             hypervolume_explored = (hypervolume_train/hypervolume_ref) * 100
            
#             n_exp += batch_size
#             print(f"Number of samples: {n_exp}")        
#             print(f"Hypervolume: {hypervolume_explored}")
            
#             dict_results = {'seed': seed,
#                             'constraints': columns_to_constrain, 
#                             'n_exp': n_exp, 
#                             'hypervolume': hypervolume_explored}                    
#             df_results = df_results.append(dict_results, ignore_index=True)
#     df_results.to_csv('constraint_optimization_results.csv')


# Plot results.
df_results = pd.read_csv('constraint_optimization_results.csv')
colors = [ '#0343DF', '#FAC205', '#DC143C']
count = 0

mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.1
plt.rcParams['font.family'] = 'Helvetica'

fig, ax = plt.subplots(figsize=(4., 4.0), dpi=500, nrows=1, ncols=1)
    
for constraints in set_constraints:    
    # Get subset for constraints.
    constraints = str(constraints)
    df_constraint = df_results[df_results['constraints'] == constraints]    
    
    # Get average, max and min hypervolume explored at each step.
    df_avg = df_constraint.groupby(['n_exp']).agg([np.average])
    df_max = df_constraint.groupby(['n_exp']).agg([np.max])
    df_min = df_constraint.groupby(['n_exp']).agg([np.min])

    
    n_exp = np.unique(df_results['n_exp'].values).flatten()
    hypervol_avg = df_avg['hypervolume'].values.flatten()
    hypervol_max = df_max['hypervolume'].values.flatten()
    hypervol_min = df_min['hypervolume'].values.flatten()

    color = colors[count]

    ax.plot(n_exp, hypervol_avg, color=color, lw=2.5,
            label=str(constraints))
    ax.fill_between(x=n_exp,
                    y1=hypervol_avg,
                    y2=hypervol_max, color=color, alpha=0.3, lw=0.)
    ax.fill_between(x=n_exp,
                    y1=hypervol_min,
                    y2=hypervol_avg, color=color, alpha=0.3, lw=0.)
    ax.plot(n_exp, hypervol_min, color=color, alpha=1., lw=1., ls='--')
    ax.plot(n_exp, hypervol_max, color=color, alpha=1., lw=1., ls='--')
    ax.plot(n_exp, np.ones_like(n_exp)*100,
                dashes=[8, 4], color='black', linewidth=0.8)
    ax.scatter(n_exp, hypervol_avg, marker='o', s=0., color=color)
    count += 1

ax.set_xticks(np.arange(0, 120, 10))
ax.set_xlim(0, np.max(n_exp[:-1]))
ax.set_ylim(0, 100)
ax.set_xlabel('Number of experiments')
ax.set_ylabel('Hypervolume (%)')
plt.legend()
plt.savefig('./results_plots/optimization_constraints.svg', format='svg')
