
import itertools
import pandas as pd
import os
from pathlib import Path


def create_reaction_scope(components, directory='./', filename='reaction.csv',
                          check_overwrite=True, chunk_size=1000000):

    """
    Reaction scope generator. Pass components dictionary, each
    dictionary key contains a list of the choices for a given component.

    ----------------------------------------------------------------------
    Example:

    components = {'temperature': [30, 40, 50],
                  'solvent': ['THF', 'DMSO'],
                  'concentration': [0.1, 0.2, 0.3, 0.4, 0.5]}
    ----------------------------------------------------------------------

    ----------------------------------------------------------------------
    Note:
        - All non-numerical choices are encoded using a One-Hot-Encoder.
    ----------------------------------------------------------------------

    ----------------------------------------------------------------------
    Returns:
          A dataframe with name *{label}.csv* including the entire
          set of choices (reaction scope).
    ----------------------------------------------------------------------
    """

    msg = "You need to pass a dictionary for components. \n"
    assert type(components) == dict, msg

    print('Generating reaction scope...')
    wdir = Path(directory)
    csv_filename = wdir.joinpath(filename)
    # Ask to overwrite previous scope.

    if os.path.exists(csv_filename) and check_overwrite is True:
        overwrite = input('Scope already exists. Overwrite? Y = yes, N = no\n')
        if overwrite.lower() != 'y':
            return

    # Predict how large will the scope be.
    n_combinations = 0
    for key in list(components.keys()):
        if n_combinations == 0:
            n_combinations = len(components[key])
        else:
            n_combinations *= len(components[key])

    remainder = n_combinations % chunk_size

    # Generate initial scope.
    keys = components.keys()
    values = (components[key] for key in keys)

    if n_combinations > chunk_size:
        # Chunks if exceeding chunk_size:
        chunck_combination = []
        n_iterations = 0
        for comb in itertools.product(*values):
            chunck_combination.append(comb)
            if len(chunck_combination) >= chunk_size:
                df_scope = pd.DataFrame(chunck_combination)
                if n_iterations == 0:
                    df_scope.to_csv(csv_filename, index=False, mode='w',
                                    header=list(keys))
                else:
                    df_scope.to_csv(csv_filename, index=False, mode='a', header=False)
                chunck_combination = []
                n_iterations += 1
        if remainder > 0:  # Last iteration.
            df_scope = pd.DataFrame(chunck_combination)
            df_scope.to_csv(csv_filename, index=False, mode='a', header=False)
    else:
        scope = [dict(zip(keys, combination)) for combination in
                 itertools.product(*values)]
        df_scope = pd.DataFrame(scope)
        df_scope.to_csv(csv_filename, index=False, mode='w',
                        header=list(keys))

    return df_scope
