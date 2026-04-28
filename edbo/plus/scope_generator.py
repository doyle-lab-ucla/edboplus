
import itertools
import pandas as pd
import os
from pathlib import Path

# Bundled solvent PCA lookup — relative to this file's location in the repo.
_SOLVENT_LOOKUP_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'Solvent_PC_clean.csv'
_SOLVENT_PCA_FEATURES = ['PC1', 'PC2', 'PC3', 'PC4']


def _load_solvent_lookup():
    """Return the bundled Solvent_PC_clean DataFrame, or None if not found."""
    if _SOLVENT_LOOKUP_PATH.exists():
        return pd.read_csv(_SOLVENT_LOOKUP_PATH)
    return None


def _auto_solvent_encodings(components, user_encodings):
    """
    Detect component columns whose values all appear in the bundled solvent
    lookup table and build encoding dicts for them automatically.
    Columns already covered by *user_encodings* are skipped.
    """
    lookup = _load_solvent_lookup()
    if lookup is None:
        return {}

    known = set(lookup['Name'])
    auto = {}
    for col, values in components.items():
        if user_encodings and col in user_encodings:
            continue
        str_vals = [v for v in values if isinstance(v, str)]
        if str_vals and all(v in known for v in str_vals):
            auto[col] = {
                'file': lookup,          # pass DataFrame directly — no re-read
                'key': 'Name',
                'features': _SOLVENT_PCA_FEATURES,
            }
    return auto


def create_reaction_scope(components, directory='./', filename='reaction.csv',
                          check_overwrite=True, encodings=None):

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
        - All non-numerical choices are encoded using a One-Hot-Encoder
          unless an encoding is provided via the *encodings* parameter.
        - Solvent columns are detected automatically: if every value in a
          component column matches a name in the bundled
          data/Solvent_PC_clean.csv lookup table, PC1–PC4 encodings are
          applied without any extra configuration.  Pass those column names
          to exclude_columns in EDBOplus.run() so the model uses the
          numeric PC features rather than the label string.
    ----------------------------------------------------------------------

    ----------------------------------------------------------------------
    encodings (optional): dict
        Maps a component name to a lookup table of numeric features.
        Useful for replacing one-hot encoding with pre-computed descriptors
        such as PCA coordinates.

        Example:
            encodings = {
                'solvent': {
                    'file': 'DATA/Solvent_PC_clean.csv',  # path or DataFrame
                    'key': 'Name',                         # column to match on
                    'features': ['PC1', 'PC2', 'PC3', 'PC4'],  # columns to join in (PC1-PC4 = 93.9% variance)
                }
            }

        The original label column (e.g. 'solvent') is kept in the CSV for
        readability. Pass it to *exclude_columns* in EDBOplus.run() so the
        model uses only the numeric feature columns.
    ----------------------------------------------------------------------

    ----------------------------------------------------------------------
    Returns:
          A dataframe with name *{label}.csv* including the entire
          set of choices (reaction scope).
    ----------------------------------------------------------------------
    """

    msg = "You need to pass a dictionary for components. \n"
    assert type(components) == dict, msg

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

    # Generate initial scope.
    keys = components.keys()
    values = (components[key] for key in keys)

    scope = [dict(zip(keys, combination)) for combination in
                itertools.product(*values)]
    df_scope = pd.DataFrame(scope)

    # Auto-detect solvent columns and add PCA encodings for them.
    auto = _auto_solvent_encodings(components, encodings)
    if auto:
        print(
            f"Auto-applied PCA solvent encodings (PC1–PC4) for: {list(auto.keys())}.\n"
            f"  Pass these column names to exclude_columns in run() so the model\n"
            f"  uses the numeric PC features instead of the label string."
        )
    merged_encodings = {**auto, **(encodings or {})}

    # Join numeric feature columns from lookup tables.
    if merged_encodings:
        for col, enc in merged_encodings.items():
            lookup = pd.read_csv(enc['file']) if isinstance(enc['file'], str) else enc['file']
            missing = set(components[col]) - set(lookup[enc['key']])
            if missing:
                raise ValueError(
                    f"The following values in '{col}' were not found in the lookup table: {missing}"
                )
            lookup_subset = lookup[[enc['key']] + enc['features']].rename(
                columns={enc['key']: col}
            )
            df_scope = df_scope.merge(lookup_subset, on=col, how='left')

    df_scope.to_csv(csv_filename, index=False, mode='w')

    return df_scope, n_combinations
