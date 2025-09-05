import sys
import os
sys.path.append(os.path.abspath(os.path.join('..','..')))
from edbo.plus.optimizer_botorch import EDBOplus
import json

'''Command line interface for EDBO+
Usage: python edboplus.py [FILENAME]'''

# Load config file
config = open("config.json", "r")
config_data = json.load(config)

# Get filename passed as command line argument
try:
    exp_data_filename = sys.argv[1]
except IndexError:
    input("Please provide the name of your experimental data file, or the filename in which you want your initial samples to be written!")
    sys.exit()

# Feed parameters into top-level EDBO method
objective_pairs = config_data.get("OBJECTIVES").items()
objectives = [p[0] for p in objective_pairs]
objective_mode = [p[1] for p in objective_pairs]

EDBOplus().run(
    filename=exp_data_filename,  # Previously completed experiments (if none are done, suggestions will be written to this filename)
    scope = config_data.get("SCOPE"), # Reaction scope for the optimiser to search over 
    objectives=objectives,  # Objectives to be optimized.
    objective_mode=objective_mode,  # Maximise or minimise each respective objective
    batch=config_data.get("BATCH_SIZE"),  # Number of experiments in parallel that we want to perform in this round.
    columns_features=config_data.get("COLUMN_FEATURES"), # features to be included in the model.
    init_sampling_method=config_data.get("INITIAL_SAMPLING_METHOD")  # Method used to draw initial samples
)









