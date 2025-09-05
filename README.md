
# <img src="EDBOLogo.png" width="190">

## **EDBO+**. Bayesian reaction optimization as a tool for chemical synthesis

WebApp: https://www.edbowebapp.com

**Reference:** Garrido Torres, Jose A.; Lau, Sii Hong; Anchuri, Pranay; Stevens, Jason M.; Tabora, Jose E.; Li, Jun; Borovika, Alina; Adams, Ryan P.; Doyle, Abigail G. "A Multi-Objective Active Learning Platform and Web App for Reaction Optimization".

**DOI:** 

10.26434/chemrxiv-2022-cljcp

10.1021/jacs.2c08592

**Links**:
[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/62f6966269f3a5df46b5584b), 
[JACS](https://pubs.acs.org/doi/full/10.1021/jacs.2c08592)


<br>

---

<br>

### Installation:

<br>

(1) Create anaconda environment:

```
conda create --name edbo_env python=3.9
```

(2) Activate conda environment:

```
conda activate edbo_env
```

(3) Install EDBO+ dependencies:

```
pip install -e .
```

<br>

---

<br>

### Usage:

(1) Open and edit **config.json** to choose your reaction scope, your objectives, and other EDBO+ parameters. Below is a description of each parameter:
- **[SCOPE]**: This contains the set of reaction variables, alongside possible values that you wish to consider. These can be numerical, for instance reactant concentration, or categorical, for instance catalyst type. Based on this scope, EDBO+ will generate and evaluate the acquisition function on all possible combination of values for the variables specified by the scope.

- **[OBJECTIVES]**: This contains each objective you with to optimise (e.g. yield), as well as the optimisation mode (either "min" or "max"). 

- **[BATCH_SIZE]**: This is the number of different conditions you wish EDBO+ to suggest. 

- **[COLUMN_FEATURES]**: This is the list of features (i.e. reaction variables) that you wish to use to train the surrogate model, in order to predict the objectives. When set to "all", the surrogate model will be trained on all features (columns in the CSV, other than the objectives).

- **[INITIAL_SAMPLING_METHOD]**: This is the method that is used to pick the method used to generate initial samples to try (based on the provided scope), in the case where no experimental data have been collected yet. Choices are: 
    - 'random' : Random seed (as implemented in Pandas).
    - 'lhs' : LatinHypercube sampling.
    - 'cvt' : CVT sampling.

(2) Ensure that your CSV containing the conditions you have tried, as well as the experimentally-determined values for the objectives (i.e. the experimental results) are in this directory. Then, **run the following command**, replacing \[FILENAME\] with the name of your experimental results CSV.

```
python edboplus.py [FILENAME]
```
If you don't have experimental results yet, just pick a filename you like and EDBO+ will choose some initial conditions (the number of which is determined by BATCH_SIZE), create a CSV file, and write the initial conditions sampled to the file. 

(3) After running EDBO+, there will be new (suggested) conditions written into the experimental results CSV, with each of the objective fields containing the placeholder variable "PENDING". Now, you can either **run EDBO+ again** with a different scope or parameters (the old suggested conditions will be deleted), or **complete the suggested experiments** and fill in the objective fields, then run EDBO+ again to get new suggested conditions, with the surrogate model updated by the experiments you just completed. 

<br>

---

<br>

### **Note**: to run the notebook tutorials install JupyterLab

```
conda install jupyterlab
```
