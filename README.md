
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

(1) Open and edit config.json to choose your scope, your objectives, as well as other EDBO+ parameters. (more detail here)

(2) Run the following command, replacing \[FILENAME\] with the name of your experimental results CSV.

```
python edboplus.py [FILENAME]
```
If you don't have experimental results yet, EDBO+ will choose some initial conditions (determined by BATCH_SIZE) for you to try. 



#### **Note**: to run the notebook tutorials install JupyterLab

```
conda install jupyterlab
```
