import os
from pathlib import Path
import random
import sys
import warnings
import itertools

from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import \
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions import \
    NondominatedPartitioning
from idaes.surrogate.pysmo.sampling import LatinHypercubeSampling, CVTSampling
import numpy as np
from ordered_set import OrderedSet
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import torch

from .utils import EDBOStandardScaler
from .model import build_and_optimize_model

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class EDBOplus:

    def __init__(self):

        self.objective_names = []
        self.predicted_mean = []
        self.predicted_variance = []
        self.ei = []
    
    @staticmethod
    def scope_on_the_fly(components, already_done):
        '''Generate a scope on the fly, automatically removing
        experiments that were already completed.'''
        keys = components.keys()
        values = (components[key] for key in keys)
        scope = [dict(zip(keys, combination)) for combination in
                    itertools.product(*values)]
        df_scope = pd.DataFrame(scope)
        # No experiments have been done
        features = set(df_scope.columns)
        if already_done.empty:
            return df_scope
        else:
            # Get features to drop
            to_drop = [col for col in already_done.columns if col not in features]
            temp = already_done.drop(columns = to_drop)
            merged = df_scope.merge(temp.drop_duplicates(), how='left', indicator=True)
            return merged[merged['_merge'] == 'left_only'].drop(columns=["_merge"])


    @staticmethod
    def _init_sampling(df, batch, sampling_method, seed):

        np.random.seed(seed)
        random.seed(seed)
        numeric_cols = df._get_numeric_data().columns
        ohe_columns = list(OrderedSet(df.columns) - OrderedSet(numeric_cols))
        if len(ohe_columns) > 0:
            print(f"The following columns are categorical and will be encoded"
                  f" using One-Hot-Encoding: {ohe_columns}")
        # OHE encoding categorical variables
        df_sampling = pd.get_dummies(df, prefix=ohe_columns,
                                     columns=ohe_columns, drop_first=True, dtype=np.float64)
        
        class HiddenPrints:
            '''Suppresses idaes output to stdout'''
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')

            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout

        # Order df according to initial sampling method (random samples).
        with HiddenPrints():
            idaes = None
            if sampling_method == 'random':
                samples = df_sampling.sample(n=batch, random_state=seed)
            elif sampling_method.lower() == 'lhs':
                idaes = LatinHypercubeSampling(df_sampling, batch, sampling_type="selection")
            elif sampling_method.lower() == 'cvt':
                idaes = CVTSampling(df_sampling, batch, sampling_type="selection")

            if idaes is not None:
                samples = idaes.sample_points()
            
            # Sometimes the LHS or CVT sampling methods return less samples than requested. Add random samples in this case.
            additional_samples = None
            if len(samples) < batch:
                additional_samples = df.sample(n=batch-len(samples), random_state=seed, replace=True)
                additional_samples = additional_samples.reset_index(drop=True)
            # Add the additional samples to the samples dataframe. If some of the additional_samples are already in samples, generate new ones until the batch size is reached.
            extra_seed = 1
            while len(samples) < batch:
                samples = pd.concat([samples,additional_samples]).drop_duplicates(ignore_index=True)
                additional_samples = df.sample(n=batch-len(samples), random_state=seed+extra_seed, replace=True)
                extra_seed +=1

        # Get index of the best samples according to the random sampling method.
        df_sampling_matrix = df_sampling.to_numpy()
        samples_drawn = []

        for sample in samples.to_numpy():
            d_i = cdist([sample], df_sampling_matrix, metric='cityblock')
            samples_drawn.append(np.argmin(d_i))
            
        print(f"Generated {len(samples)} initial samples using {sampling_method} sampling (seed = {seed})")
        return samples_drawn
    
    def _model_run(self, data, df_train_x,  df_test_x, df_train_y, batch,
                   objective_mode, objective_thresholds, seed,
                   scaler_x, scaler_y, acquisition_function):
        """
        Runs the surrogate machine learning model.
        Returns the indices of the experiments chosen by the optimiser.
        """

        # Check number of objectives.
        n_objectives = len(df_train_y.columns.values)

        scaler_x.fit(df_train_x.to_numpy())
        init_train = scaler_x.transform(df_train_x.to_numpy())
        test_xnp = scaler_x.transform(df_test_x.to_numpy())
        test_x = torch.tensor(test_xnp.tolist()).double().to(**tkwargs)
        y = df_train_y.astype(float).to_numpy()  # not scaled.

        individual_models = []
        for i in range(0, n_objectives):
            if objective_mode[i].lower() == 'min':
                y[:, i] = -y[:, i]
        y = scaler_y.fit_transform(y)

        print("Generating surrogate model...")
        for i in range(0, n_objectives):
            train_x = torch.tensor(init_train).to(**tkwargs).double()
            train_y = np.array(y)[:, i]
            train_y = (np.atleast_2d(train_y).reshape(len(train_y), -1))
            train_y_i = torch.tensor(train_y.tolist()).to(**tkwargs).double()

            gp, likelihood = build_and_optimize_model(train_x=train_x, train_y=train_y_i,)

            model_i = SingleTaskGP(train_X=train_x, train_Y=train_y_i,
                                   covar_module=gp.covar_module, likelihood=likelihood)
            individual_models.append(model_i)

        print("Model generated!")

        # Reference point is the minimum seen so far.
        ref_mins = np.min(y, axis=0)
        if objective_thresholds is None:
            ref_point = torch.tensor(ref_mins).double().to(**tkwargs)
        else:
            ref_point = np.zeros(n_objectives)
            for i in range(0, n_objectives):
                if objective_thresholds[i] is None:
                    ref_point[i] = ref_mins[i]
                else:
                    ref_point[i] = objective_thresholds[i]
                    if objective_mode[i].lower() == 'min':
                        ref_point[i] = -ref_point[i]
            # Scale.
            ref_point = scaler_y.transform(np.array([ref_point]))
            # Loop again.
            for i in range(0, n_objectives):
                if objective_thresholds[i] is None:
                    ref_point[0][i] = ref_mins[i]
            ref_point = torch.tensor(ref_point[0]).double().to(**tkwargs)

        if len(data.values) > 100000:
            sobol_num_samples = 64
        elif len(data.values) > 50000:
            sobol_num_samples = 128
        elif len(data.values) > 10000:
            sobol_num_samples = 256
        else:
            sobol_num_samples = 512

        y_torch = torch.tensor(y).to(**tkwargs).double()

        if self.acquisition_sampler == 'IIDNormalSampler':
            sampler = IIDNormalSampler(num_samples=sobol_num_samples, collapse_batch_dims=True, seed=seed)
        if self.acquisition_sampler == 'SobolQMCNormalSampler':
            sampler = SobolQMCNormalSampler(num_samples=sobol_num_samples, collapse_batch_dims=True, seed=seed) 

        print ("Optimizing acqusition function...")

        surrogate_model = None

        if acquisition_function.lower() == 'ehvi':

            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=y_torch)
            
            surrogate_model = ModelListGP(*individual_models)
            individual_models = []  # empty to reuduce memory
            
            EHVI = qExpectedHypervolumeImprovement(
                model=surrogate_model, sampler=sampler,
                ref_point=ref_point,  # use known reference point
                partitioning=partitioning
            )

            acq_result = optimize_acqf_discrete(
                acq_function=EHVI,
                choices=test_x,
                q=batch,
                unique=True
            )


        if acquisition_function.lower() == 'noisyehvi':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acq_fct = None
                if n_objectives > 1:  # NOTE: NoisyEHVI fails in case of n_objectives = 1 --> added that it uses EI in this case
                    surrogate_model = ModelListGP(*individual_models)
                    train_x = torch.tensor(init_train).to(**tkwargs).double()
                    acq_fct = qNoisyExpectedHypervolumeImprovement(
                        model=surrogate_model, sampler=sampler,
                        ref_point=ref_point,
                        alpha = 0.0,
                        incremental_nehvi=True, X_baseline=train_x, prune_baseline=True
                    )
                else:
                    surrogate_model = individual_models[0]
                    best_value = y_torch.max()
                    acq_fct = qExpectedImprovement(
                        model = surrogate_model, 
                        best_f = best_value,
                        sampler = sampler
                    )
                
                acq_result = optimize_acqf_discrete(
                    acq_function=acq_fct,
                    choices=test_x,
                    q=batch,
                    unique=True
                )

        best_samples = scaler_x.inverse_transform(acq_result[0].detach().cpu().numpy())
        print('Acquisition function optimized.')

        # Get predictions in chunks.
        chunk_size = 1000
        n_chunks = (len(data.values) // chunk_size) + 1

        self.predicted_mean = np.zeros(shape=(len(data.values), n_objectives))
        self.predicted_variance = np.zeros(shape=(len(data.values), n_objectives))
        self.ei = np.zeros(shape=(len(data.values), n_objectives))

        observed_raw_values = df_train_y.astype(float).to_numpy()

        for i in range(0, len(data.values), n_chunks):
            vals = data.values[i:i+n_chunks]
            data_tensor = torch.tensor(scaler_x.transform(vals)).double().to(**tkwargs)
            preds = surrogate_model.posterior(X=data_tensor)
            self.predicted_mean[i:i+n_chunks] = scaler_y.inverse_transform(preds.mean.detach().cpu().numpy())
            self.predicted_variance[i:i+n_chunks] = scaler_y.inverse_transform_var(preds.variance.detach().cpu().numpy())

            for j in range(0, len(objective_mode)):
                maximizing = False
                if objective_mode[j] == 'max':
                    maximizing = True
                self.ei[i:i+n_chunks, j] = self.expected_improvement(
                    train_y=observed_raw_values[:, j],
                    mean=self.predicted_mean[i:i+n_chunks, j],
                    variance=self.predicted_variance[i:i+n_chunks, j],
                    maximizing=maximizing
                )

        print('Predictions and expected improvement obtained.')

        # Flip predictions if needed.
        for i in range(0, len(objective_mode)):
            if objective_mode[i] == 'min':
                self.predicted_mean[:, i] = -self.predicted_mean[:, i]

        # Rescale samples.
        all_samples = data.values
        chosen_sample_indices = []

        # Find best samples in data.
        # NOTE: Here, the best samples are in vector form. This iterates through the data and returns
        # the index of the entry that possess minimum distance between the 'best samples' and the entry
        for sample in best_samples:
            d_i = cdist([sample], all_samples, metric='cityblock')
            chosen_sample_indices.append(np.argmin(d_i))

        return chosen_sample_indices

    def expected_improvement(self, train_y, mean, variance,
                             maximizing=False):
        """ expected_improvement
        Expected improvement acquisition function.
        Arguments:
        ----------
            mean: Numpy array.
                predicted mean of the Gaussian Process.
            variance: Numpy array.
                predicted variance of the Gaussian Process.
            train_y: Numpy array.
                Numpy array that contains the values of previously observed train targets.
            maximizing: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
        """

        sigma = variance ** 0.5

        if maximizing:
            loss_optimum = np.max(train_y)
        else:
            loss_optimum = np.min(train_y)

        scaling_factor = (-1) ** (not maximizing)

        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mean - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mean - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] = 0.0

        return expected_improvement    

    def run(self,
            objectives, objective_mode, scope, objective_thresholds=None,
            directory='.', filename='reaction.csv',
            columns_features='all',
            batch=5, init_sampling_method='cvt', seed=0,
            scaler_features=MinMaxScaler(),
            scaler_objectives=EDBOStandardScaler(),
            acquisition_function='NoisyEHVI',
            acquisition_function_sampler='SobolQMCNormalSampler'):

        """
        Parameters
        ----------
        objectives: list
            list of string containing the name for each objective.
            Example:
                objectives = ['yield', 'cost', 'impurity']

        objective_mode: list
            list to select whether the objective should be maximized or minimized.
            Examples:
                A) Example for single-objective optimization:
                    objective_mode = ['max']
                B) Example for multi-objective optimization:
                    objective_mode = ['max', 'min', 'min']
        
        scope: dictionary[string, list]
            dictionary specifying the reaction scope, where each key corresponds to a reaction variable
            (e.g. concentration, solvent type, catalyst type) and each value correspond to all possible
            conditions you want to evaluate the acquisition function with. For instance, the corresponding
            value to "concentration" could be [0.05, 0.1, 0.15, 0.2].

        objective_thresholds: list
            List of worst case values for each objective.
            Example:
                objective_threshold = [50.0, 10.0, 10.0]

        columns_features: list
            List containing the names of the columns to be included in the regression model. By default set to
            'all', which means the algorithm will automatically select all the columns that are not in
            the *objectives* list.

        batch: int
            Number of experiments that you want to run in parallel. For instance *batch = 5* means that you
            will run 5 experiments in each EDBO+ run. You can change this number at any stage of the optimization,
            so don't worry if you change  your mind after creating or initializing the reaction scope.

        directory: string
            name of the directory to save the results of the optimization.

        filename: string
            Name of the file to save a *csv* with the priority list. If *get_predictions=True* EDBO+ will automatically
            save a second file including the predictions (*pred_filename.csv*).

        init_sampling_method: string:
            Method for selecting the first samples in the scope (in absence)  Choices are:
            - 'random' : Random seed (as implemented in Pandas).
            - 'lhs' : LatinHypercube sampling.
            - 'cvt' : CVT sampling.

        scaler_features: sklearn class
            sklearn.preprocessing class for transforming the features.
            Example:
                sklearn.preprocessing.MinMaxScaler()

        scaler_objectives: sklearn class
            sklearn.preprocessing class for transforming the objective values.
            Examples:
                - sklearn.preprocessing.StandardScaler()
            Default:
                EDBOStandardScaler()

        seed: int
            Seed for the random initialization.

        acquisition_function_sampler: string
            Options are: 'SobolQMCNormalSampler' or 'IIDNormalSampler'.

        """

        wdir = Path(directory)
        csv_filename = wdir.joinpath(filename)
        torch.manual_seed(seed=seed)
        np.random.seed(seed)
        self.acquisition_sampler = acquisition_function_sampler

        # 1. Safe checks.
        self.objective_names = objectives

        # Ensure that all column features are actually present in the scope. If not, then they will be removed
        if columns_features != 'all':
            scope_features = set(scope.keys())
            columns_features = [feature for feature in columns_features if feature in scope_features]

        # Check that each variable has a nonzero number of possible values trialed in the scope
        if any(not value for value in scope.values()):
            raise ValueError("Error, one of the reaction variables have no possible values! Please check config.json!")

        # 2. Load training data, if it exists
        try:
            training_df = pd.read_csv(f"{csv_filename}")
            training_df = training_df.dropna(axis='columns', how='all')
            # Strip entries where a particular y value is 'PENDING' from the training dataframe 
            for obj in objectives:
                training_df = training_df[training_df[obj] != "PENDING"].copy()
        except FileNotFoundError:
            # Initialise empty dataframe 
            training_df = pd.DataFrame({var: [] for var in scope.keys()})
        
        # 2.1 Check for the categorical (OHE) variables that the scope actually includes all possible
        # values (e.g. solvents). Throws an error if this is not the case.
        for var, values in scope.items():
            if type(values[0]) != str:
                continue
            permissible_values = set(values)
            values_in_df = set(training_df[var])
            val_not_in_scope = [val for val in values_in_df if val not in permissible_values] 
            if val_not_in_scope:
                raise ValueError(f"""Unknown {var} type/s {tuple(val_not_in_scope)} detected in the training data! Please ensure 
                the scope contains all possible values for all categorical variables!""") 

        print("Generating reaction scope...")
        test_df = EDBOplus.scope_on_the_fly(scope, training_df)
        print(f"Scope generated! Total size (minus already completed experiments): {len(test_df)}")

        # 2.2 No training (experimental) data yet, perform initial sampling and exit early
        if training_df.empty:
            print("There are no experimental observations yet. Random samples will be drawn.")
            samples_drawn = self._init_sampling(df=test_df, batch=batch, seed=seed,
                                     sampling_method=init_sampling_method)
            samples = test_df.iloc[samples_drawn].copy()
            # Append objectives.
            for objective in objectives:
                samples[objective] = ['PENDING'] * len(samples)
            # Write initial samples to (empty) destination file
            samples.to_csv(csv_filename, index=False)
            print(f"Initial samples written to {filename}!")
            return 

        # 2.3 Display features considered by model, then strip training and test dataframes of extraneous features/columns
        if columns_features == 'all':  
            feature_list = test_df.columns.to_list()
        else: 
            feature_list = columns_features
        print(f"This run will optimize for the following objectives: {objectives}")
        print(f"The following features will be used: {feature_list}")

        trialed_columns = set(feature_list + objectives)
        test_df = test_df[[col for col in test_df.columns if col in trialed_columns]]
        training_df = training_df[[col for col in training_df.columns if col in trialed_columns]]

        # 3. Auto-detect categorical variables and insert dummy features (one-hot-encoding).
        numeric_cols = test_df._get_numeric_data().columns
        for nc in numeric_cols:
            test_df[nc] = pd.to_numeric(test_df[nc], downcast='float')
        ohe_columns = list(OrderedSet(test_df.columns) - OrderedSet(numeric_cols))
        ohe_columns = list(OrderedSet(ohe_columns) - OrderedSet(objectives))

        if len(ohe_columns) > 0:
            print(f"The following columns are categorical and will be encoded"
                  f" using One-Hot-Encoding: {ohe_columns}")

        df_train = pd.get_dummies(training_df, prefix=ohe_columns, columns=ohe_columns, drop_first=True, dtype=np.float64)
        # Separates predictive and response variables in training set
        df_train_y = df_train.loc[:,objectives]
        df_train_x = df_train.drop(columns=objectives)
        df_test_x = pd.get_dummies(test_df, prefix=ohe_columns, columns=ohe_columns, drop_first=True, dtype=np.float64)
        data = df_test_x.copy(deep=True)

        # 4. Run the BO model and get indices of new experiments to try
        samples_chosen = self._model_run(
                data=data,
                df_train_x=df_train_x,
                df_test_x=df_test_x,
                df_train_y=df_train_y,
                batch=batch,
                objective_mode=objective_mode,
                objective_thresholds=objective_thresholds,
                seed=seed,
                scaler_x=scaler_features,
                scaler_y=scaler_objectives,
                acquisition_function=acquisition_function
        )

        # 5. Attach objectives predictions and expected improvement.
        tests_with_predictions = test_df.copy(deep=True)
        cols_for_preds = []
        for idx_obj in range(0, len(objectives)):
            name = objectives[idx_obj]
            mean = self.predicted_mean[:, idx_obj]
            var = self.predicted_variance[:, idx_obj]
            ei = self.ei[:, idx_obj]
            tests_with_predictions[f"{name}_predicted_mean"] = mean
            tests_with_predictions[f"{name}_predicted_variance"] = var
            tests_with_predictions[f"{name}_expected_improvement"] = ei
            cols_for_preds.append([f"{name}_predicted_mean",
                                   f"{name}_predicted_variance",
                                   f"{name}_expected_improvement"
                                   ])
        cols_for_preds = np.ravel(cols_for_preds)

        # 6. Retrieve suggested samples
        suggested = test_df.iloc[samples_chosen].copy()
        suggested_with_predictions = tests_with_predictions.iloc[samples_chosen]
        
        # 7. Write prediction results over entire scope to a separate CSV and display suggested samples with predictions
        exp_improvements = [col for col in test_df.columns if col.endswith("expected_improvement")]
        tests_with_predictions = tests_with_predictions.sort_values(exp_improvements, ascending=False)
        tests_with_predictions.to_csv(f"{directory}/pred_{filename}", index=False)
        print(f"Prediction results written to {directory}/pred_{filename}!")
        print("Run finished! Here are the suggested experiments (with predictions)")
        print(suggested_with_predictions)

        # 8. Append suggested samples to file (optional)
        cmd = str(input("Would you like to append the suggested experiments to the data file? (y/n): "))
        if cmd.strip().lower().startswith("y"):
            # Fill in objectives with 'PENDING'
            for obj in objectives:
                suggested[obj] = ['PENDING'] * len(suggested)
            combined = pd.concat([suggested, training_df]).round(4)
            combined.to_csv(csv_filename, index=False)
            print(f"File {csv_filename} updated with suggested experiments!")
    