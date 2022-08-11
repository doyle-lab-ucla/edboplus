# EDBO plus through only BOTorch functions.


import torch
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from botorch.models import SingleTaskGP, ModelListGP
from ordered_set import OrderedSet

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.spatial.distance import cdist
from idaes.surrogate.pysmo.sampling import LatinHypercubeSampling, CVTSampling
from .model import build_and_optimize_model
from .scope_generator import create_reaction_scope
from botorch.utils.multi_objective.box_decompositions import \
    NondominatedPartitioning
from botorch.optim import optimize_acqf_discrete
from botorch.acquisition.multi_objective.monte_carlo import \
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement

from pathlib import Path
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from edbo.plus.utils import EDBOStandardScaler

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class EDBOplus:

    def __init__(self):

        self.predicted_mean = []
        self.predicted_variance = []

    @staticmethod
    def generate_reaction_scope(components, directory='./', filename='reaction.csv',
                                check_overwrite=True):
        """
        Creates a reaction scope from a dictionary of components and values.
        """
        df = create_reaction_scope(components=components, directory=directory,
                                   filename=filename,
                                   check_overwrite=check_overwrite)
        return df

    @staticmethod
    def _init_sampling(df, batch, sampling_method, seed):

        numeric_cols = df._get_numeric_data().columns
        ohe_columns = list(OrderedSet(df.columns) - OrderedSet(numeric_cols))
        if len(ohe_columns) > 0:
            print(f"The following columns are categorical and will be encoded"
                  f" using One-Hot-Encoding: {ohe_columns}")
        # Encode OHE.
        df_sampling = pd.get_dummies(df, prefix=ohe_columns,
                                     columns=ohe_columns, drop_first=True)

        # Order df according to initial sampling method (random samples).
        idaes = None
        if sampling_method == 'seed':
            print(f"Using seeded random sampling (seed={seed}).")
            samples = df_sampling.sample(n=batch, random_state=seed)
        elif sampling_method.lower() == 'lhs':
            idaes = LatinHypercubeSampling(df_sampling, batch, sampling_type="selection")
        elif sampling_method.lower() == 'cvtsampling':
            idaes = CVTSampling(df_sampling, batch, sampling_type="selection")

        if idaes is not None:
            samples = idaes.sample_points()

        print(f"Creating a priority list using random sampling: {sampling_method}")

        # Get index of the best samples according to the random sampling method.
        df_sampling_matrix = df_sampling.to_numpy()
        priority_list = np.zeros_like(df_sampling.index)

        for sample in samples.to_numpy():
            d_i = cdist([sample], df_sampling_matrix, metric='cityblock')
            a = np.argmin(d_i)
            priority_list[a] = 1.
        df['priority'] = priority_list
        return df

    def run(self,
            objectives, objective_mode, objective_thresholds=None,
            directory='.', filename='reaction.csv',
            columns_features='all',
            batch=5, init_sampling_method='cvtsampling', seed=0,
            scaler_features=MinMaxScaler(),
            scaler_objectives=EDBOStandardScaler(),
            acquisition_function='EHVI',
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

        get_predictions: boolean
            If True it will print out a *csv file* with the predictions.
            You can also access the *predicted_mean* and *predicted_variance* through the EDBOPlus class.

        directory: string
            name of the directory to save the results of the optimization.

        filename: string
            Name of the file to save a *csv* with the priority list. If *get_predictions=True* EDBO+ will automatically
            save a second file including the predictions (*pred_filename.csv*).

        init_sampling_method: string:
            Method for selecting the first samples in the scope (in absence)  Choices are:
            - 'seed' : Random seed (as implemented in Pandas).
            - 'lhs' : LatinHypercube sampling.
            - 'cvtsampling' : CVT sampling.

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
        # Check whether the columns_features contains the objectives.
        if columns_features != 'all':
            for objective in objectives:
                if objective in columns_features:
                    columns_features.remove(objective)
                if 'priority' in columns_features:
                    columns_features.remove('priority')

        # Check objectives is a list (even for single objective optimization).
        ohe_features = False
        if type(objectives) != list:
            objectives = [objectives]
        if type(objective_mode) != list:
            objective_mode = [objective_mode]

        # Check that the user's scope exists.
        msg = "Scope was not found. Please create an scope (csv file)."
        assert os.path.exists(csv_filename), msg

        # 2. Load reaction.
        df = pd.read_csv(f"{csv_filename}")
        df = df.dropna(axis='columns', how='all')
        original_df = df.copy(deep=True)  # Make a copy of the original data.

        # 2.1. Initialize sampling (only in the first iteration).
        obj_in_df = list(filter(lambda x: x in df.columns.values, objectives))

        # TODO CHECK: Check whether new objective has been added – if not add PENDING.
        for obj_i in self.objective_names:
            if obj_i not in original_df.columns.values:
                original_df[obj_i] = ['PENDING'] * len(original_df.values)

        if columns_features != 'all':
            if 'priority' in df.columns.values:
                for obj_i in objectives:
                    if obj_i not in df.columns.values:
                        df[obj_i] = ['PENDING'] * len(df.values)

                df = df[columns_features + objectives + ['priority']]
            else:
                if len(obj_in_df) == 0:
                    df = df[columns_features]
                else:
                    df = df[columns_features + objectives]

        # No objectives columns in the scope? Then random initialization.
        if len(obj_in_df) == 0:
            df = self._init_sampling(df=df, batch=batch, seed=seed,
                                     sampling_method=init_sampling_method)
            original_df['priority'] = df['priority']
            # Append objectives.
            for objective in objectives:
                if objective not in original_df.columns.values:
                    original_df[objective] = ['PENDING'] * len(original_df)

            # Sort values and save dataframe.
            original_df = original_df.sort_values('priority', ascending=False)
            original_df = original_df.loc[:,~original_df.columns.str.contains('^Unnamed')]
            original_df.to_csv(csv_filename, index=False)
            return original_df

        # 3. Separate train and test data.

        # 3.1. Auto-detect dummy features (one-hot-encoding).
        numeric_cols = df._get_numeric_data().columns
        for nc in numeric_cols:
            df[nc] = pd.to_numeric(df[nc], downcast='float')
        ohe_columns = list(OrderedSet(df.columns) - OrderedSet(numeric_cols))
        ohe_columns = list(OrderedSet(ohe_columns) - OrderedSet(objectives))

        if len(ohe_columns) > 0:
            print(f"The following columns are categorical and will be encoded"
                  f" using One-Hot-Encoding: {ohe_columns}")
            ohe_features = True

        data = pd.get_dummies(df, prefix=ohe_columns, columns=ohe_columns, drop_first=True)

        # 3.2. Any sample with a value 'PENDING' in any objective is a test.
        idx_test = (data[data.apply(lambda r: r.str.contains('PENDING', case=False).any(), axis=1)]).index.values
        idx_train = (data[~data.apply(lambda r: r.str.contains('PENDING', case=False).any(), axis=1)]).index.values

        # Data only contains featurized information (train and test).
        df_train_y = data.loc[idx_train][objectives]
        if 'priority' in data.columns.tolist():
            data = data.drop(columns=objectives + ['priority'])
        else:
            data = data.drop(columns=objectives)
        df_train_x = data.loc[idx_train]
        df_test_x = data.loc[idx_test]

        if len(df_train_x.values) == 0:
            msg = 'The scope was already generated, please ' \
                  'insert at least one experimental observation ' \
                  'value and then press run.'
            print(msg)
            return original_df

        # Run the BO process.
        priority_list = self._model_run(
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

        # Low priority to the samples that have been already collected.
        for i in range(0, len(idx_train)):
            priority_list[idx_train[i]] = -1

        original_df['priority'] = priority_list

        cols_sort = ['priority'] + original_df.columns.values.tolist()
        # Attach objectives predictions and expected improvement.
        cols_for_preds = []
        for idx_obj in range(0, len(objectives)):
            name = objectives[idx_obj]
            mean = self.predicted_mean[:, idx_obj]
            var = self.predicted_variance[:, idx_obj]
            ei = self.ei[:, idx_obj]
            original_df[f"{name}_predicted_mean"] = mean
            original_df[f"{name}_predicted_variance"] = var
            original_df[f"{name}_expected_improvement"] = ei
            cols_for_preds.append([f"{name}_predicted_mean",
                                   f"{name}_predicted_variance",
                                   f"{name}_expected_improvement"
                                   ])
        cols_for_preds = np.ravel(cols_for_preds)

        original_df = original_df.sort_values(cols_sort, ascending=False)
        # Save extra df containing predictions, uncertainties and EI.
        original_df.to_csv(f"{directory}/pred_{filename}", index=False)
        # Drop predictions, uncertainties and EI.
        original_df = original_df.drop(columns=cols_for_preds, axis='columns')
        original_df = original_df.sort_values(cols_sort, ascending=False)
        original_df.to_csv(csv_filename, index=False)

        return original_df

    def _model_run(self, data, df_train_x,  df_test_x, df_train_y, batch,
                   objective_mode, objective_thresholds, seed,
                   scaler_x, scaler_y, acquisition_function):
        """
        Runs the surrogate machine learning model.
        Returns a priority list for a given scope (top priority to low priority).
        """

        # Check number of objectives.
        n_objectives = len(df_train_y.columns.values)

        print(f"Using {acquisition_function} acquisition function.")
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

        for i in range(0, n_objectives):
            train_x = torch.tensor(init_train).to(**tkwargs).double()
            train_y = np.array(y)[:, i]
            train_y = (np.atleast_2d(train_y).reshape(len(train_y), -1))
            train_y_i = torch.tensor(train_y.tolist()).to(**tkwargs).double()

            gp, likelihood = build_and_optimize_model(train_x=train_x, train_y=train_y_i,)

            model_i = SingleTaskGP(train_X=train_x, train_Y=train_y_i,
                                   covar_module=gp.covar_module, likelihood=likelihood)
            individual_models.append(model_i)

        bigmodel = ModelListGP(*individual_models)

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
        elif len(data.values) > 5000:
            sobol_num_samples = 128
        elif len(data.values) > 10000:
            sobol_num_samples = 256
        else:
            sobol_num_samples = 512

        print(f'Number of QMC samples using {self.acquisition_sampler} sampler:', sobol_num_samples)
        y_torch = torch.tensor(y).to(**tkwargs).double()

        if self.acquisition_sampler == 'IIDNormalSampler':
            sampler = IIDNormalSampler(num_samples=sobol_num_samples, collapse_batch_dims=False, seed=seed)
        if self.acquisition_sampler == 'SobolQMCNormalSampler':
            sampler = SobolQMCNormalSampler(num_samples=sobol_num_samples, collapse_batch_dims=False)

        if acquisition_function == 'EHVI':

            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=y_torch)
            EHVI = qExpectedHypervolumeImprovement(
                model=bigmodel, sampler=sampler,
                ref_point=ref_point,  # use known reference point
                partitioning=partitioning,
            )

            acq_result = optimize_acqf_discrete(
                acq_function=EHVI,
                choices=test_x,
                q=batch,
                unique=True
            )

        if acquisition_function == 'noisyEHVI':
            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=torch.tensor(y)).double().to(**tkwargs)
            nEHVI = qNoisyExpectedHypervolumeImprovement(
                model=bigmodel, sampler=sampler,
                ref_point=ref_point,  # use known reference point
                partitioning=partitioning,
                incremental_nehvi=True, X_baseline=train_x, prune_baseline=False,
            )
            acq_result = optimize_acqf_discrete(
                acq_function=nEHVI,
                choices=test_x,
                q=batch,
                unique=True
            )

        best_samples = scaler_x.inverse_transform(acq_result[0].detach().cpu().numpy())

        print('Acquisition function optimized.')

        # Save rescaled predictions (only for first fantasy).

        # Get predictions in chunks.
        chunk_size = 1000
        n_chunks = len(data.values) // chunk_size

        if n_chunks == 0:
            n_chunks = 1

        self.predicted_mean = np.zeros(shape=(len(data.values), n_objectives))
        self.predicted_variance = np.zeros(shape=(len(data.values), n_objectives))
        self.ei = np.zeros(shape=(len(data.values), n_objectives))

        observed_raw_values = df_train_y.astype(float).to_numpy()

        for i in range(0, len(data.values), n_chunks):
            vals = data.values[i:i+n_chunks]
            data_tensor = torch.tensor(scaler_x.transform(vals)).double().to(**tkwargs)
            preds = bigmodel.posterior(X=data_tensor)
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

        print('Predictions obtained and expected improvement obtained.')

        # Flip predictions if needed.
        for i in range(0, len(objective_mode)):
            if objective_mode[i] == 'min':
                self.predicted_mean[:, i] = -self.predicted_mean[:, i]

        # Rescale samples.
        all_samples = data.values

        priority_list = [0] * len(data.values)

        # Find best samples in data.
        for sample in best_samples:
            d_i = cdist([sample], all_samples, metric='cityblock')
            a = np.argmin(d_i)
            priority_list[a] = 1.

        return priority_list

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

        sigma = variance * 2.

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
