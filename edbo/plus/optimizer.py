import copy

import torch
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

import numpy as np
import pandas as pd
import os
from ordered_set import OrderedSet
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP, ModelListGP, MixedSingleTaskGP
from botorch.optim import optimize_acqf_discrete

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .utils import EDBOStandardScaler
from scipy.spatial.distance import cdist
from idaes.surrogate.pysmo.sampling import LatinHypercubeSampling, CVTSampling
from .model import build_and_optimize_model
from .scope_generator import create_reaction_scope
from .acquisition import acq_multiobjective_EHVI, acq_multiobjective_MOUCB, acq_EI
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
from pathlib import Path

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

class EDBOplus:

    def __init__(self, gpu=False):

        self.gpu = gpu
        self.predicted_mean = []
        self.predicted_variance = []

    @staticmethod
    def generate_reaction_scope(components, directory='./', filename='reaction.csv', check_overwrite=True):
        """
        Creates a reaction scope from a dictionary of components and values.
        """
        df = create_reaction_scope(components=components, directory=directory,
                                   filename=filename, check_overwrite=check_overwrite)
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
            batch=5, init_sampling_method='seed', seed=0,
            scaler_features=MinMaxScaler(),
            scaler_objectives=StandardScaler(),
            get_predictions=True, acquisition_function='EHVI',
            sigma_uncertainty=1.,
            continuous_features=True,
            add_random_samples=True):

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

        seed: int
            Seed for the random initialization.
        """

        wdir = Path(directory)
        csv_filename = wdir.joinpath(filename)
        torch.manual_seed(seed=seed)
        np.random.seed(seed)

        # 1. Safe checks.
        self.objective_names = objectives
        # Check whether the columns_features contains the objectives.
        if columns_features != 'all':
            for objective in objectives:
                if objective in columns_features:
                    columns_features.remove(objective)

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

        if columns_features != 'all':
            if 'priority' in df.columns.values:
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

        labels_ohe = ['OHE_' + sub for sub in ohe_columns]
        data = pd.get_dummies(df, prefix=labels_ohe,
                              columns=ohe_columns, drop_first=True)

        # 3.2. Any sample with a value 'PENDING' in any objective is a test.
        idx_test = (data[data.apply(lambda r: r.str.contains('PENDING', case=False).any(), axis=1)]).index.values
        idx_train = (data[~data.apply(lambda r: r.str.contains('PENDING', case=False).any(), axis=1)]).index.values

        # Data only contains featurized information (train and test).
        df_train_y = data.loc[idx_train][objectives]
        data = data.drop(columns=objectives + ['priority'])
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
                get_predictions=get_predictions,
                scaler_x=scaler_features,
                scaler_y=scaler_objectives,
                acquisition_function=acquisition_function,
                sigma_uncertainty=sigma_uncertainty,
                continuous_features=continuous_features,
                add_random_samples=add_random_samples
        )

        original_df['priority'] = priority_list

        if get_predictions is True:
            # Attach objectives predictions and expected improvement.
            cols_to_delete = []
            for idx_obj in range(0, len(objectives)):
                name = objectives[idx_obj]
                mean = np.array(self.predicted_mean)[0][:, idx_obj]
                var = np.array(self.predicted_variance)[0][:, idx_obj]
                ei = np.array(self.ei)[0][:, idx_obj]
                original_df[f"{name}_predicted_mean"] = mean
                original_df[f"{name}_predicted_variance"] = var
                original_df[f"{name}_expected_improvement"] = ei
                cols_to_delete.append([f"{name}_predicted_mean",
                                       f"{name}_predicted_variance",
                                       f"{name}_expected_improvement"
                                       ])
            cols_to_delete = np.ravel(cols_to_delete)

            original_df = original_df.sort_values('priority', ascending=False)
            # Save extra df containing predictions, uncertainties and EI.
            original_df.to_csv(f"{directory}/pred_{filename}", index=False)
            # Drop predictions, uncertainties and EI.
            original_df = original_df.drop(columns=cols_to_delete,
                                           axis='columns')
        original_df = original_df.sort_values('priority', ascending=False)
        original_df.to_csv(csv_filename, index=False)

        return original_df

    def _model_run(self, data, df_train_x,  df_test_x, df_train_y, batch,
                   objective_mode, objective_thresholds,
                   scaler_x, scaler_y, acquisition_function, get_predictions,
                   sigma_uncertainty, continuous_features, add_random_samples):
        """
        Runs the surrogate machine learning model.
        Returns a priority list for a given scope (top priority to low priority).
        """

        # Check number of objectives.
        n_objectives = len(df_train_y.columns.values)

        init_train = scaler_x.fit_transform(df_train_x.to_numpy())

        raw_y = df_train_y.astype(float).to_numpy()

        # Get the name of the categorical columns.
        categorical_columns = [col for col in data.columns if 'OHE' in col]
        # Get the index for the categorical columns
        idx_categorical_columns = [data.columns.get_loc(c) for c in categorical_columns if c in data]

        if n_objectives > 1:
            print(f"Using {acquisition_function} acquisition function.")
            y = copy.deepcopy(raw_y)

            # Flip if the optimization is minimizing instead of maximizing.
            for i in range(0, n_objectives):
                if objective_mode[i].lower() == 'min':
                    y[:, i] = -y[:, i]

            init_train = scaler_x.fit_transform(df_train_x.to_numpy())
            y = scaler_y.fit_transform(y)
            test_xnp = scaler_x.transform((df_test_x.to_numpy()).tolist())
            test_x = torch.tensor(test_xnp.tolist())

            # Train a model for each objective.
            cumulative_train_x = init_train.tolist()
            cumulative_train_y = y.tolist()

            best_samples = []
            for q in range(0, batch):

                individual_models = []
                ref_mins = np.min(cumulative_train_y, axis=0)

                for i in range(0, n_objectives):
                    train_x = torch.tensor(cumulative_train_x)
                    train_y = np.array(cumulative_train_y)[:, i]
                    train_y = (np.atleast_2d(train_y).reshape(len(train_y), -1))
                    train_y_i = torch.tensor(train_y.tolist())

                    gp, likelihood = build_and_optimize_model(
                        train_x=train_x,
                        train_y=train_y_i)

                    if continuous_features is False:
                        print('Using Mixed (continuous/categorical) model.')
                        model_i = MixedSingleTaskGP(train_X=train_x,
                                                    train_Y=train_y_i,
                                                    likelihood=likelihood,
                                                    cat_dims=idx_categorical_columns)
                    else:
                        print('Using continuous model.')
                        model_i = SingleTaskGP(train_X=train_x,
                                               train_Y=train_y_i,
                                               covar_module=gp.covar_module,
                                               likelihood=likelihood)

                    individual_models.append(model_i)
                    gp = []

                bigmodel = ModelListGP(*individual_models)

                # Reference point is the minimum seen so far.
                if objective_thresholds is None:
                    ref_point = torch.tensor(ref_mins)
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
                        # Set to min if no collected sampled is above threshold.
                        if objective_thresholds[i] is None or ref_point[0][i] > np.max(np.array(cumulative_train_y)[:, 0]):
                            ref_point[0][i] = ref_mins[i]
                    ref_point = torch.tensor(ref_point[0])

                if acquisition_function == 'EHVI':
                    acq = acq_multiobjective_EHVI(model=bigmodel,
                                                  ref_points=ref_point,
                                                  train_y=cumulative_train_y,
                                                  test_x=test_x
                                                  )
                if acquisition_function == 'MOUCB':  # Entropy based.
                    acq = acq_multiobjective_MOUCB(model=bigmodel,
                                                   train_y=cumulative_train_y,
                                                   test_x=test_x,
                                                   greedy=False)
                if acquisition_function == 'MOGreedy':  # Entropy based.
                    acq = acq_multiobjective_MOUCB(model=bigmodel,
                                                   train_y=cumulative_train_y,
                                                   test_x=test_x,
                                                   greedy=True)

                best_samples.append(acq)

                # Update fantasy.
                cumulative_train_x.append(acq)

                y_pred = bigmodel.posterior(
                    torch.tensor([acq])).mean.detach().cpu().numpy()[0].tolist()
                cumulative_train_y.append(y_pred)

                # Save rescaled predictions (only for first fantasy).
                if get_predictions is True and q == 0:
                    data_scaled = scaler_x.transform(data.values)
                    data_tensor = torch.tensor(data_scaled)

                    scaled_mean = np.zeros((len(data_scaled), n_objectives))
                    scaled_var = np.zeros((len(data_scaled), n_objectives))
                    for obj in range(0, n_objectives):
                        bigmodel.models[obj].eval()
                        pred = bigmodel.models[obj](data_tensor.double())
                        mean_i = pred.mean.detach().numpy()
                        var_i = pred.variance.detach().numpy()
                        scaled_mean[:, obj] = mean_i
                        scaled_var[:, obj] = sigma_uncertainty * var_i

                    self.predicted_mean = [
                        scaler_y.inverse_transform(scaled_mean)]
                    self.predicted_variance = [np.abs(scaler_y.inverse_transform(scaled_var))]

                    for i in range(0, n_objectives):  # Reverse max/mim
                        if objective_mode[i].lower() == 'min':
                            self.predicted_mean[0][:, i] = - \
                            self.predicted_mean[0][:, i]

                    # Return the Expected improvement of the individual models.
                    self.ei = np.zeros_like(self.predicted_mean)

                    for i in range(0, n_objectives):
                        if objective_mode[i].lower() == 'min':
                            y_best = np.min(raw_y[:, i])
                            pred_best = np.min(self.predicted_mean[0][:, i])
                        if objective_mode[i].lower() == 'max':
                            y_best = np.max(raw_y[:, i])
                            pred_best = np.max(self.predicted_mean[0][:, i])
                        ei_i = acq_EI(y_best=y_best,
                                      predictions=self.predicted_mean[0][:, i],
                                      uncertainty=self.predicted_variance[0][:, i],
                                      objective=objective_mode[i].lower()
                                      )
                        self.ei[0][:, i] = ei_i

        if n_objectives == 1:  # Single-objective
            # Scale data.
            y = df_train_y.astype(float).to_numpy()

            if objective_mode[0].lower() == 'min':
                y = -y

            y = scaler_y.fit_transform(y)
            test_xnp = scaler_x.transform((df_test_x.to_numpy()).tolist())
            test_x = torch.tensor(test_xnp.tolist())

            cumulative_train_x = init_train.tolist()
            cumulative_train_y = y.tolist()

            best_samples = []
            for q in range(0, batch):
                # Build tensors.
                train_x = torch.tensor(cumulative_train_x)
                train_y = torch.tensor(cumulative_train_y)
                model, likelihood = build_and_optimize_model(train_x=train_x, train_y=train_y)

                model_stask = SingleTaskGP(train_X=train_x, train_Y=train_y,
                                           covar_module=model.covar_module,
                                           likelihood=likelihood)
                best_value = train_y.max()
                # Do normal expected improvement first.
                EI = ExpectedImprovement(model_stask, best_f=best_value,
                                         maximize=True)
                acq = optimize_acqf_discrete(EI, choices=test_x, q=1
                                             )[0][0].detach().numpy().tolist()

                # Update fantasy.
                cumulative_train_x.append(acq)

                y_pred = model_stask.posterior(torch.tensor([acq])).mean.detach().numpy()[0].tolist()
                cumulative_train_y.append(y_pred)

                # Collect best sample
                best_samples.append(acq)

                # Save rescaled predictions only for the first process (no fantasy)..
                if get_predictions is True and q == 0:
                    data_np = scaler_x.transform(data.values)
                    data_tensor = torch.tensor(data_np)
                    scaled_mean = model_stask.posterior(data_tensor).mean.detach().numpy()
                    self.predicted_mean = [scaler_y.inverse_transform(scaled_mean)]

                    # Revert back if minimizing.
                    if objective_mode[0].lower() == 'min':
                        self.predicted_mean = (-np.array(self.predicted_mean)).tolist()
                    scaled_var = model_stask.posterior(data_tensor).variance.detach().numpy()
                    self.predicted_variance = [np.abs(scaler_y.inverse_transform(scaled_var))]

                    # Return the Expected improvement of the individual models.
                    self.ei = np.zeros_like(self.predicted_mean)

                    if objective_mode[0].lower() == 'min':
                        y_best = np.min(raw_y)
                        pred_best = np.min(self.predicted_mean)
                    if objective_mode[0].lower() == 'max':
                        y_best = np.max(raw_y)
                        pred_best = np.max(self.predicted_mean)

                    ei_i = acq_EI(y_best=y_best,
                                  predictions=self.predicted_mean[0].flatten(),
                                  uncertainty=self.predicted_variance[0].flatten(),
                                  objective=objective_mode[0].lower()
                                  )
                    e_i_res = np.reshape(ei_i, (len(ei_i), -1))
                    self.ei[0, :] = e_i_res


        # Rescale samples.
        best_samples = scaler_x.inverse_transform(best_samples)

        priority_list = [0] * len(data.values)

        # Give very low priority to already collected data.
        all_samples = data.values
        for sample in df_train_x.to_numpy().tolist():
            # Compute distance between all samples and the collected data.
            d_i = cdist([sample], all_samples)[0]
            a = np.argmin(d_i)
            priority_list[a] = -1.

        # Find best samples in data.
        for sample in best_samples:
            d_i = cdist([sample], all_samples)[0]
            a = np.argmin(d_i)
            priority_list[a] = 1.

        # Add extra random samples if below the batch size.
        if add_random_samples:
            seed = 0
            while (np.array(priority_list) >= 0.5).sum() < batch:
                np.random.seed(seed)
                print('Adding extra random samples.')
                random_c = np.random.choice(np.arange(0, len(priority_list)))
                if priority_list[random_c] == 0:
                    priority_list[random_c] = 0.5
                seed += 1
                print('selected', (np.array(priority_list) >= 0.5).sum())

        return priority_list
