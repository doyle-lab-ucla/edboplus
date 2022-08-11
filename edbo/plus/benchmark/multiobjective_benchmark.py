
import os
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from sklearn.preprocessing import MinMaxScaler
import pareto
import pandas as pd
from edbo.plus.optimizer_botorch import EDBOplus
from edbo.plus.optimizer import EDBOplus as EDBO
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from scipy.spatial.distance import euclidean
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def is_pareto(objectives):
    """
    Find the pareto-efficient points.

    Parameters
    ----------
    objectives: array
        array containing all the objectives values.
    Returns
    -------
    is_efficient
        list of booleans, True: Pareto efficient and False: not Pareto efficient.

    """
    is_efficient = np.ones(objectives.shape[0], dtype=bool)
    for i, c in enumerate(objectives):
        is_efficient[i] = np.all(np.any(objectives[:i]>c, axis=1)) and np.all(np.any(objectives[i+1:]>c, axis=1))
    return is_efficient


class Benchmark():

    """
    Class for Benchmarking HTE datasets.
    """

    def __init__(self,
                 df_ground, index_column,
                 objective_names, objective_modes,
                 features_regression='all',
                 objective_thresholds=None,
                 filename='benchmark.csv',
                 acquisition_function='EHVI',
                 filename_results='benchmark_results.csv'):
        """

        Parameters
        ----------
        df_ground: Pandas dataframe.
            Pandas dataframe containing the features and objectives.
        index_column: string
            String with the name of the index column of the 'df_ground' Pandas dataframe.
        objective_names: list
            list of strings including the names of the columns that will be considered as objectives.
        objective_modes: list
            list of strings for deciding whether the objective has to be minimize ('min') or maximize ('max').
        features_regression: list
            list of strings containing the names of the columns that will be considered as features for the regression model.
        objective_thresholds: list
            list of floats containing the threshold values for each objective.
        acquisition_function: string
            name of the acquisition function to use (implemented are: 'EHVI', 'MOUCB', 'MOGreedy').
        filename: string
            name of the file to run the optimization (temporary file).
        filename_results:
            name of the file to store the results of the benchmark.
        """
        self.df_ground = df_ground
        self.objective_names = objective_names
        self.objective_modes = objective_modes
        self.objective_thresholds = objective_thresholds
        self.features_regressions = features_regression
        self.filename = filename
        self.filename_results = filename_results
        self.index_column = index_column
        self.acquisition_function = acquisition_function

        # Safe check. Check whether there are no duplicates in the regression domain.
        df_check = self.df_ground[self.features_regressions]
        msg = 'There are entries that are degenerate. Check your dataset.'
        assert len(df_check) == len(df_check.drop_duplicates()), msg

        # Calculate data for the ground truth function.
        self.objective_values_ground = self.get_objective_values(df=self.df_ground)
        self.pareto_ground, self.idx_pareto_ground = self.get_pareto_points(objective_values=self.objective_values_ground)
        self.tradeoff_ground = self.get_high_tradeoff_points(
                pareto_points=self.pareto_ground)
        print('High trade-off ground truth:', self.tradeoff_ground)

        # Fit scaler for a fair comparison between objectives.
        self.scaler_ground = MinMaxScaler()
        self.scaler_ground.fit(self.objective_values_ground)

        # Calculate Hypervolume using the ground scaler.
        self.hypervolume_ground = self.get_hypervolume(
                pareto_points=self.pareto_ground,
                thresholds=self.objective_thresholds)

        # Instanciate EDBOplus.
        if self.acquisition_function == 'EHVI':
            self.edbo = EDBOplus()
        else:
            self.edbo = EDBO()

        # Generate DataFrame for storing collected data and predictions.
        if not os.path.exists(filename):
            # Create first scope if filename is not found.
            df_edbo = self.df_ground.copy(deep=True)
            df_edbo = df_edbo.drop(columns=self.objective_names)
            df_edbo.to_csv(filename, index=False)

    def get_hypervolume(self, pareto_points, thresholds):
        """
        Calculate hypervolume using a reference point for each objective
        (if None it takes automatically the minimum of each objective in the ground truth).
        Note 1: It is always scaled w.r.t. the ground truth for a fair comparison between objectives.
        Note 2: Assumes maximization.
        """

        references = copy.deepcopy(thresholds)

        if references is None:
            references = [None] * len(self.objective_names)

        for i in range(0, len(references)):
            if references[i] is None:
                references[i] = np.min(self.objective_values_ground, axis=0)[i]

        references_scaled = self.scaler_ground.transform(np.array([references]))[0]
        pareto_points_scaled = self.scaler_ground.transform(pareto_points)

        pareto_torch = torch.Tensor(pareto_points_scaled)
        hv = Hypervolume(ref_point=torch.Tensor(references_scaled))
        hypervolume = hv.compute(pareto_Y=pareto_torch)

        return hypervolume

    def get_objective_values(self, df):  # Maximizing.
        """ Get a list of objective values from a given dataframe (for instance yields and cost)."""
        predata = []
        for obj in range(0, len(self.objective_names)):
            d = [float(i) for i in df[self.objective_names[obj]]]
            if self.objective_modes[obj] == 'min':
                d = -np.array(d)
            else:
                d = np.array(d)
            predata.append(d.flatten())
        objective_values = np.array(predata).T

        # Calculate also bounds (best and worst samples).
        max_bounds = []
        min_bounds = []
        for i in range(0, len(self.objective_names)):
            if self.objective_modes[i] == 'min':
                max_bounds.append(-np.max(self.df_ground[self.objective_names[i]]))
                min_bounds.append(-np.min(self.df_ground[self.objective_names[i]]))
            if self.objective_modes[i] == 'max':
                max_bounds.append(np.max(self.df_ground[self.objective_names[i]]))
                min_bounds.append(np.min(self.df_ground[self.objective_names[i]]))
        self.max_bounds = max_bounds
        self.min_bounds = min_bounds

        return objective_values

    def get_pareto_points(self, objective_values):
        """ Get pareto for the ground truth function.
        NOTE: Assumes maximization."""
        pareto_ground = pareto.eps_sort(tables=objective_values,
                                        objectives=np.arange(len(self.objective_names)),
                                        maximize_all=True)
        idx_pareto = is_pareto(objectives=-objective_values)
        return np.array(pareto_ground), idx_pareto

    def get_high_tradeoff_points(self, pareto_points):
        """ Pass a numpy array with the pareto points and returns a numpy
            array with the high tradeoff points."""

        scaler_pareto = MinMaxScaler()
        pareto_scaled = scaler_pareto.fit_transform(pareto_points)
        try:
            tradeoff = HighTradeoffPoints()

            tradeoff_args = tradeoff.do(-pareto_scaled)  # Always minimizing.
            tradeoff_points = pareto_points[tradeoff_args]
        except:
            tradeoff_points = []
            pass
        return tradeoff_points

    def get_distance_tradeoff_to_ground(self, set_of_points):
        """
        Max. minimum Euclidean distance between a set of points and
        the ground truth tradeoff point(s).
        """
        if len(self.tradeoff_ground) != 0:
            d_ij = []
            try:
                for tradeoff_point_ground in self.tradeoff_ground:
                    d_i = []
                    for point in set_of_points:
                        d_i.append(euclidean(point, tradeoff_point_ground))
                    d_ij.append(np.min(d_i))
            except:
                for tradeoff_point_ground in self.tradeoff_ground[0]:
                    d_i = []
                    for point in set_of_points:
                        d_i.append(euclidean(point, tradeoff_point_ground))
                    d_ij.append(np.min(d_i))
            return np.max(d_ij)
        else:
            return 'NA'

    def get_maximin_distance_pareto_to_ground(self, pareto_set):
        """
        Calculate maximin distance between a pareto set and the ground truth
        pareto. Also called Hausdorff
        """
        d_ij = []
        for pareto_point_ground in self.pareto_ground:
            d_i = []
            for point in pareto_set:
                d_i.append(euclidean(point, pareto_point_ground))
            d_ij.append(np.min(d_i))
        return np.max(d_ij)

    def get_predictions_errors(self):
        """
        Obtain errors in the predictions.
        """
        dict_pred_errors = {}
        df_ground = self.df_ground.copy(deep=True)

        if os.path.exists(f'pred_{self.filename}'):
            df_preds = pd.read_csv(f'pred_{self.filename}')
            df_preds = df_preds.sort_values(by=self.index_column)
            df_ground = df_ground.sort_values(by=self.index_column)

            # Collect information about the MAE, RMSE, ,
            for objective in self.objective_names:
                y_true = df_ground[objective].values
                y_pred = df_preds[f"{objective}_predicted_mean"].values

                mae = (np.sum(np.abs(y_pred - y_true))) / len(y_pred)
                rmse = np.sqrt((np.sum((y_pred - y_true)**2))/len(y_pred))

                from sklearn.metrics import r2_score

                r2 = r2_score(y_true=y_true, y_pred=y_pred)

                dict_pred_errors.update({
                    f"MAE_{objective}": mae,
                    f"RMSE_{objective}": rmse,
                    f"R2_{objective}": r2,
                    })
            print(dict_pred_errors)
        else:
            for objective in self.objective_names:
                dict_pred_errors.update({
                    f"MAE_{objective}": 'NaN',
                    f"RMSE_{objective}": 'NaN',
                    f"R2_{objective}": 'NaN',
                })

        return dict_pred_errors

    def _store_benchmark(self):
        for bt in range(self.batch):
            dict_i = {'step': self.step,
                      'init_method': self.init_method,
                      'init_sample': self.seed,
                      'batch': self.batch,
                      'n_experiments': self.n_experiments,
                      'thresholds': self.objective_thresholds,
                      'dmaximin_pareto': self.pareto_distance_ground_train,
                      'acquisition': self.acquisition_function,
                      'dmaximin_tradeoff': self.tradeoff_distance_ground_train,
                      'hypervolume_ground': self.hypervolume_ground,
                      'hypervolume_sampled': self.hypervolume_train,
                      'hypervolume completed (%)': (self.hypervolume_train/self.hypervolume_ground) * 100,
                      'sample_index': self.idx_next_samples[bt],
            }

            dict_i.update(self.predicted_errors)

            sample_vals = {}
            for kcol in self.collected_values.keys():
                sample_vals[kcol] = self.collected_values[kcol][bt]
            dict_i.update(sample_vals)
            best_samples_values = {}

            for i in range(0, len(self.objective_names)):
                best_i = self.best_values_found[i]
                best_samples_values.update({f"{self.objective_names[i]}_best": best_i})

            dict_i.update(best_samples_values)

            store_columns = list(dict_i.keys())

            if os.path.exists(self.filename_results):
                df_results = pd.read_csv(self.filename_results)
            else:
                df_results = pd.DataFrame(columns=store_columns)

            df_results = df_results.append(dict_i, ignore_index=True)
            df_results.to_csv(self.filename_results, index=False)

        return df_results

    def run(self, steps, batch, seed=0,
            init_method='seed',
            run_folder='./results', plot_ground=True,
            plot_train=False, plot_predictions=False,
            random_sampling=False):

        self.init_method=init_method
        self.seed = seed
        self.batch = batch
        self.run_folder = run_folder

        if not os.path.exists(run_folder):
            os.mkdir(run_folder)

        if plot_ground is True:
            if len(self.objective_names) == 2:
                self._plot_ground_2d()
            if len(self.objective_names) == 3:
                self._plot_ground_3d()

        for step in range(0, steps):
            self.step = step

            df_run = self.edbo.run(
                    filename=self.filename,
                    batch=batch,
                    columns_features=self.features_regressions,
                    objectives=self.objective_names,
                    objective_mode=self.objective_modes,
                    objective_thresholds=self.objective_thresholds,
                    acquisition_function=self.acquisition_function,
                    init_sampling_method=self.init_method,
                    seed=seed)

            df_next = df_run[df_run['priority'] >= 0.5]
            idx_next = df_next[self.index_column].values

            if len(idx_next) < batch:  # Add random if not enough.
                df_next = df_run[df_run['priority'] >= 0]
                idx_next_2 = df_next[self.index_column].values
                l_choice = np.arange(0, len(idx_next_2))
                r_choices = np.random.choice(l_choice, size=(batch-len(idx_next)))
                random_choice_idx = idx_next_2[r_choices]
                idx_next = np.append(idx_next, random_choice_idx)

            self.idx_next_samples = idx_next
            # Test performance of the model with large training data.
            # idx_next = np.random.choice(len(df_run), int(len(df_run)*0.2))
            if random_sampling is True:
                df_next = df_run[df_run['priority'] >= 0]
                idx_next = df_next[self.index_column].values
                choice_idx = idx_next[np.random.choice(np.arange(0, len(idx_next)), size=batch)]
                idx_next = choice_idx

            # Append solutions.
            self.collected_values = {}
            for id in idx_next:
                argwhere_idx_next_ground = np.argwhere(self.df_ground[self.index_column].values == id)[0][0]
                argwhere_dix_next_run = np.argwhere(df_run[self.index_column].values == id)[0][0]
                for obj in self.objective_names:
                    df_run[obj].values[argwhere_dix_next_run] = self.df_ground[obj].values[argwhere_idx_next_ground]
                    try:
                        self.collected_values[obj + "_collected_values"].append(self.df_ground[obj].values[argwhere_idx_next_ground])
                    except:
                        self.collected_values[obj + "_collected_values"] = [self.df_ground[obj].values[argwhere_idx_next_ground]]

            # Update the DataFrame with the new collected data.
            df_run.to_csv(self.filename, index=False)

            new_df = df_run.copy(deep=True)
            # Print best values for each objective and save sampled objectives.
            cumulative_train_y = []
            self.best_values_found = []
            for i in range(0, len(self.objective_names)):
                if self.objective_modes[i] == 'min':
                    best_value = pd.to_numeric(new_df[self.objective_names[i]], 'coerce').min()
                else:
                    best_value = pd.to_numeric(new_df[self.objective_names[i]], 'coerce').max()
                print(f"Best {self.objective_names[i]} found: {best_value}.")
                self.best_values_found.append(best_value)
                vals = pd.to_numeric(new_df[self.objective_names[i]], 'coerce').dropna().values
                if self.objective_modes[i] == 'min':
                    cumulative_train_y.append(-vals)
                else:
                    cumulative_train_y.append(vals)
            cumulative_train_y = np.reshape(cumulative_train_y, (len(cumulative_train_y), -1)).T

            self.pareto_train, self.idx_pareto_train = self.get_pareto_points(objective_values=cumulative_train_y)
            self.tradeoff_train = self.get_high_tradeoff_points(pareto_points=self.pareto_train)
            self.hypervolume_train = self.get_hypervolume(pareto_points=self.pareto_train,
                                                          thresholds=self.objective_thresholds)

            self.tradeoff_distance_ground_train = \
                self.get_distance_tradeoff_to_ground(
                        set_of_points=cumulative_train_y)
            self.pareto_distance_ground_train = \
                    self.get_maximin_distance_pareto_to_ground(
                            pareto_set=self.pareto_train)

            self.n_experiments = len(cumulative_train_y)

            print('Total number of experiments:', self.n_experiments)

            # Print collected samples.
            print('Collected samples \n', df_run[df_run['priority'] < 0])

            print(f"Hypervolume train (w.r.t to ground truth in %): "
                  f"{(self.hypervolume_train/self.hypervolume_ground) * 100}")

            print(f"Maximin distance any train to any ground truth "
                  f"Pareto: {self.pareto_distance_ground_train}")
            print(f"Maximin distance any train to ground truth "
                  f"Tradeoff: {self.tradeoff_distance_ground_train}")

            # Store all information.

            self.predicted_errors = self.get_predictions_errors()
            self._store_benchmark()
            print("\n \n")

            # Get plots.
            self.cumulative_train_y = cumulative_train_y

            if plot_train:
                if len(self.objective_names) == 2:
                    self._plot_train_pareto_2d()
                if len(self.objective_names) == 3:
                    self._plot_train_pareto_3d()

            if plot_predictions and os.path.exists(f"pred_{self.filename}"):
                self._plot_predictions()

    def _plot_predictions(self):

        df_preds = pd.read_csv(f'pred_{self.filename}')
        df_ground = self.df_ground.copy(deep=True)
        df_preds = df_preds.sort_values(by=self.index_column)
        df_ground = df_ground.sort_values(by=self.index_column)

        total_entries = len(df_preds)
        n_columns = 30
        n_rows = int(total_entries / n_columns) + 1

        for i in self.objective_names:
            ground = [np.nan] * n_columns * n_rows
            pred_mean = [np.nan] * n_columns * n_rows
            ei = [np.nan] * n_columns * n_rows
            ground[0:total_entries] = self.df_ground[i].values
            pred_mean[0:total_entries] = df_preds[f"{i}_predicted_mean"].values
            ei[0:total_entries] = df_preds[f"{i}_expected_improvement"].values
            ground = np.reshape(ground, (n_rows, n_columns))
            pred_mean = np.reshape(pred_mean, (n_rows, n_columns))
            ei = np.reshape(ei, (n_rows, n_columns))

            min_threshold = np.min(df_ground[i].values)
            max_threshold = np.max(df_ground[i].values)

            f, (ax1, ax2, ax3) = plt.subplots(
                1, 3,
                figsize=(n_columns, n_rows),
                gridspec_kw={
                  'width_ratios': [1, 1, 1]
                }
            )

            g1 = sns.heatmap(
                ground, cmap='Spectral_r', square=True,
                xticklabels=[], yticklabels=[],
                linewidths=0.7, cbar=False,
                cbar_kws=dict(use_gridspec=False, location="bottom"),
                vmin=min_threshold, vmax=max_threshold,
                ax=ax1
            )

            g2 = sns.heatmap(
                pred_mean, cmap='Spectral_r', square=True,
                xticklabels=[], yticklabels=[],
                linewidths=0.7, cbar=False,
                cbar_kws=dict(use_gridspec=False, location="bottom"),
                vmin=min_threshold, vmax=max_threshold,
                ax=ax2
            )

            g3 = sns.heatmap(
                ei,
                cmap='Blues', square=True,
                xticklabels=[], yticklabels=[],
                linewidths=0.7, cbar=False,
                cbar_kws=dict(use_gridspec=False, location="bottom"),
                vmin=np.min(df_preds[f"{i}_expected_improvement"].values),
                vmax=np.max(df_preds[f"{i}_expected_improvement"].values),
                ax=ax3
            )

            ax1.set_title(f"Ground truth ({i})")
            ax2.set_title(f"Predicted mean ({i})")
            ax3.set_title(f"Expected improvement ({i})")
            plt.tight_layout()
            plt.show()

    def _plot_ground_2d(self):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.objective_values_ground[:, 0],
                   self.objective_values_ground[:, 1], s=30, lw=1.,
                   edgecolors='black')
        ax.scatter(self.pareto_ground[:, 0], self.pareto_ground[:, 1],
                   s=100, c='orange', lw=1., edgecolors='black')

        import matplotlib.patches as patches
        ax.add_patch(
            patches.Rectangle(
                tuple(self.min_bounds),
                self.max_bounds[0],
                self.max_bounds[1],
                facecolor="red",
                alpha=0.1
            )
        )

        idx_pareto = np.argwhere(self.idx_pareto_ground)
        print('Indices pareto front (ground truth):', idx_pareto)
        for i in range(0, len(idx_pareto)):
            idxi = idx_pareto[i][0]
            ax.text(x=self.objective_values_ground[idxi][0] +0.,
                    y=self.objective_values_ground[idxi][1] +0.,
                    s=str(idxi),
                    )

        ax.set_xlabel(f"{self.objective_names[0]} ({self.objective_modes[0]})")
        ax.set_ylabel(f"{self.objective_names[1]} ({self.objective_modes[1]})")

        if not os.path.exists(f"{self.run_folder}_plots"):
            os.mkdir(f"{self.run_folder}_plots")
        plt.rc("axes.spines", top=False, right=False)
        plt.savefig(f"{self.run_folder}_plots/{self.filename_results}_ground.svg")
        plt.show()

    def _plot_ground_3d(self):

        plt.rcParams['grid.color'] = "lightgray"
        plt.rcParams['grid.linestyle'] = [6., 2., 6., 2.]
        plt.rcParams['grid.linewidth'] = 1.5

        # Scatter plots 3d.
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        # Plot properties.
        ax.view_init(elev=25, azim=290)
        fig.set_facecolor('white')
        ax.set_facecolor('white')

        ax.scatter(self.objective_values_ground[:, 0],
                   self.objective_values_ground[:, 1],
                   self.objective_values_ground[:, 2],
                   s=100, lw=1.,
                   color='C0',
                   zorder=1.,
                   edgecolors='gray'
                   )

        ax.scatter(self.pareto_ground[:, 0],
                   self.pareto_ground[:, 1],
                   self.pareto_ground[:, 2],
                   marker='o', s=100, c='C1', lw=1.,
                   alpha=0.8,
                   zorder=10.,
                   edgecolors='gray',
                   )

        xmin = np.min(self.objective_values_ground[:, 0])
        xmax = np.max(self.objective_values_ground[:, 0])
        ymin = np.min(self.objective_values_ground[:, 1])
        ymax = np.max(self.objective_values_ground[:, 1])
        zmin = np.min(self.objective_values_ground[:, 2])
        zmax = np.max(self.objective_values_ground[:, 2])

        ax.axes.set_xlim3d(xmin, xmax)
        ax.axes.set_ylim3d(ymin, ymax)
        ax.axes.set_zlim3d(zmin, zmax)

        ax.set_xlabel(f"{self.objective_names[0]} ({self.objective_modes[0]})")
        ax.set_ylabel(f"{self.objective_names[1]} ({self.objective_modes[1]})")
        ax.set_zlabel(f"{self.objective_names[2]} ({self.objective_modes[2]})")

        if not os.path.exists(f"{self.run_folder}_plots"):
            os.mkdir(f"{self.run_folder}_plots")

        plt.savefig(f"{self.run_folder}_plots/{self.filename_results}_ground.svg")

        plt.show()

    def _plot_train_pareto_2d(self):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.cumulative_train_y[:, 0],
                   self.cumulative_train_y[:, 1], s=30, lw=1.,
                   edgecolors='black')
        ax.scatter(self.pareto_train[:, 0], self.pareto_train[:, 1],
                   s=100, c='orange', lw=1., edgecolors='black')
        ax.scatter(self.pareto_ground[:, 0], self.pareto_ground[:, 1],
                   marker='x', s=100, c='black', lw=1.)

        ax.set_xlabel(f"{self.objective_names[0]} ({self.objective_modes[0]})")
        ax.set_ylabel(f"{self.objective_names[1]} ({self.objective_modes[1]})")

        if not os.path.exists(f"{self.run_folder}_plots"):
            os.mkdir(f"{self.run_folder}_plots")

        plt.savefig(f"{self.run_folder}_plots/{self.filename_results}_train_{self.step}.svg")

        plt.show()

    def _plot_train_pareto_3d(self):

        plt.rcParams['grid.color'] = "lightgray"
        plt.rcParams['grid.linestyle'] = [6., 2., 6., 2.]
        plt.rcParams['grid.linewidth'] = 1.5

        # Scatter plots 3d.
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        # Plot properties.
        ax.view_init(elev=25, azim=290)
        fig.set_facecolor('white')
        ax.set_facecolor('white')

        ax.scatter(self.cumulative_train_y[:, 0],
                   self.cumulative_train_y[:, 1],
                   self.cumulative_train_y[:, 2],
                   s=100, lw=1.,
                   color='C0',
                   zorder=1.,
                   edgecolors='gray'
                   )

        ax.scatter(self.pareto_ground[:, 0],
                   self.pareto_ground[:, 1],
                   self.pareto_ground[:, 2],
                   marker='o', s=100, c='C1', lw=1.,
                   alpha=0.8,
                   zorder=10.,
                   edgecolors='gray',
                   )

        xmin = np.min(self.objective_values_ground[:, 0])
        xmax = np.max(self.objective_values_ground[:, 0])
        ymin = np.min(self.objective_values_ground[:, 1])
        ymax = np.max(self.objective_values_ground[:, 1])
        zmin = np.min(self.objective_values_ground[:, 2])
        zmax = np.max(self.objective_values_ground[:, 2])

        ax.axes.set_xlim3d(xmin, xmax)
        ax.axes.set_ylim3d(ymin, ymax)
        ax.axes.set_zlim3d(zmin, zmax)

        ax.set_xlabel(f"{self.objective_names[0]} ({self.objective_modes[0]})")
        ax.set_ylabel(f"{self.objective_names[1]} ({self.objective_modes[1]})")
        ax.set_zlabel(f"{self.objective_names[2]} ({self.objective_modes[2]})")

        if not os.path.exists(f"{self.run_folder}_plots"):
            os.mkdir(f"{self.run_folder}_plots")

        plt.savefig(f"{self.run_folder}_plots/{self.filename_results}_train_{self.step}.svg")

        plt.show()


