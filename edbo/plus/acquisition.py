
import torch
import numpy as np
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.optim import optimize_acqf_discrete
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
import pareto
from scipy.stats import norm


def acq_multiobjective_EHVI(model, train_y, test_x, ref_points):

    partitioning = NondominatedPartitioning(
        ref_point=ref_points,
        Y=torch.tensor(train_y))

    sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=False)

    EHVI = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_points,  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
    )

    acq = optimize_acqf_discrete(
        acq_function=EHVI,
        choices=test_x,
        q=1,
        unique=True
    )[0][0].detach().numpy().tolist()

    return acq


def acq_multiobjective_MOUCB(model, train_y, test_x, v=1, delta=.1,
                             greedy=False):

    # UCB paramters.
    t = np.shape(train_y)[0]  # t: number of iterations.
    d = np.shape(test_x)[1]  # d: number of dimensions.

    try:
        kappa = np.sqrt(v * (2 * np.log((t**(d/2. + 2))*(np.pi**2)/(3. * delta))))
    except:
        d = 10
        kappa = np.sqrt(v * (2 * np.log((t ** (d / 2. + 2)) * (np.pi ** 2) / (3. * delta))))

    if greedy is True:
        kappa = 0

    # Build pareto train.
    pareto_train_y = pareto.eps_sort(tables=train_y, maximize_all=True)
    pareto_train_y = np.reshape(pareto_train_y, (-1, model.num_outputs))

    # Predict in all models.
    means = model.posterior(test_x).mean.detach().numpy()
    variances = model.posterior(test_x).variance.detach().numpy()

    dmaximin = []
    for i in range(0, len(means)):
        diff_pareto_to_test = (means[i] + kappa * variances[i]) - pareto_train_y
        dmin_i = np.min(diff_pareto_to_test, axis=0)
        if greedy is False:
            dmin_i[dmin_i < 0] = 0
        dmaximin_i = np.max(dmin_i)
        dmaximin.append(dmaximin_i)

    best_sample = test_x[np.argmax(dmaximin)].detach().numpy().tolist()
    return best_sample


def acq_EI(y_best, predictions, uncertainty, objective='max'):
    """Return expected improvement acq. function.
    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    objective: str
        Choices are 'min' or 'max' for minimization and maximization respectively.
    """
    if objective == 'max':
        z = (predictions - y_best) / (uncertainty)
        return (predictions - y_best) * norm.cdf(z) + \
            uncertainty * norm.pdf(
            z)

    if objective == 'min':
        z = (-predictions + y_best) / (uncertainty)
        return -((predictions - y_best) * norm.cdf(z) -
                 uncertainty * norm.pdf(z))