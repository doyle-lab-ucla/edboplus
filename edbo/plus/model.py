
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
import numpy as np

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

def build_and_optimize_model(train_x, train_y):
    """ Builds model and optimizes it."""

    print('Using hyperparameters optimized for continuous variables.')
    gp_options = {
        'ls_prior1': 2.0, 'ls_prior2': 0.2, 'ls_prior3': 5.0,
        'out_prior1': 5.0, 'out_prior2': 0.5, 'out_prior3': 8.0,
        'noise_prior1': 1.5, 'noise_prior2': 0.1, 'noise_prior3': 5.0,
        'noise_constraint': 1e-5,
    }

    n_features = np.shape(train_x)[1]

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y,
                                               likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            kernels = MaternKernel(
                ard_num_dims=n_features,
                lengthscale_prior=GammaPrior(gp_options['ls_prior1'],
                                             gp_options['ls_prior2'])
            )

            self.covar_module = ScaleKernel(
                kernels,
                outputscale_prior=GammaPrior(gp_options['out_prior1'],
                                             gp_options['out_prior2']))
            try:
                ls_init = gp_options['ls_prior3']
                self.covar_module.base_kernel.lengthscale = ls_init
            except:
                uniform = gp_options['ls_prior3']
                ls_init = torch.ones(n_features).to(**tkwargs) * uniform
                self.covar_module.base_kernel.lengthscale = ls_init

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        GammaPrior(gp_options['noise_prior1'], gp_options['noise_prior2'])
    )

    likelihood.noise = gp_options['noise_prior3']
    model = ExactGPModel(train_x, train_y, likelihood).to(**tkwargs)

    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(gp_options['noise_constraint'])
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 1000
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.squeeze(-1).to(**tkwargs))
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood  # Optimized model

