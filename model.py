import gpytorch
import numpy as np
from rich.console import Console
import torch
from gpytorch.models import ExactGP


# extend the gptorch.models.ExactGP class
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(
        self, train_x, train_y, likelihood, kernel=gpytorch.kernels.RQKernel()
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # add some noise to the covraiance matrix for numerical stability
        #covar_x = covar_x.add_jitter(1e-3)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @classmethod
    def load_from_file(cls, file_path):

        # Load the dictionary from the file
        checkpoint = torch.load(file_path)

        # Retrieve the necessary components from the state dictionary
        # Assuming that the state dictionary includes the kernel type and parameters
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())  
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Restore the training inputs
        train_x = torch.tensor(checkpoint["train_x"]).float().cuda()
        train_y = torch.tensor(checkpoint["train_y"]).float().cuda()

        # Create an instance of the model with placeholder data
        model = cls(train_x, train_y, likelihood, kernel)

        # Load the state dictionary into the model
        model.load_state_dict(checkpoint["model"])

        return model
