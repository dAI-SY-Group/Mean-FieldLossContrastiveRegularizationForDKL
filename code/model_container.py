# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30, 2023

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import gpytorch
import numpy as np
import torch
from utils import get_gp, get_loss, get_model


def container_chooser(name):
    if name in ["SVDKL", "SVGP", "DUE", "DKL", "AVDKL"]:
        return DKL
    else:
        return DNN


class _BaseModelContainer(torch.nn.Module):
    def __int__(self):
        super().__init__()

    def parameter_list(self):
        raise NotImplementedError

    def train_value_names(self):
        # return list of strings with metric name
        raise NotImplementedError

    def test_value_names(self):
        raise NotImplementedError

    def forward(self, x):
        # insert all operations, that loss and predict share
        raise NotImplementedError

    def forward_loss(self, x, y, logger, inds):
        raise NotImplementedError

    def predict(self, x, y=None):
        raise NotImplementedError


class DNN(_BaseModelContainer):
    """
    This class wraps a classic deep neural network to be trained with the pipeline scipt
    """

    def __init__(self, name, model_args, loss_args, optimizer_args):
        super().__init__()
        self.name = name
        self.model = get_model(model_args)
        self.loss, _ = get_loss(loss_args)
        self.optimizer_args = optimizer_args["args"] if "args" in optimizer_args else {}

    def parameter_list(self):
        return self.model.parameters(), self.optimizer_args

    def train_value_names(self):
        return ["loss"]

    def test_value_names(self):
        return ["loss"]

    def forward(self, x):
        return self.model(x)

    def forward_loss(self, x, y, logger, inds):
        yhat = self.forward(x)
        return self.loss(yhat, y)

    @torch.no_grad()
    def predict(self, x, y=None):
        return self.forward(x)


class DKL(_BaseModelContainer):
    """
    This class wraps deep kernel learning models to be trained with the pipeline script

    The following DKL models are supported (the name is stored in the attribute name):
        - SVDKL
        - SVGP
        - DUE

    At the moment all of them are trained with the gpytorch SoftmaxLikelihood
    """

    def __init__(
        self,
        name,
        feature_extractor_args,
        gaussian_process_args,
        loss_args,
        optimizer_args,
        dataset: str,
        n_train_samples: int,
        n_classes: int,
        device: str,
    ):
        super().__init__()
        self.name = name
        self.feature_extractor, feature_size = get_model(feature_extractor_args, True)
        self.gaussian_process, self.feature_scaler = get_gp(
            gaussian_process_args,
            dataset,
            self.feature_extractor,
            device,
            n_classes,
            feature_size,
        )
        self.loss, self.likelihood = get_loss(
            loss_args,
            gaussian_process=self.gaussian_process,
            n_train_samples=n_train_samples,
            n_features=feature_size,
            n_classes=n_classes,
        )
        self.optimizer_args = optimizer_args["args"] if "args" in optimizer_args else {}
        self.num_likelihood_samples = (0, 0)
        if "num_likelihood_samples" in optimizer_args:
            self.num_likelihood_samples = optimizer_args["num_likelihood_samples"]

    def parameter_list(self):
        """Allowed keys of optimizer_args:
        - feature_extractor
        - gaussian_process
            - hyperparameter
            - variational
        - likelihood
        """
        parameters = []
        return_args = self.optimizer_args.copy()
        if "feature_extractor" in self.optimizer_args:
            args = self.optimizer_args["feature_extractor"]
            parameters.append({"params": self.feature_extractor.parameters(), **args})
            print("Extra feature extractor args: ", args)
            del return_args["feature_extractor"]
        else:
            parameters.append({"params": self.feature_extractor.parameters()})
        if "gaussian_process" in self.optimizer_args:
            args = self.optimizer_args["gaussian_process"]
            if "hyperparameter" in args:
                hp_args = args["hyperparameter"]
                parameters.append(
                    {"params": self.gaussian_process.hyperparameters(), **hp_args}
                )
                print("Extra hyperparameter args: ", hp_args)
            else:
                parameters.append(
                    {"params": self.gaussian_process.hyperparameters(), **args}
                )
            if "variational" in args:
                var_args = args["variational"]
                parameters.append(
                    {
                        "params": self.gaussian_process.variational_parameters(),
                        **var_args,
                    }
                )
                print("Extra variational args: ", var_args)
            else:
                parameters.append(
                    {"params": self.gaussian_process.variational_parameters(), **args}
                )
            del return_args["gaussian_process"]
        else:
            parameters.append({"params": self.gaussian_process.parameters()})
        if "likelihood" in self.optimizer_args:
            args = self.optimizer_args["likelihood"]
            parameters.append({"params": self.likelihood.parameters(), **args})
            print("Extra likelihood args: ", args)
            del return_args["likelihood"]
        else:
            if self.likelihood is None:
                pass
            else:
                parameters.append({"params": self.likelihood.parameters()})
        return parameters, return_args

    def train_value_names(self):
        return ["elbo"]

    def test_value_names(self):
        return ["elbo"]

    def gp_forward(self, x):
        if self.name == "SVDKL":
            with gpytorch.settings.use_toeplitz(False):
                return self.gaussian_process(x)
        else:
            return self.gaussian_process(x)

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = self.feature_scaler(feature)
        res = self.gp_forward(feature)
        return res

    def forward_loss(self, x, y, logger, inds):
        n_likelihood_samples = (
            self.num_likelihood_samples[0] if self.num_likelihood_samples[0] else 10
        )  # default=10
        with gpytorch.settings.num_likelihood_samples(n_likelihood_samples):
            yhat = self.forward(x)
            return self.loss(yhat, y)

    @torch.no_grad()
    def predict(self, x, y=None):
        n_likelihood_samples = (
            self.num_likelihood_samples[1] if self.num_likelihood_samples[1] else 10
        )
        with gpytorch.settings.num_likelihood_samples(n_likelihood_samples):
            if isinstance(self.likelihood, gpytorch.likelihoods.SoftmaxLikelihood):
                if self.name == "SVDKL":
                    out = self.likelihood(self.forward(x)).logits
                else:
                    out = self.likelihood(
                        self.forward(x).to_data_independent_dist()
                    ).logits
                # log p_i = log \sum_j exp(f_ij - log \sum_k f_kj) - log |J|
                denominator = torch.logsumexp(out, dim=-1, keepdim=True)
                exponent = out - denominator
                logits = torch.logsumexp(exponent, dim=0) - np.log(n_likelihood_samples)
                return torch.nn.functional.softmax(logits)
            elif isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood):
                pred_samples = (
                    self.likelihood(self.forward(x)).sample(torch.Size((100,))).exp()
                )
                # pred_samples = self.likelihood(self.forward(x)).loc
                probabilities = pred_samples / pred_samples.sum(-2, keepdim=True)
                return probabilities
            elif self.likelihood is None:
                return self.forward(x)
            else:
                raise NotImplementedError
