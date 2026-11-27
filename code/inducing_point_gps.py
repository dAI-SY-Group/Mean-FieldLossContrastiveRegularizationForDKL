# -*- coding: utf-8 -*-
"""
Created on Mon Sept 12, 2023

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import typing

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    TriangularLinearOperator,
)
from sklearn import cluster
from tqdm import tqdm
from utils import prepare_targets

# --------------------------------------------------------------------- #
# methods to generate initial inducing points
# --------------------------------------------------------------------- #


@torch.no_grad()
def cluster_minibatch_inducing_points(
    datamodule,
    feature_extractor,
    n_inducing_points: int = 64,
    compute_from_n_points: int = 1000,
    device: str = "cpu",
    n_inducing_point_sets: int = 1,
):
    """
    https://github.com/y0ast/DUE/blob/main/due/dkl.py
    """
    datamodule = datamodule(
        batch_size=compute_from_n_points // 5,
        num_workers_train=1,
        num_workers_test=1,
        cluster=torch.cuda.is_available(),
        normalize=True,
        transform="Base",
        insert_corrupted_data=False,
    )
    datamodule.setup(stage="fit")
    feature_extractor.to(device)
    f_X_samples = []
    for x, y in datamodule.train_dataloader():
        x = x.to(device)
        embedded_x = feature_extractor(x)
        f_X_samples.append(embedded_x)
        if (
            len(f_X_samples) * x.shape[0]
        ) > n_inducing_point_sets * compute_from_n_points:
            break
    f_X_samples = torch.cat(f_X_samples)[
        : n_inducing_point_sets * compute_from_n_points
    ]

    print("Inducing Point computation via Kmeans finished!")

    initial_inducing_points_list = []
    initial_lengthscale = 0.0
    for i in range(n_inducing_point_sets):
        kmeans = cluster.MiniBatchKMeans(
            n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
        )
        kmeans.fit(f_X_samples.cpu().numpy())
        initial_inducing_points_list.append(torch.from_numpy(kmeans.cluster_centers_))
        initial_lengthscale += torch.pdist(f_X_samples.to(device)).mean()

    if len(initial_inducing_points_list) > 1:
        initial_inducing_points = torch.stack(initial_inducing_points_list)
        initial_lengthscale /= n_inducing_point_sets
    else:
        initial_inducing_points = initial_inducing_points_list[0]

    print("Inducing Point computation via Kmeans finished!")
    return initial_lengthscale.to(device), initial_inducing_points.to(device)


@torch.no_grad()
def get_forward_grid_ips(
    datamodule,
    feature_extractor,
    n_classes: int = 10,
    n_additional_ips: int = 10,
    compute_from_n_points: int = 10,
    device: str = "cpu",
):
    """
    https://github.com/y0ast/DUE/blob/main/due/dkl.py
    """
    datamodule = datamodule(
        batch_size=100,
        num_workers_train=1,
        num_workers_test=1,
        normalize=True,
        transform="Base",
        insert_corrupted_data=False,
    )
    datamodule.setup(stage="fit")
    feature_extractor.to(device)
    class_wise_samples = {i: [] for i in range(n_classes)}
    continue_sampling = np.ones(n_classes).astype(bool)
    for x, y in datamodule.train_dataloader():
        x = x.to(device)
        embedded_x = feature_extractor(x)
        if continue_sampling.any():
            for i, y_i in enumerate(y):
                if len(class_wise_samples[y_i.item()]) <= compute_from_n_points:
                    class_wise_samples[y_i.item()].append(embedded_x[i])
                else:
                    continue_sampling[y_i.item()] = False
        else:
            break

    image_dims = tuple(x.shape[1:])
    is_bw = image_dims[0] == 1
    images = []
    for j in range(n_additional_ips):
        image, _ = random_shapes(
            (image_dims[1], image_dims[2]),
            min_shapes=5,
            max_shapes=10,
            min_size=image_dims[1] / 3,
            allow_overlap=True,
        )
        if is_bw:
            image = rgb2gray(image)
            image = torch.from_numpy(image).unsqueeze(0).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        image += torch.randn_like(image) * 0.1  # Add some noise to the image
        images.append(image)

    ips = [torch.stack(m).mean(dim=0) for m in class_wise_samples.values()]
    ips = torch.stack(ips)

    if len(images) > 0:
        other_samples = torch.stack(images).float()
        other_samples = other_samples - other_samples.mean(
            dim=(1, 2, 3), keepdim=True
        ) / other_samples.std(dim=(1, 2, 3), keepdim=True)
        other_feature_vectors = feature_extractor(other_samples.to(device))
        ips = torch.cat([ips, other_feature_vectors], dim=0)
    return ips


# --------------------------------------------------------------------- #
# gpytorch based models for reference
# --------------------------------------------------------------------- #


def get_kernel(kernel: str, initial_lengthscale: float = 1.0, **kwargs):
    if kernel == "RBF":
        kernel = gpytorch.kernels.RBFKernel(**kwargs)
    elif kernel == "Matern12":
        kernel = gpytorch.kernels.MaternKernel(nu=1 / 2, **kwargs)
    elif kernel == "Matern32":
        kernel = gpytorch.kernels.MaternKernel(nu=3 / 2, **kwargs)
    elif kernel == "Matern52":
        kernel = gpytorch.kernels.MaternKernel(nu=5 / 2, **kwargs)
    elif kernel == "RQ":
        kernel = gpytorch.kernels.RQKernel(**kwargs)
    else:
        raise ValueError("Specified kernel not known.")

    kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)
    return kernel


class GpytorchGPLayerGridVariationalStrategy(gpytorch.models.ApproximateGP):
    """
    https://docs.gpytorch.ai/en/latest/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
    """

    def __init__(self, num_dim, grid_bounds=(-10.0, 10.0), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self,
                    grid_size=grid_size,
                    grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ),
                num_tasks=num_dim,
            )
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GpytorchGPLayerVariationalStrategySM(gpytorch.models.ApproximateGP):
    """
    https://github.com/y0ast/DUE/blob/main/due/dkl.py
    """

    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        kernel="RBF",
    ):
        n_inducing_points = initial_inducing_points.shape[0]

        if num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = (
                gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    variational_strategy, num_tasks=num_outputs
                )
            )

        super().__init__(variational_strategy)

        kwargs = {
            "batch_shape": batch_shape,
        }

        kernel = get_kernel(kernel, initial_lengthscale, **kwargs)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel, batch_shape=batch_shape
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param


class Variational(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        num_tasks: int,
        num_features: int,
        num_inducing: int,
        saturation: typing.Callable[..., torch.nn.Module] = torch.nn.Tanh(),
        transform_chol_diag: typing.Optional[
            typing.Union[torch.nn.Module, typing.Callable]
        ] = torch.nn.Softplus(),
        transform_chol_lower: typing.Optional[
            typing.Union[torch.nn.Module, typing.Callable]
        ] = None,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_features = num_features
        self.num_inducing = num_inducing
        self.saturation = saturation
        self.transform_chol_diag = transform_chol_diag
        self.transform_chol_lower = transform_chol_lower

        self.chol_lower_dim = num_inducing * (num_inducing - 1) // 2
        self._chol_diag = torch.arange(num_inducing)
        self._chol_lower = torch.tril_indices(num_inducing, num_inducing, offset=-1)

        num_inducing_units = num_features * num_inducing
        num_mean_units = num_tasks * num_inducing
        num_chol_units = self.chol_lower_dim * num_tasks

        self.initialize_projections(
            in_features=in_features,
            num_inducing_units=num_inducing_units,
            num_mean_units=num_mean_units,
            num_chol_units=num_chol_units,
        )
        self.reset_parameters()

    def initialize_projections(
        self,
        in_features: int,
        num_inducing_units: int,
        num_mean_units: int,
        num_chol_units: int,
    ):
        self.induc_proj = torch.nn.Linear(in_features, num_inducing_units)
        self.mean_proj = torch.nn.Linear(in_features, num_mean_units)
        self.chol_diag_proj = torch.nn.Linear(in_features, num_mean_units)
        self.chol_lower_proj = torch.nn.Linear(in_features, num_chol_units)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def build_chol_variational_covar(self, diag: torch.Tensor, lower: torch.Tensor):
        if self.transform_chol_diag is not None:
            diag = self.transform_chol_diag(diag)

        if self.transform_chol_lower is not None:
            lower = self.transform_chol_lower(lower)

        shape = (diag.size(0), self.num_tasks, self.num_inducing, self.num_inducing)
        chol_var_covar = torch.zeros(shape, dtype=diag.dtype, device=diag.device)
        chol_var_covar[..., self._chol_diag, self._chol_diag] = diag[..., :]
        chol_var_covar[..., self._chol_lower[0], self._chol_lower[1]] = lower[..., :]
        return chol_var_covar

    def compute_variational_parameters(
        self, x: torch.Tensor, **kwargs
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_data = x.size(0)

        x = self.saturation(x)
        inducing_points = self.induc_proj(x, **kwargs).reshape(
            [num_data, self.num_inducing, self.num_features]
        )
        variational_mean = self.mean_proj(x, **kwargs).reshape(
            [num_data, self.num_tasks, self.num_inducing]
        )
        chol_diag = self.chol_diag_proj(x, **kwargs).reshape(
            [num_data, self.num_tasks, self.num_inducing]
        )
        chol_lower = self.chol_lower_proj(x, **kwargs).reshape(
            [num_data, self.num_tasks, self.chol_lower_dim]
        )
        chol_var_covar = self.build_chol_variational_covar(chol_diag, chol_lower)

        return (
            inducing_points,
            variational_mean.transpose(0, 1),
            chol_var_covar.transpose(0, 1),
        )

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute_variational_parameters(x, **kwargs)


class AmortizedSVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        num_inducing: int,
        num_tasks: int,
        mean_field: bool = False,
    ):
        if mean_field:
            variational_distribution = AmortizedMeanFieldVariationalDistribution(
                num_inducing, torch.Size([num_tasks])
            )
        else:
            variational_distribution = AmortizedVariationalDistribution(
                num_inducing, torch.Size([num_tasks])
            )
        variational_strategy = AmortizedVariationalStrategy(
            self, variational_distribution
        )

        if num_tasks > 1:
            variational_strategy = AmortizedMultitaskVariationalStrategy(
                variational_strategy, num_tasks
            )

        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.num_inducing = num_inducing
        self.num_tasks = num_tasks

    def set_variational_parameters(
        self, Z: torch.Tensor, m: torch.Tensor, L: torch.Tensor
    ):
        if self.num_tasks > 1:
            self.variational_strategy.base_variational_strategy.inducing_points = Z
            self.variational_strategy.base_variational_strategy._variational_distribution.set_parameters(
                m, L
            )
        else:
            self.variational_strategy.inducing_points = Z
            self.variational_strategy._variational_distribution.set_parameters(m, L)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        function_mean = self.mean_module(x)
        function_covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(function_mean, function_covar)


class AmortizedSGP(gpytorch.Module):
    def __init__(
        self,
        num_outputs,
        saturation_str="tanh",
        bottleneck=2,
        n_classes=10,
        num_inducing=10,
        kernel="RBF",
    ):
        super().__init__()
        saturation_activation = (
            torch.nn.Tanh() if saturation_str == "tanh" else torch.nn.Sigmoid()
        )
        bottleneck = num_outputs // bottleneck
        saturation = torch.nn.Sequential(
            torch.nn.Linear(num_outputs, bottleneck), saturation_activation
        )
        self.variational_estimator = Variational(
            in_features=bottleneck,
            num_tasks=n_classes,
            num_features=num_outputs,
            num_inducing=num_inducing,
            saturation=saturation,
        )

        kernel = get_kernel(kernel)

        self.gplayer = AmortizedSVGP(
            mean_module=gpytorch.means.ZeroMean(),
            covar_module=kernel,
            num_inducing=num_inducing,
            num_tasks=n_classes,
        )

    def parameters(self):
        return [
            {"params": self.variational_estimator.parameters()},
            {"params": self.gplayer.hyperparameters(), "lr": 0.001},
        ]

    def forward(self, x: torch.Tensor):
        Z, m, L = self.variational_estimator(x)
        self.gplayer.set_variational_parameters(Z, m, L)
        function_dist = self.gplayer(x.unsqueeze(-2))
        return function_dist


class AmortizedVariationalStrategy(gpytorch.variational.VariationalStrategy):
    def __init__(self, model, variational_distribution, jitter_val=1e-4):
        super().__init__(
            model,
            torch.Tensor(0),
            variational_distribution,
            jitter_val=jitter_val,
            learn_inducing_locations=False,
        )

        delattr(self, "inducing_points")
        self.inducing_points = torch.Tensor()
        self.variational_params_initialized.fill_(1)

    def kl_divergence(self):
        return super().kl_divergence().mean(-1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        function_dist = super().forward(x, *args, **kwargs)
        function_mean = function_dist.mean.squeeze(-1)
        function_covar = DiagLinearOperator(
            function_dist.covariance_matrix.squeeze(-1).squeeze(-1)
        )
        return gpytorch.distributions.MultivariateNormal(function_mean, function_covar)

    def __call__(self, x: torch.Tensor, prior=False, **kwargs):
        self._clear_cache()
        return super().__call__(x, prior=prior, **kwargs)


class AmortizedMultitaskVariationalStrategy(
    gpytorch.variational.IndependentMultitaskVariationalStrategy
):
    def kl_divergence(self):
        return self.base_variational_strategy.kl_divergence().sum()


class AmortizedMeanFieldVariationalDistribution(
    gpytorch.variational._VariationalDistribution
):
    def __init__(
        self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3
    ):
        super().__init__(num_inducing_points, batch_shape, mean_init_std)
        self.inducing_points = torch.Tensor()
        self.variational_mean = torch.Tensor()
        self.variational_stddev = torch.Tensor()

    def set_parameters(self, variational_mean, variational_stddev):
        self.variational_mean = variational_mean
        self.variational_stddev = variational_stddev

    def forward(self):
        variational_covar = DiagLinearOperator(self.variational_stddev.pow(2))
        return gpytorch.variational.MultivariateNormal(
            self.variational_mean, variational_covar
        )

    def initialize_variational_distribution(self, *args, **kwargs):
        pass

    def shape(self):
        return self.variational_mean.shape[-2:]

    @property
    def dtype(self):
        return self.variational_mean.dtype

    @property
    def device(self):
        return self.variational_mean.device


class AmortizedVariationalDistribution(gpytorch.variational._VariationalDistribution):
    def __init__(
        self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3
    ):
        super().__init__(
            num_inducing_points=num_inducing_points,
            batch_shape=batch_shape,
            mean_init_std=mean_init_std,
        )

        self.inducing_points = torch.Tensor()
        self.variational_mean = torch.Tensor()
        self.variational_covar = torch.Tensor()

    def forward(self):
        return gpytorch.distributions.MultivariateNormal(
            self.variational_mean, self.variational_covar
        )

    def initialize_variational_distribution(self, *args, **kwargs):
        pass

    def shape(self):
        return self.variational_mean.shape[-2:]

    @property
    def dtype(self):
        return self.variational_mean.dtype

    @property
    def device(self):
        return self.variational_mean.device

    def set_parameters(self, variational_mean, chol_variational_covar):
        self._set_variational_mean(variational_mean)
        self._set_chol_variational_covar(chol_variational_covar)

    def _set_variational_mean(self, variational_mean):
        self.variational_mean = variational_mean

    def _set_chol_variational_covar(self, chol_variational_covar):
        self.variational_covar = CholLinearOperator(
            TriangularLinearOperator(chol_variational_covar)
        )


def psd_safe_cholesky(A, jitter=None, max_tries=None):
    L, info = torch.linalg.cholesky_ex(A)
    if not torch.any(info):
        return L, 0

    isnan = torch.isnan(A)
    if isnan.any():
        raise ValueError(
            f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
        )

    if jitter is None:
        jitter = 1e-9
    if max_tries is None:
        max_tries = 15
    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        # add jitter only where needed
        diag_add = (
            ((info > 0) * (jitter_new - jitter_prev))
            .unsqueeze(-1)
            .expand(*Aprime.shape[:-1])
        )
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        L, info = torch.linalg.cholesky_ex(Aprime)
        if not torch.any(info):
            return L, i
    raise ValueError(
        f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}."
    )


def get_kernel_function(kernel_desc: str):
    if kernel_desc == "scaled_linear" or kernel_desc == "scaled_quadratic":
        exponent_k = 1 if kernel_desc == "scaled_linear" else 2

        def fn(
            x: torch.Tensor,
            ips: torch.Tensor,
            scale: torch.Tensor,
            var: torch.Tensor,
            use_ips: bool = True,
            exponent: int = exponent_k,
        ) -> torch.Tensor:
            # scale = torch.log1p(torch.exp(scale))
            if not use_ips:
                return (
                    scale
                    * ((x * var.reshape(1, -1, 1) * x) ** exponent).sum(1).squeeze()
                )
            else:
                return scale * ((x * var.reshape(1, -1, 1) * ips) ** exponent).sum(1)
    elif kernel_desc == "scaled_cosine":

        def fn(
            x: torch.Tensor,
            ips: torch.Tensor,
            scale: torch.Tensor,
            var: torch.Tensor,
            use_ips: bool = True,
        ) -> torch.Tensor:
            if not use_ips:
                return (
                    scale * (x * var.reshape(1, -1, 1) * x).sum(1) / torch.norm(x) ** 2
                ).squeeze()
            else:
                return (
                    scale
                    * (x * var.reshape(1, -1, 1) * ips).sum(1)
                    / torch.norm(x)
                    / torch.norm(ips)
                )
    elif kernel_desc == "scaled_l1_difference":

        def fn(
            x: torch.Tensor,
            ips: torch.Tensor,
            scale: torch.Tensor,
            var: torch.Tensor,
            use_ips: bool = True,
        ) -> torch.Tensor:
            # scale = torch.log1p(torch.exp(scale))
            if not use_ips:
                return scale * torch.ones(x.size(0), device=x.device)
            else:
                return (
                    scale
                    * ((x - ips) * var.reshape(1, -1, 1)).abs().sum(1).mul(-1).exp()
                )
    elif kernel_desc == "scaled_l2_difference":

        def fn(
            x: torch.Tensor,
            ips: torch.Tensor,
            scale: torch.Tensor,
            var: torch.Tensor,
            use_ips: bool = True,
        ) -> torch.Tensor:
            # scale = torch.log1p(torch.exp(scale))
            if not use_ips:
                return scale * torch.ones(x.size(0), device=x.device)
            else:
                return (
                    scale
                    * ((x - ips) * var.reshape(1, -1, 1)).square().sum(1).mul(-1).exp()
                )
    elif kernel_desc == "scaled_l2_difference_minmax":

        def fn(
            x: torch.Tensor,
            ips: torch.Tensor,
            scale: torch.Tensor,
            var: torch.Tensor,
            use_ips: bool = True,
        ) -> torch.Tensor:
            # scale = torch.log1p(torch.exp(scale))
            if not use_ips:
                return scale * torch.ones(x.size(0), device=x.device)
            else:
                diff = x - ips
                q = torch.tensor([0.05, 0.95], device=diff.device)
                quantile = torch.quantile(diff, q, dim=1, keepdim=True)
                diff = torch.minimum(quantile[1], diff)
                idx = diff < quantile[0]
                diff[idx] = 0.0
                return (
                    scale * (diff * var.reshape(1, -1, 1)).square().sum(1).mul(-1).exp()
                )
    else:
        raise NotImplementedError(f"Kernel {kernel_desc} not implemented")
    return torch.jit.script(fn)


class IPModule(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        inducing_points: torch.Tensor,
        alpha_eps: float = 0.0,
        kernel: str = "scaled_linear",
        learnable_kernel_variance: bool = False,
        fixed_kernel_var: float = 1.0,
        extra_linear: int = 0,
        psd_safe_cholesky: bool = True,
        diagonal_q_u_covar: bool = True,
        reg: float = 1.0,
        debug_mode: bool = False,
        whiten: bool = False,
        temperature: float = 1.0,
        loss_type: str = "gaussian_nll",
        inducing_points_method: str = "vectors",
        inducing_points_reg: float = 0.1,
        start_zero_q_u_mean: bool = False,
        test_jitter: float = 1e-9,  # formally 1e-6
        train_jitter: float = 1e-9,
        normalize_gp_features: bool = False,
        scale_cce_logits: float = 1.0,
        smoothing: float = 0.0,
    ):
        super().__init__()
        # general parameter
        start_zero_q_u_mean = (
            start_zero_q_u_mean
            if isinstance(start_zero_q_u_mean, bool)
            else (True if start_zero_q_u_mean == "true" else False)
        )
        fixed_kernel_var = float(fixed_kernel_var)
        self.in_features = in_features
        self.n_classes = n_classes
        self.n_inducing_points = inducing_points.size(0)
        self.reg = float(
            reg
        )  # regularization parameter for the KL divergence: 1 / dataset_size
        self.debug_mode = debug_mode
        self.debug_values = {}
        self.whiten = (
            whiten
            if isinstance(whiten, bool)
            else (True if whiten == "true" else False)
        )
        self.temperature = temperature
        self.loss_type = loss_type
        self.special_inducing_points = (
            inducing_points_method != "vectors"
        )
        self.inducing_points_reg = float(inducing_points_reg)
        self.test_jitter = float(test_jitter)
        self.train_jitter = float(train_jitter)
        self.normalize_gp_features = (
            normalize_gp_features
            if isinstance(normalize_gp_features, bool)
            else (True if normalize_gp_features == "true" else False)
        )
        if self.normalize_gp_features:
            self.norm_module = torch.nn.LayerNorm(
                in_features if not extra_linear else extra_linear
            )

        # loss related parameter
        self.alpha_eps = float(alpha_eps)
        self.scale_cce_logits = float(scale_cce_logits)
        self.label_smoothing = float(smoothing)

        # kernel definition
        self.kernel_desc = kernel
        self.kernel = get_kernel_function(kernel)
        # kernel parameter
        # self.kernel_scale = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_buffer("kernel_scale", torch.ones(1))
        self.learnable_kernel_variance = bool(learnable_kernel_variance)

        # mean of the gp
        self.constant_mean = torch.nn.Parameter(
            torch.zeros(n_classes), requires_grad=True
        )

        # inducing points function output
        # q_u
        self.diagonal_q_u_covar = (
            diagonal_q_u_covar
            if isinstance(diagonal_q_u_covar, bool)
            else (True if diagonal_q_u_covar == "true" else False)
        )
        covariance = (
            torch.eye(self.n_inducing_points).repeat(n_classes, 1, 1)
            if not diagonal_q_u_covar
            else torch.ones(n_classes, self.n_inducing_points)
        )
        q_u_mean = prepare_targets(
            targets=torch.arange(n_classes), n_classes=n_classes, alpha_eps=alpha_eps
        )[1]
        # q_u_mean = torch.nn.functional.one_hot(torch.arange(n_classes), n_classes).float() + 0.01
        if self.n_inducing_points > n_classes:
            q_u_mean = torch.cat(
                [q_u_mean, torch.randn(self.n_inducing_points - n_classes, n_classes)]
            )
        factor = (
            0.0
            if (self.loss_type.__contains__("regression") or start_zero_q_u_mean)
            else 1.0
        )
        self.q_u_mean = torch.nn.Parameter(factor * q_u_mean, requires_grad=True)
        # self.q_u_mean = torch.nn.Parameter(torch.rand(self.n_inducing_points, n_classes), requires_grad=True)
        # self.register_buffer('q_u_mean', q_u_mean)
        # self.register_buffer('q_u_halfvar', covariance)
        self.q_u_halfvar = torch.nn.Parameter(covariance, requires_grad=True)

        # p_u
        self.register_buffer("p_u_mean", q_u_mean.clone())
        self.register_buffer(
            "p_u_covar", torch.eye(self.n_inducing_points).unsqueeze(0)
        )

        # extra projection as in contrastive learning
        self.extra_linear = int(extra_linear)
        if self.extra_linear:
            self.projection = torch.nn.Linear(in_features, self.extra_linear)
            if inducing_points_method == "vectors":
                self.inducing_points = torch.nn.Parameter(
                    inducing_points[:, : self.extra_linear], requires_grad=True
                )
            else:
                raise NotImplementedError(
                    f"Inducing points method {inducing_points_method} not implemented"
                )
        else:
            self.projection = torch.nn.Identity()
            if inducing_points_method == "vectors":
                self.inducing_points = torch.nn.Parameter(
                    inducing_points, requires_grad=True
                )
            else:
                raise NotImplementedError(
                    f"Inducing points method {inducing_points_method} not implemented"
                )
        self.psd_safe_cholesky = psd_safe_cholesky

        if learnable_kernel_variance:
            self.kernel_logvar = torch.nn.Parameter(
                fixed_kernel_var
                * torch.ones(self.inducing_points.size(1), requires_grad=True)
            )
        else:
            self.register_buffer(
                "kernel_logvar",
                fixed_kernel_var * torch.ones(self.inducing_points.size(1)),
            )

    def get_q_u_covar(self):
        if self.diagonal_q_u_covar:
            out = []
            for c in self.q_u_halfvar:
                # if c.max() > 1.:
                #     c = c / c.max()
                out.append(torch.diag(c**2))
            return torch.stack(out)
        else:
            mask = (
                torch.ones(
                    self.n_inducing_points,
                    self.n_inducing_points,
                    device=self.q_u_mean.device,
                )
                .tril(0)
                .unsqueeze(0)
            )
            return torch.bmm(
                self.q_u_halfvar * mask, (self.q_u_halfvar * mask).transpose(1, 2)
            )

    def get_inducing_points(self):
        ips = (
            self.inducing_points()
            if self.special_inducing_points
            else self.inducing_points
        )
        if self.normalize_gp_features:
            ips = self.norm_module(ips)
        return ips  # torch.nn.functional.normalize(ips, dim=-1)

    def kl_q_u(self):
        q_u_covar = self.get_q_u_covar()
        q_u = torch.torch.distributions.MultivariateNormal(self.q_u_mean.T, q_u_covar)
        p_u = torch.distributions.MultivariateNormal(self.p_u_mean.T, self.p_u_covar)
        return torch.distributions.kl.kl_divergence(q_u, p_u)

    def evaluate_kernel(self, x, x_ref):
        x, x_ref = self.projection(x), self.projection(x_ref)
        var = torch.log1p(torch.exp(self.kernel_logvar))
        k_xxref = self.kernel(
            x.unsqueeze(-1), x_ref.unsqueeze(0), self.kernel_logscale, var
        )
        return k_xxref.squeeze()

    def linear_with_ip(self, x):
        ips = self.get_inducing_points()
        return x @ ips.transpose(0, 1)

    def forward(self, x):
        # z for the inducing points
        if self.extra_linear:
            x = self.projection(x)
        if self.normalize_gp_features:
            x = self.norm_module(x)
        var = torch.maximum(self.kernel_logvar, torch.zeros_like(self.kernel_logvar))
        ips = self.get_inducing_points()
        kernel_logscale = (
            self.kernel_scale if self.training else self.kernel_scale.clone()
        )
        kernel_logscale = torch.maximum(
            kernel_logscale, torch.zeros_like(kernel_logscale)
        )
        k_xz = self.kernel(
            x.unsqueeze(-1), ips.T.unsqueeze(0), kernel_logscale, var.unsqueeze(-1)
        )
        k_xx = self.kernel(
            x.unsqueeze(-1), ips.unsqueeze(0), kernel_logscale, var, False
        )
        k_zz = self.kernel(
            ips.unsqueeze(-1), ips.T.unsqueeze(0), kernel_logscale, var.unsqueeze(-1)
        ).T

        jitter = self.train_jitter if self.training else self.test_jitter
        L, i = (
            psd_safe_cholesky(k_zz, jitter=jitter)
            if self.psd_safe_cholesky
            else torch.linalg.cholesky(k_zz)
        )

        if self.debug_mode:
            self.debug_values["cholesky_tries"] = i

        if not self.whiten:
            # during testing L can be cached
            interpolation = torch.cholesky_solve(self.q_u_mean, L, upper=False)
            pred_mean = k_xz @ interpolation + self.constant_mean.unsqueeze(0)

            q_u_covar = self.get_q_u_covar()
            interpolation = torch.cholesky_solve(k_xz.T, L, upper=False).transpose(
                -1, -2
            )[:, None, None, :]
            pred_cov = (k_xx + 1e-7).unsqueeze(-1) + (
                interpolation
                @ (q_u_covar - k_zz.unsqueeze(0))
                @ interpolation.transpose(2, 3)
            ).squeeze(-2, -1)
        else:
            # during testing L can be cached
            interpolation = torch.linalg.solve_triangular(
                L.T, self.q_u_mean, upper=True
            )
            pred_mean = k_xz @ interpolation + self.constant_mean.unsqueeze(0)

            q_u_covar = self.get_q_u_covar()
            # Qinv = torch.cholesky_solve(q_u_covar - self.p_u_covar, L)
            B = self.p_u_covar - q_u_covar
            LinvT_B = torch.linalg.solve_triangular(L.T, B, upper=True)
            Qinv = torch.linalg.solve_triangular(
                L.T, LinvT_B.transpose(-2, -1), upper=True
            )

            k_xz = k_xz[:, None, None, :]  # add batch and class dimension
            pred_cov = (k_xx + 1e-7).unsqueeze(-1) - (
                k_xz @ Qinv @ k_xz.transpose(-2, -1)
            ).squeeze(-2, -1)

        if self.loss_type.__contains__("mfcce"):
            mf_scale = float(self.loss_type[5:]) if len(self.loss_type) > 5 else 1.0
            logits_scale = torch.sqrt(1.0 + pred_cov * mf_scale)
            mf_logits = pred_mean / logits_scale
            return mf_logits, pred_cov

        return pred_mean, pred_cov

    def get_loss(self, x, y):
        pred_mean, pred_cov = self.forward(x)
        loss = self.loss_forward(pred_mean, pred_cov, y)
        return loss

    def loss_forward(self, pred_mean, pred_cov, y, debug=True, loss_extensions=True):
        if self.debug_mode and debug:
            self.debug_values["max_pred_mean"] = float(pred_mean.max())
            self.debug_values["min_pred_mean"] = float(pred_mean.min())
            self.debug_values["max_cum_var"] = float(pred_cov.max())
            self.debug_values["min_cum_var"] = float(pred_cov.min())

        if self.loss_type == "gaussian_nll":
            sigma_y, y_trans = prepare_targets(
                y, n_classes=self.n_classes, alpha_eps=self.alpha_eps
            )
            sigma_y = sigma_y.T
            res = (
                ((y_trans.T - pred_mean) ** 2 + pred_cov) / sigma_y
                + sigma_y.log()
                + math.log(2 * math.pi)
            )
            loss = res.mul(0.5).mean()

        elif self.loss_type == "cce":
            loss = torch.nn.functional.cross_entropy(
                self.scale_cce_logits * pred_mean,
                y,
                label_smoothing=self.label_smoothing,
            )

        elif self.loss_type.__contains__("mfcce"):
            cov_scale = float(self.loss_type[5:]) if len(self.loss_type) > 5 else 1.0
            # unc = torch.sqrt(1 + pred_cov * (math.pi / 8))
            unc = torch.sqrt(1 + cov_scale * pred_cov)
            loss = torch.nn.functional.cross_entropy(
                self.scale_cce_logits * pred_mean / unc,
                y,
                label_smoothing=self.label_smoothing,
            )

        elif self.loss_type.__contains__("mccce"):
            # does not work!
            n_mc = int(self.loss_type[5:]) if len(self.loss_type) > 5 else 5
            scores = pred_mean + pred_cov * torch.randn(
                n_mc, pred_mean.size(0), pred_mean.size(1), device=pred_mean.device
            )
            prob_scores = torch.nn.functional.softmax(scores, dim=-1).mean(0)
            probs_at_targets = prob_scores[:, y]
            loss = -torch.log(probs_at_targets).mean()

        elif self.loss_type.__contains__("umargin_cce"):
            if len(self.loss_type[11:]) > 0:
                max_u = float(self.loss_type[11:])
            else:
                max_u = 0.2
            unc = torch.sqrt(pred_cov + 1e-7)
            unc = unc * (max_u / unc.max())
            if y.dtype == torch.long:
                index = torch.zeros_like(pred_mean, dtype=torch.uint8)
                index.scatter_(1, y.data.view(-1, 1), 1)
            else:
                index = y
            batch_m = unc * index.float()
            pred = pred_mean - batch_m
            output = torch.where(index > 0, pred, pred_mean)
            return torch.nn.functional.cross_entropy(
                self.scale_cce_logits * output, y
            )  # 3 worked nice

        elif self.loss_type == "gaussian_nll_regression":
            loss = 0.5 * ((y.float() - pred_mean) ** 2 + pred_cov).mean()

        elif self.loss_type == "gaussian_pll_regression":
            marginal = Normal(pred_mean.flatten(), pred_cov.flatten() + 1e-7)
            loss = -marginal.log_prob(y.flatten()).mean()

        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        if self.debug_mode and debug:
            self.debug_values["loss_1"] = float(loss.item())

        if loss_extensions:
            loss += self.reg * self.kl_q_u().sum()

            if self.debug_mode and debug:
                self.debug_values["loss_2"] = float(
                    loss.item() - self.debug_values["loss_1"]
                )

        return loss
