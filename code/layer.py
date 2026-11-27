# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10, 2023

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import math
from typing import Optional

import torch


class SpectralBatchNorm(torch.nn.modules.batchnorm._NormBase):
    """stolen from https://github.com/y0ast/DUE/blob/main/due/layers/spectral_batchnorm.py
    BUT extended for training without running mean - what is considerably more stable against
    noisy input data!
    """

    def __init__(
        self,
        num_features,
        coeff,
        eps=1e-5,
        momentum=0.01,
        affine=True,
        track_running_stats=True,
    ):  # momentum is 0.01 by default instead of 0.1 of BN which alleviates noisy power iteration
        # Code is based on torch.nn.modules._NormBase
        super(SpectralBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats=track_running_stats
        )
        self.coeff = coeff

    @staticmethod
    def input_statistics(input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if len(input.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = input.mean(dim=0)
            var = ((input - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the mean and variance on the channel
            # dimension (axis=1). Here we need to maintain the shape of X, so that the broadcasting operation
            # can be carried out later
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            var = ((input - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        return mean, var

    @staticmethod
    def get_lipschitz_factor(weight, var, eps, coeff):
        # see https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct.
        lipschitz = torch.max(torch.abs(weight * (var + eps) ** -0.5))

        # if lipschitz of the operation is greater than coeff, then we want to divide the input by a constant to
        # force the overall lipchitz factor of the batch norm to be exactly coeff
        lipschitz_factor = torch.max(lipschitz / coeff, torch.ones_like(lipschitz))
        return lipschitz_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # before the foward pass, estimate the lipschitz constant of the layer and
        # divide through by it so that the lipschitz constant of the batch norm operator is approximately
        # 1
        weight = (
            torch.ones_like(self.running_var) if self.weight is None else self.weight
        )

        if self.track_running_stats:
            lipschitz_factor = self.get_lipschitz_factor(
                weight, self.running_var, self.eps, self.coeff
            )
        else:
            assert len(input.shape) in (2, 4)
            mean, var = self.input_statistics(input)
            lipschitz_factor = self.get_lipschitz_factor(
                weight, var, self.eps, self.coeff
            )
            out = (input - mean) / torch.sqrt(var + self.eps)
            weight = weight / lipschitz_factor
            return out * weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        weight = weight / lipschitz_factor

        return torch.nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class SpectralBatchNorm1d(SpectralBatchNorm, torch.nn.BatchNorm1d):
    pass


class SpectralBatchNorm2d(SpectralBatchNorm, torch.nn.BatchNorm2d):
    pass


class SpectralBatchNorm3d(SpectralBatchNorm, torch.nn.BatchNorm3d):
    pass


# --------------------------------------------------------------------------- #
# Spectral Normalization to regularize the DNN bi-lipschitz
# --------------------------------------------------------------------------- #


# This class implements a simple extension of
# https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#spectral_norm
# to include spectral norms, that are not exactly 1.
# also compare https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/normalization.py
class _SpectralNorm(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        norm_multiplier: float = 0.95,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )

        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        self.norm_multiplier = norm_multiplier
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer(
                "_u", torch.nn.functional.normalize(u, dim=0, eps=self.eps)
            )
            self.register_buffer(
                "_v", torch.nn.functional.normalize(v, dim=0, eps=self.eps)
            )

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(
                self.dim, *(d for d in range(weight.dim()) if d != self.dim)
            )

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # Precondition
        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = torch.nn.functional.normalize(
                torch.mv(weight_mat, self._v),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._u,
            )  # type: ignore[has-type]
            self._v = torch.nn.functional.normalize(
                torch.mv(weight_mat.t(), self._u), dim=0, eps=self.eps, out=self._v
            )  # type: ignore[has-type]

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return self.norm_multiplier * torch.nn.functional.normalize(
                weight, dim=0, eps=self.eps
            )
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            # similar to https://github.com/jhjacobsen/invertible-resnet/blob/master/spectral_norm_conv_inplace.py
            # only invoke as constraint!
            factorReverse = torch.max(
                torch.ones(1, device=weight.device), sigma / self.norm_multiplier
            )
            return weight / (factorReverse + 1e-5)  # add for stability

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


def spectral_norm(
    module: torch.nn.Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
    norm_multiplier: float = 0.95,
) -> torch.nn.Module:
    # extensive documentation is removed here. Compare
    # https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html

    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    torch.nn.utils.parametrize.register_parametrization(
        module,
        name,
        _SpectralNorm(weight, n_power_iterations, dim, eps, norm_multiplier),
    )
    return module


class RandomFeatureProjection(torch.nn.Module):
    def __init__(self, in_dim, high_dim, out_dim):
        super().__init__()
        self.register_buffer(
            "random_reduce",
            torch.normal(0, 0.05, (out_dim, in_dim)),
        )
        self.reduce = lambda x: torch.nn.functional.linear(x, self.random_reduce)

        self.normalize = torch.nn.LayerNorm(out_dim)

        self.register_buffer(
            "random_extend",
            torch.normal(0, 0.05, (high_dim, out_dim)),
        )
        self.extend = lambda x: torch.nn.functional.linear(x, self.random_extend)

        self.linear = torch.nn.Linear(high_dim, out_dim)

    def forward(self, x):
        x = self.reduce(x)
        x = self.normalize(x)
        x = torch.tanh(self.extend(x))
        x = self.linear(x)
        return x


# --------------------------------------------------------------------------- #
# SNGP implementation stolen from
# https://github.com/y0ast/DUE/blob/main/due/sngp.py
# --------------------------------------------------------------------------- #


def random_ortho(n, m):
    q, _ = torch.linalg.qr(torch.randn(n, m))
    return q


class RandomFourierFeatures(torch.nn.Module):
    def __init__(self, in_dim, num_random_features, feature_scale=None):
        super().__init__()
        if feature_scale is None:
            feature_scale = math.sqrt(num_random_features / 2)

        self.register_buffer("feature_scale", torch.tensor(feature_scale))

        if num_random_features <= in_dim:
            W = random_ortho(in_dim, num_random_features)
        else:
            # generate blocks of orthonormal rows which are not neccesarily orthonormal
            # to each other.
            dim_left = num_random_features
            ws = []
            while dim_left > in_dim:
                ws.append(random_ortho(in_dim, in_dim))
                dim_left -= in_dim
            ws.append(random_ortho(in_dim, dim_left))
            W = torch.cat(ws, 1)

        # From: https://github.com/google/edward2/blob/d672c93b179bfcc99dd52228492c53d38cf074ba/edward2/tensorflow/initializers.py#L807-L817
        feature_norm = torch.randn(W.shape) ** 2
        W = W * feature_norm.sum(0).sqrt()
        self.register_buffer("W", W)

        b = torch.empty(num_random_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x):
        k = torch.cos(x @ self.W + self.b)
        k = k / self.feature_scale

        return k


class Laplace(torch.nn.Module):
    #  feature_extractor,
    def __init__(
        self,
        num_deep_features,
        num_gp_features,
        normalize_gp_features,
        num_random_features,
        num_outputs,
        num_data,
        train_batch_size,
        ridge_penalty=1.0,
        feature_scale=None,
        mean_field_factor=None,  # required for classification problems
    ):
        super().__init__()
        # self.feature_extractor = feature_extractor
        self.mean_field_factor = mean_field_factor
        self.ridge_penalty = ridge_penalty
        self.train_batch_size = train_batch_size

        if num_gp_features > 0:
            self.num_gp_features = num_gp_features
            self.register_buffer(
                "random_matrix",
                torch.normal(0, 0.05, (num_gp_features, num_deep_features)),
            )
            self.jl = lambda x: torch.nn.functional.linear(x, self.random_matrix)
        else:
            self.num_gp_features = num_deep_features
            self.jl = torch.nn.Identity()

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize = torch.nn.LayerNorm(num_gp_features)

        self.rff = RandomFourierFeatures(
            num_gp_features, num_random_features, feature_scale
        )
        self.beta = torch.nn.Linear(num_random_features, num_outputs)

        self.num_data = num_data
        self.register_buffer("seen_data", torch.tensor(0))

        precision = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)

        self.recompute_covariance = True
        self.register_buffer("covariance", torch.eye(num_random_features))

    def reset_precision_matrix(self):
        identity = torch.eye(self.precision.shape[0], device=self.precision.device)
        self.precision = identity * self.ridge_penalty
        self.seen_data = torch.tensor(0)
        self.recompute_covariance = True

    def mean_field_logits(self, logits, pred_cov):
        # Mean-Field approximation as alternative to MC integration of Gaussian-Softmax
        # Based on: https://arxiv.org/abs/2006.07584

        logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
        if self.mean_field_factor > 0:
            logits = logits / logits_scale.unsqueeze(-1)

        return logits

    def forward(self, x):
        f_reduc = self.jl(x)
        # f = self.feature_extractor(x)
        # f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)

        k = self.rff(f_reduc)

        pred = self.beta(k)

        if self.training:
            precision_minibatch = k.t() @ k
            self.precision += precision_minibatch
            self.seen_data += x.shape[0]

            assert self.seen_data <= self.num_data, (
                "Did not reset precision matrix at start of epoch"
            )
        else:
            assert self.seen_data > (self.num_data - self.train_batch_size), (
                "Not seen sufficient data for precision matrix"
            )

            if self.recompute_covariance:
                with torch.no_grad():
                    eps = 1e-7
                    jitter = eps * torch.eye(
                        self.precision.shape[1],
                        device=self.precision.device,
                    )
                    u, info = torch.linalg.cholesky_ex(self.precision + jitter)
                    assert (info == 0).all(), "Precision matrix inversion failed!"
                    torch.cholesky_inverse(u, out=self.covariance)

                self.recompute_covariance = False

            with torch.no_grad():
                pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

            if self.mean_field_factor is None:
                return pred, pred_cov
            else:
                pred = self.mean_field_logits(pred, pred_cov)

        return pred
