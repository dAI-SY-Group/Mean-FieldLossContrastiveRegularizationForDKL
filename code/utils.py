# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24, 2026

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import typing
import gpytorch
import inducing_point_gps as IPGPs
import matplotlib.pyplot as plt
import models as ResNetModels
import numpy as np
import torch
import torch.optim as PytorchOptimizers
import torchmetrics as tm
import torchvision.models as models
from data_modules import get_datamodule_by_name
from layer import Laplace
from torch import nn, optim
from torch.nn.functional import softmax


def get_model(model_args, return_feature_size=False):
    model_name = model_args["name"]
    model_args = model_args["args"]
    if model_name.split("_")[0] == "torchvision":
        model_name = model_name[12:]
        model = getattr(models, model_name)
        if "pretrained" in model_args:
            model = model(pretrained=model_args["pretrained"])
            model.fc = torch.nn.Linear(model.fc.in_features, model_args["num_classes"])
        else:
            model = model(**model_args)
        model.aux_logits = False
        model.AuxLogits = None
        return model
    if model_name.__contains__("ResNet"):
        model = getattr(ResNetModels, model_name)
    if return_feature_size:
        init_with_classifier = False
        if not model_args["with_classifier"]:
            init_with_classifier = True
            model_args["with_classifier"] = True
        feature_extractor = model(**model_args)
        if hasattr(feature_extractor, "fc"):
            feature_size = feature_extractor.fc.in_features
        elif hasattr(feature_extractor, "classifier"):
            feature_size = feature_extractor.classifier.in_features
        else:
            raise NotImplementedError(
                "Model classifier is not named fc or classifier! Cannot extract feature size"
            )
        if init_with_classifier:
            model_args["with_classifier"] = False
            feature_extractor = model(**model_args)
        return feature_extractor, feature_size

    return model(**model_args)


def get_pytorch_optimizer_from_name(name, parameter, args):
    optimizer_class = getattr(PytorchOptimizers, name)
    return optimizer_class(parameter, **args)


def get_lr_scheduler(scheduler_params, optimizer):
    """Returns Scheduler from optimizer.schedulers and bool: requires batch_idx?"""
    if "warm_up_epochs" in scheduler_params and "decay_ratio" in scheduler_params:
        return warm_up_piecewise_constant_scheduler(scheduler_params, optimizer), True
    elif "milestones" in scheduler_params:
        return multi_step_scheduler(scheduler_params, optimizer), False
    else:
        raise NotImplementedError


class WarmUpPiecewiseConstant:
    """
    This Scheduler works similar to the one used in uncertainty_baselines deterministic.
    """

    def __init__(
        self, optimizer, steps_per_epoch, decay_ratio, decay_epochs, warm_up_epochs
    ):
        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.base_learning_rates = [pg["lr"] for pg in optimizer.param_groups]
        self.steps_per_epoch = steps_per_epoch
        self.decay_ratio = decay_ratio
        self.decay_epochs = decay_epochs
        self.warm_up_epochs = warm_up_epochs
        self.last_lr = self.base_learning_rates

    def step(self, epoch, batch_idx):
        # warm up
        self.last_lr = []
        for ind, lr in enumerate(self.base_learning_rates):
            if self.warm_up_epochs >= 1:
                lr *= (
                    batch_idx
                    / self.steps_per_epoch
                    / self.warm_up_epochs
                    / self.warm_up_epochs
                )
            decay_epochs = [self.warm_up_epochs] + self.decay_epochs
            for i, de in enumerate(decay_epochs):
                lr = np.where(
                    epoch >= de, self.base_learning_rates[ind] * self.decay_ratio**i, lr
                )
            lr = float(lr)
            self.last_lr.append(lr)
            self._set_lr(lr, ind)

    def get_last_lr(self):
        return self.last_lr

    def _set_lr(self, lr, n_param_group):
        self.optimizer.param_groups[n_param_group]["lr"] = lr

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def warm_up_piecewise_constant_scheduler(scheduler_params, optimizer):
    """Compare uncertainty_baselines (google)
    https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/schedules.py
    scheduler_params = {'steps_per_epoch': len(dataloader),
                        'base_learning_rate': 0.1,
                        'decay_ratio': 0.2,
                        'decay_epochs': [60, 120, 160] * epochs // 200
                        'warm_up_epochs': 1}
    """
    scheduler = WarmUpPiecewiseConstant(optimizer, **scheduler_params)
    return scheduler


def multi_step_scheduler(scheduler_params, optimizer):
    milestones = scheduler_params["milestones"]
    decay_ratio = scheduler_params["decay_ratio"]
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=decay_ratio
    )


def get_calibration_plot_from_ece_metric(ece, path=""):
    confidences = torch.cat(ece.confidences, dim=0)
    accuracies = torch.cat(ece.accuracies)
    bin_bounds = torch.linspace(
        0, 1, ece.n_bins + 1, dtype=confidences.dtype, device=confidences.device
    )

    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = torch.zeros(
        len(bin_bounds), device=confidences.device, dtype=confidences.dtype
    )
    conf_bin = torch.zeros(
        len(bin_bounds), device=confidences.device, dtype=confidences.dtype
    )
    count_bin = torch.zeros(
        len(bin_bounds), device=confidences.device, dtype=confidences.dtype
    )

    indices = torch.bucketize(confidences, bin_bounds, right=True) - 1

    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = torch.nan_to_num(conf_bin / count_bin)

    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)

    prop_bin = count_bin / count_bin.sum()

    # convert to numpy
    bin_bounds = bin_bounds.cpu().numpy()[:-1]
    conf_bin = conf_bin.cpu().numpy()[1:]
    acc_bin = acc_bin.cpu().numpy()[1:]
    count_bin = count_bin.cpu().numpy()[1:]
    bin_width = 1 / len(bin_bounds)

    f, (ax0, ax) = plt.subplots(
        2,
        1,
        figsize=(3, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "width_ratios": [1]},
    )
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    ax0.bar(
        bin_bounds,
        count_bin,
        color="blue",
        width=bin_width,
        edgecolor="black",
        align="edge",
    )
    ax0.set_ylabel("# values per bin")
    ax0.grid(True, axis="y", linestyle="--")
    # ax0.set_yticks([np.round(e, -2) for e in np.linspace(200, max(count_bin), 5)])
    plt.setp(ax0.get_yticklabels()[0], visible=False)

    ax.bar(
        bin_bounds,
        acc_bin,
        color="blue",
        width=bin_width,
        edgecolor="black",
        align="edge",
    )
    # ax.bar(bin_bounds, np.array(bin_boundaries[:-1]) - accs, bottom=accs, color='red', hatch="/",
    #        alpha=0.5, width=bin_width, edgecolor='black', align='edge')
    ax.plot(bin_bounds, bin_bounds)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_yticks(np.linspace(0, 1.0, len(bin_bounds)))
    plt.xticks(np.linspace(0, 1, len(bin_bounds)), rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--")

    if path:
        f.savefig(path + ".pdf", bbox_inches="tight")
        np.savez(
            path,
            binbounds=bin_bounds,
            confbin=conf_bin,
            accbin=acc_bin,
            countbin=count_bin,
        )

    return f


def entropy(probs: torch.Tensor):
    probs += 1e-8
    ent = -probs * torch.log(probs)
    ent = torch.sum(ent, -1)
    return ent


class MeanEntropy(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("entropy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("epsilon", default=torch.tensor(1e-8))

    def update(self, yhat, y=None):
        probs = yhat + self.epsilon
        self.entropy += (-probs * torch.log(probs)).sum(-1).mean()
        self.n += 1

    def compute(self):
        return self.entropy.float() / self.n


class AdaptiveECE(tm.Metric):
    def __init__(self, n_bins=15):
        super().__init__()
        self.add_state("confidences", default=[], dist_reduce_fx="cat")
        self.add_state("accuracies", default=[], dist_reduce_fx="cat")
        self.n_bins = n_bins

    def update(self, yhat, y):
        confidences, predictions = torch.max(yhat, 1)
        accuracies = predictions.eq(y)
        self.confidences.append(confidences.clone())
        self.accuracies.append(accuracies.clone())

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(
            np.linspace(0, npt, self.n_bins + 1), np.arange(npt), np.sort(x)
        )

    def compute(self):
        confidences = torch.cat(self.confidences, dim=0).cpu()
        accuracies = torch.cat(self.accuracies, dim=0).cpu()
        n, bin_boundaries = np.histogram(
            confidences.detach(), self.histedges_equalN(confidences.detach())
        )
        # print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=torch.device("cpu"))
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ClasswiseECE(tm.Metric):
    def __init__(self, num_classes=10, n_bins=15):
        super().__init__()
        self.num_classes = num_classes
        self.n_bins = 15
        self.add_state("confidences", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def update(self, yhat, y):
        self.confidences.append(yhat.clone())
        self.labels.append(y.clone())

    def compute(self):
        confidences = torch.cat(self.confidences, dim=0).cpu()
        labels = torch.cat(self.labels, dim=0).cpu()

        for i in range(self.num_classes):
            class_confidences = confidences[:, i]
            class_sce = torch.zeros(1, device=torch.device("cpu"))
            labels_in_class = labels.eq(
                i
            )  # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(
                    bin_upper.item()
                )
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += (
                        torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    )

            if i == 0:
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce


def get_inducing_points(
    method: str,
    dataset: str,
    n_inducing_points: int,
    feature_extractor: torch.nn.Module,
    use_n_points: int,
    device: str = "cpu",
    n_inducing_point_sets: int = 1,
):
    datamodule, n_classes = get_datamodule_by_name(dataset)
    if method == "cluster_minibatch":
        initial_ls, IPs = IPGPs.cluster_minibatch_inducing_points(
            datamodule,
            feature_extractor,
            n_inducing_points,
            use_n_points,
            device,
            n_inducing_point_sets,
        )
        return IPs, initial_ls
    else:
        raise NotImplementedError


def get_gp(
    gp_args: dict,
    dataset: str,
    feature_extractor: torch.nn.Module,
    device: str,
    n_classes: int = 1,
    feature_size=None,
):
    gp_name = gp_args["name"]
    del gp_args["name"]
    if gp_name == "Laplace":
        scaler = torch.nn.Identity()
        return Laplace(**gp_args), scaler
    elif gp_name == "GridVariational":
        grid_bounds = gp_args["grid_bounds"]
        grid_size = gp_args["grid_size"]
        gaussian_process = IPGPs.GpytorchGPLayerGridVariationalStrategy(
            num_dim=feature_size, grid_bounds=grid_bounds, grid_size=grid_size
        )
        tmp_scaler = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])

        def scaler(x):
            x = tmp_scaler(x)
            return x.transpose(-1, -2).unsqueeze(-1)

        return gaussian_process, scaler
    elif gp_name == "IPVariational":
        inducing_point_method = gp_args["inducing_point_method"]
        n_inducing_points = gp_args["n_inducing_points"]
        use_n_points = gp_args["use_n_points"]
        kernel = gp_args["kernel"]
        if dataset:
            inducing_points, initial_length_scale = get_inducing_points(
                inducing_point_method,
                dataset,
                n_inducing_points,
                feature_extractor,
                use_n_points,
                device,
            )
        else:
            input_size = gp_args["input_size"]
            # inducing_points = feature_extractor(torch.randn((n_inducing_points, input_size), device=device))
            inducing_points = torch.from_numpy(
                np.linspace(-10.0, 3.0, n_inducing_points).reshape(1, -1)
            ).float()
            initial_length_scale = torch.tensor(
                0.1
            )  # torch.pdist(inducing_points).mean()
        gp = IPGPs.GpytorchGPLayerVariationalStrategySM
        gaussian_process = gp(
            num_outputs=n_classes,
            initial_lengthscale=initial_length_scale,
            initial_inducing_points=inducing_points,
            kernel=kernel,
        )
        scaler = torch.nn.Identity()
        return gaussian_process, scaler
    elif gp_name == "AVDKL":
        return IPGPs.AmortizedSGP(**gp_args), torch.nn.Identity()
    else:
        raise NotImplementedError


@torch.jit.script
def prepare_targets(
    targets: torch.Tensor,
    dtype: torch.dtype = torch.float,
    alpha_eps: float = 0.01,
    n_classes: int = 10,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    if targets.ndim == 1:
        # set alpha = \alpha_\epsilon
        alpha = alpha_eps * torch.ones(
            targets.shape[-1], n_classes, device=targets.device, dtype=dtype
        )
        n = (targets < 0.0).sum()
        if n > 0:
            # alpha[class_labels] = 1 + \alpha_\epsilon
            alpha[torch.arange(len(targets))[:-n], targets[:-n]] = (
                alpha[torch.arange(len(targets))[:-n], targets[:-n]] + 1.0
            )
        else:
            alpha[torch.arange(len(targets)), targets] = (
                alpha[torch.arange(len(targets)), targets] + 1.0
            )
    else:
        alpha = targets.float()
        alpha[alpha > 0] = alpha[alpha > 0] + 1.0
        alpha += alpha_eps
    # sigma^2 = log(1 / alpha + 1)
    sigma2_i = torch.log(alpha.reciprocal() + 1.0)
    # y = log(alpha) - 0.5 * sigma^2
    transformed_targets = alpha.log() - 0.5 * sigma2_i
    return sigma2_i.transpose(-2, -1).type(dtype), transformed_targets.transpose(
        -2, -1
    ).type(dtype)


def get_loss(
    loss_args,
    gaussian_process=None,
    n_train_samples=1,
    n_features=1,
    n_classes=1,
    with_noise=False,
):
    loss_name = loss_args["name"]
    del loss_args["name"]
    if loss_name in ["CCE", "cce", "cross_entropy"]:
        if "smoothing" in loss_args:
            return torch.nn.CrossEntropyLoss(
                label_smoothing=loss_args["smoothing"]
            ), None
        else:
            return torch.nn.CrossEntropyLoss(), None
    if gaussian_process is not None:
        if loss_name == "Softmax":
            mixing_weights = loss_args["mixing_weights"]
            likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
                num_features=n_features,
                num_classes=n_classes,
                mixing_weights=mixing_weights,
            )
            mll = gpytorch.mlls.VariationalELBO(
                likelihood, gaussian_process, num_data=n_train_samples
            )
            if with_noise:
                return lambda yhat, y, n: -mll(yhat, y, noise=n), likelihood
            else:
                return lambda yhat, y: -mll(yhat, y), likelihood
        elif loss_name == "Dirichlet":
            alpha_eps = loss_args["alpha_eps"]
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            mll = gpytorch.mlls.VariationalELBO(
                likelihood, gaussian_process, num_data=n_train_samples
            )

            def loss(yhat, y):
                sigma2, y_trans = prepare_targets(y, alpha_epsilon=alpha_eps)
                return -mll(yhat, y_trans, noise=sigma2)

            return loss, likelihood
        else:
            raise NotImplementedError


def compute_temperature_scale(model_forward, valid_loader):
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = nn.CrossEntropyLoss().cuda()

    def temperature_scale(logits, temp):
        temp = temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temp.to(logits.device)

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.cuda()
            logits = model_forward(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    print("Final temperature:", temperature.item())

    return lambda x: temperature_scale(x, temperature)


def compute_temperature_scale_mc(model_forward, valid_loader, mc_steps):
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = nn.NLLLoss().cuda()

    def temperature_scale(mean, cov, temp):
        temp = temp.unsqueeze(1).expand(mean.size(0), mean.size(1))
        tmp = mean + torch.randn(
            mc_steps, mean.size(0), mean.size(1), device=mean.device
        ) * cov.unsqueeze(0)
        fn = lambda x: x / temp.to(x.device)
        tmp = torch.vmap(fn)(tmp)
        return softmax(tmp, dim=-1).mean(0).log()  # might be unstable!

    # First: collect all the logits and labels for the validation set
    mean_list = []
    cov_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.cuda()
            mean, cov = model_forward(input)
            mean_list.append(mean)
            cov_list.append(cov)
            labels_list.append(label)
        mean = torch.cat(mean_list).cuda()
        cov = torch.cat(cov_list).cuda()
        labels = torch.cat(labels_list).cuda()

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.003, max_iter=100)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(mean, cov, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    print("Final temperature:", temperature.item())

    def temperature_scale(logits, temp):
        temp = temp.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temp.to(logits.device)

    return lambda x: temperature_scale(x, temperature)


class GenSupConLoss(torch.nn.Module):
    """
    (Modified) from https://github.com/kiimmm/GenSCL/blob/main/loss.py
    """

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        distance_type="mm",
    ):
        super(GenSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.distance_type = distance_type
        assert self.distance_type in ["mm", "l2"], "Only mm and l2 currently supported"

    def forward(self, features, labels):
        """
        Args:
            feats: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each either [N, num_cls] or N
        """
        if self.contrast_mode == "all":  # anchor+contrast @ anchor+contrast
            anchor_labels = torch.cat(labels, dim=0).float()
            contrast_labels = anchor_labels

            anchor_features = torch.cat(features, dim=0)
            contrast_features = anchor_features
        elif self.contrast_mode == "one":  # anchor @ contrast
            anchor_labels = labels[0].float()
            contrast_labels = labels[1].float()

            anchor_features = features[0]
            contrast_features = features[1]

        # 1. compute similarities among targets
        anchor_norm = torch.norm(
            anchor_labels, p=2, dim=-1, keepdim=True
        )  # [anchor_N, 1]
        contrast_norm = torch.norm(
            contrast_labels, p=2, dim=-1, keepdim=True
        )  # [contrast_N, 1]

        deno = torch.mm(anchor_norm, contrast_norm.T)
        mask = (
            torch.mm(anchor_labels, contrast_labels.T) / deno
        )  # cosine similarity: [anchor_N, contrast_N]

        logits_mask = torch.ones_like(mask)
        if self.contrast_mode == "all":
            logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # 2. compute logits
        if self.distance_type == "mm":
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_features, contrast_features.T), self.temperature
            )
        elif self.distance_type == "l2":
            anchor_dot_contrast = -torch.div(
                torch.cdist(anchor_features, contrast_features), self.temperature
            )
        else:
            raise NotImplementedError

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def get_defaults(name, dataset, model):
    assert model == "WideResNet_28_10", "Defaults available for WideResNet_28_10 only!"
    if dataset == "Cifar10" or dataset == "Cifar100":
        model_params = {
            "name": "WideResNet_28_10",
            "args": {
                "num_classes": 10 if dataset == "Cifar10" else 100,
                "shortcut_with_bn": False,
                "track_running_stats": True,
                "with_classifier": False,
                "preact": True,
                "sp_norm_p": [3.0, 3.0],
            },
        }
        if name == "SVDKL":
            # model_params['args']['with_classifier'] = True
            # model_params['args']['num_classes'] = 10
            # model_params = {'name': 'DenseNet3x6',
            #                 'args': {'with_classifier': False}}
            optim_params = {
                "name": "SGD",
                "num_likelihood_samples": (16, 320),  # default=10
                "args": {
                    "lr": 0.1,
                    "nesterov": True,
                    "momentum": 0.9,
                    "feature_extractor": {"weight_decay": 5e-4},
                    "gaussian_process": {"hyperparameter": {"lr": 0.0001}},
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = {
                "name": "GridVariational",
                "grid_bounds": (-10.0, 10.0),
                "grid_size": 64,
            }
            loss_params = {"name": "Softmax", "mixing_weights": True}
        elif name == "DUE":
            optim_params = {
                "name": "SGD",
                "num_likelihood_samples": (16, 320),  # default=10
                "args": {
                    "lr": 0.1,
                    "nesterov": False,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = {
                "name": "IPVariational",
                "inducing_point_method": "cluster_minibatch",
                "n_inducing_points": 10 if dataset == "Cifar10" else 200,
                "use_n_points": 2000,
                "kernel": "RBF",
            }
            loss_params = {"name": "Softmax", "mixing_weights": None}
        elif name == "SVGP":
            optim_params = {
                "name": "SGD",
                "args": {
                    "lr": 0.05,
                    "nesterov": False,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = {
                "name": "Laplace",
                "num_deep_features": 640,
                "num_gp_features": 128,
                "normalize_gp_features": True,
                "num_random_features": 1024,
                "num_outputs": 10 if dataset == "Cifar10" else 100,
                "train_batch_size": 128,
                "num_data": 47500,
                "mean_field_factor": 25,
                "ridge_penalty": 1,
                "feature_scale": 2,
            }
            loss_params = {"name": "CCE"}
            print("!!CHECK Batch Size!! - default assumes 128.")
        elif name == "DNN":
            model_params["args"]["with_classifier"] = True
            # model_params['args']['sp_norm_p'] = (0., 0.)
            optim_params = {
                "name": "SGD",
                "args": {
                    "lr": 0.1,
                    "nesterov": True,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = None
            loss_params = {"name": "CCE"}
        elif name == "AVDKL":
            optim_params = {
                "name": "SGD",
                "num_likelihood_samples": (16, 320),  # default=10
                "args": {
                    "lr": 0.1,
                    "nesterov": False,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = dict(
                name="AVDKL",
                num_outputs=640,
                bottleneck=8,
                n_classes=10 if dataset == "Cifar10" else 100,
                num_inducing=15 if dataset == "Cifar10" else 200,
                kernel="RBF",
            )
            loss_params = {"name": "Softmax", "mixing_weights": None}
        elif name == "PosteriorNetwork":
            model_params["args"]["with_classifier"] = True
            optim_params = {
                "name": "SGD",
                "args": {
                    "lr": 0.1,
                    "nesterov": True,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = dict(
                one_dist_estimator=False,
                type="iaf_flow",
                args={
                    "features": 10 if dataset == "Cifar10" else 100,
                    "flow_length": 8 if dataset == "Cifar10" else 4,
                },
            )
            gp_params = None
            loss_params = {"name": "uce", "regr": 1.0e-5}
        elif name.__contains__("DNNNoiseEst"):
            # model_params = {'name': 'DenseNet3x6',
            #                 'args': {'with_classifier': False}}
            if not model_params["name"].__contains__("DenseNet"):
                model_params["args"]["sp_norm_p"] = (0.0, 0.0)
            model_params["mlp_model"] = False
            model_params["n_train_samples"] = 47500
            model_params["dataset"] = dataset
            model_params["track_features_of_n_samples"] = 100
            optim_params = {
                "name": "SGD",
                "args": {
                    "lr": 0.05,
                    "nesterov": True,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },  # 'uncertainty_mlp': {'lr': 0.005}
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = None
            loss_params = {"name": "CCE"}
            if name == "DNNNoiseEst2":
                model_params["mlp_model"] = True
                optim_params["args"]["lr"] = 0.05
                optim_params["args"]["uncertainty_mlp"] = {"lr": 0.05}
        elif name == "DUENoiseEst":
            model_params["track_features_of_n_samples"] = 100
            optim_params = {
                "name": "SGD",
                "args": {
                    "lr": 0.05,
                    "nesterov": False,
                    "momentum": 0.9,
                    "weight_decay": 5e-4,
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = {
                "name": "IPVariational",
                "inducing_point_method": "cluster_minibatch",
                "n_inducing_points": 10 if dataset == "Cifar10" else 100,
                "use_n_points": 1000,
                "kernel": "RBF",
                "mlp_model": True,
            }
            loss_params = {"name": "Softmax", "mixing_weights": None}
        elif name == "SVDKLNoiseEst":
            # model_params = {'name': 'DenseNet3x6',
            #                 'args': {'with_classifier': False}}
            model_params["track_features_of_n_samples"] = 100
            optim_params = {
                "name": "SGD",
                "num_likelihood_samples": (8, 16),  # default=10
                "args": {
                    "lr": 0.1,
                    "nesterov": True,
                    "momentum": 0.9,
                    "feature_extractor": {"weight_decay": 5e-4},
                    "gaussian_process": {"hyperparameter": {"lr": 0.0001}},
                    "uncertainty_mlp": {"lr": 0.01},
                },
            }
            scheduler_params = {"decay_ratio": 0.2, "milestones": [60, 120, 160]}
            gp_params = {
                "name": "GridVariational",
                "grid_bounds": (-10.0, 10.0),
                "grid_size": 64,
                "mlp_model": True,
            }
            loss_params = {"name": "Softmax", "mixing_weights": True}
        else:
            raise NotImplementedError
        return model_params, gp_params, loss_params, optim_params, scheduler_params
    else:
        raise NotImplementedError
