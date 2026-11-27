# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24, 2026

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import copy
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from data_modules import get_datamodule_by_name
from inducing_point_gps import (
    IPModule,
    cluster_minibatch_inducing_points,
    get_forward_grid_ips,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import ArgsType, LightningCLI
from torch.distributions import Dirichlet, Independent
from torch.nn.functional import softmax
from torchvision.transforms import v2
from utils import (
    AdaptiveECE,
    ClasswiseECE,
    GenSupConLoss,
    MeanEntropy,
    compute_temperature_scale,
    compute_temperature_scale_mc,
    entropy,
    get_calibration_plot_from_ece_metric,
    get_lr_scheduler,
    get_model,
    get_pytorch_optimizer_from_name,
)

USE_COMPILATION = False


@torch.no_grad()
def get_ood_scores(
    ood_loader,
    model,
    device,
    mc_prediction_steps,
    save_dir="",
    temperature_scale=False,
    ts_fn=None,
    name="train",
):
    model.to(device)
    model.eval()
    mean_scores = []
    cov_scores = []
    for x, y in ood_loader:
        x = x.to(device)
        if model.use_contrastive:
            z, features = model(x)
        else:
            features = model(x)
        mean, cov = model.gp_classifier(features)
        if mc_prediction_steps > 1:
            # print(mean.shape, cov.shape, mc_prediction_steps, x.shape, model.n_classes)
            tmp = mean + torch.randn(
                mc_prediction_steps, mean.size(0), model.n_classes, device=x.device
            ) * cov.reshape(1, mean.size(0), -1)
            if temperature_scale:
                tmp = torch.vmap(ts_fn)(tmp)
            mc_probs = softmax(tmp, dim=-1)
            probs = mc_probs.mean(0)
            cov = mc_probs.std(0)
        else:
            if temperature_scale:
                mean = ts_fn(mean)
            probs = softmax(mean, dim=-1)
        mean_scores.append(probs)
        cov_scores.append(cov)
    mean_scores = torch.cat(mean_scores)
    cov_scores = torch.cat(cov_scores)
    entropy_scores = entropy(mean_scores)
    if save_dir:
        np.savez(
            save_dir + f"/ood_scores_{name}_{int(temperature_scale)}.npz",
            entropy_scores=entropy_scores.clone().detach().cpu().numpy(),
            cov_scores=cov_scores.clone().detach().cpu().numpy(),
        )
    return entropy_scores.mean(), cov_scores.mean()


class NetworkWrapper(torch.nn.Module):
    """ """

    def __init__(
        self,
        feature_extractor,
        fe_args: dict = {},
        gp_args: dict = {},
        use_contrastive: bool = False,
        head: str = "mlp",
        head_dim: int = 128,
        initialization_ips="clustering",
        load_weight_file: str = "",
        n_additional_inducing_points: int = 0,
        init_forward_data=None,
    ):
        super().__init__()
        if isinstance(feature_extractor, str):
            self.feature_extractor = get_model(
                {"name": feature_extractor, "args": fe_args}
            )
        else:
            self.feature_extractor = feature_extractor

        if load_weight_file:
            self.feature_extractor.load_state_dict(torch.load(load_weight_file))

        if hasattr(self.feature_extractor, "classifier"):
            self.classifier = copy.deepcopy(self.feature_extractor.classifier)
            self.feature_extractor.classifier = torch.nn.Identity()
        elif hasattr(self.feature_extractor, "fc"):
            self.classifier = copy.deepcopy(self.feature_extractor.fc)
            self.feature_extractor.fc = torch.nn.Identity()
        else:
            raise ValueError("Feature Extractor requires classifier or fc attribute!")

        self.n_classes = fe_args["num_classes"]
        self.n_additional_inducing_points = int(n_additional_inducing_points)

        if initialization_ips == "clustering":
            cluster_ips = cluster_minibatch_inducing_points(
                DataModule,
                self.feature_extractor,
                self.n_classes + n_additional_inducing_points,
                10000,
                "cuda" if torch.cuda.is_available() else "cpu",
            )[1]
        elif initialization_ips == "random_linear":
            assert n_additional_inducing_points < 1, (
                "random_linear does not support additional induction points at the moment"
            )
            cluster_ips = torch.nn.Linear(
                self.classifier.in_features, self.n_classes
            ).weight
        elif initialization_ips == "random_gaussian":
            cluster_ips = torch.randn(
                self.classifier.in_features,
                self.n_classes + n_additional_inducing_points,
            )  # * 0.01
        elif initialization_ips == "forward_grid":
            if init_forward_data is None:
                cluster_ips = get_forward_grid_ips(
                    DataModule,
                    self.feature_extractor,
                    self.n_classes,
                    n_additional_inducing_points,
                    10,
                    "cuda" if torch.cuda.is_available() else "cpu",
                ).T
            else:
                cluster_ips = self.feature_extractor(init_forward_data).T
        elif initialization_ips == "zeros":
            cluster_ips = torch.zeros(
                self.classifier.in_features,
                self.n_classes + n_additional_inducing_points,
            )
        elif initialization_ips == "pretrained":
            raise NotImplementedError("Not supported in version 2")
            cluster_ips = self.classifier.weight.unsqueeze(0)
            gp_args["pretrained_bias"] = self.classifier.bias.unsqueeze(-1).clone()
        else:
            raise NotImplementedError(
                "Initialization of IPs must be in [clustering, random_linear, pretrained]"
            )

        self.use_contrastive = use_contrastive
        if use_contrastive:
            n_features = self.classifier.in_features
            if head == "mlp":
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(n_features, n_features),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_features, head_dim),
                )
            elif head == "linear":
                self.head = torch.nn.Linear(n_features, head_dim)
            else:
                raise NotImplementedError

            # assert initialization_ips.__contains__('random'), \
            #     "Inducing points initialization only supported with random initialization"
            # self.gp_classifier = IPModule(head_dim, self.n_classes, cluster_ips[:head_dim].T, **gp_args)
            self.gp_classifier = IPModule(
                self.classifier.in_features, self.n_classes, cluster_ips.T, **gp_args
            )
        else:
            self.gp_classifier = IPModule(
                self.classifier.in_features, self.n_classes, cluster_ips.T, **gp_args
            )
            # if init_forward_data is not None:
            #     self.gp_classifier.init_manifold_regularization(init_forward_data, dims=min(tuple(init_forward_data.shape)))
            # else:
            #     self.gp_classifier.init_manifold_regularization(init_forward_data, dims=self.classifier.out_features,
            #                                                     classification=True)

    def get_params_for_optimizer(self, mode="all"):
        if mode == "all":
            params = [
                {"params": self.feature_extractor.parameters()},
                {"params": self.classifier.parameters()},
                {"params": self.gp_classifier.parameters(), "weight_decay": 0.0},
            ]
            if self.use_contrastive:
                params.append({"params": self.head.parameters()})
            return params
        elif mode == "pretrained_eval":
            return [{"params": self.gp_classifier.parameters()}]

    def classify_linear(self, x):
        return self.classifier(x)

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.use_contrastive:
            z = self.head(features)
            return z, features
        else:
            return features


def cla_constrastive_loss(f, y, bsz, mode="mm"):
    f = torch.nn.functional.normalize(f, dim=-1)
    temperature = 0.07
    base_temperature = 0.07

    # compute lam
    y1, y2 = torch.split(y, [bsz, bsz], dim=0)
    lam = torch.nn.functional.threshold(y2 - y1, 0.0, 0.0).sum(-1, keepdim=True)
    lam = 1 - lam

    f1, f2 = torch.split(f, [bsz, bsz], dim=0)
    f2_from_f1 = f1 * lam + f1.roll(1) * (1 - lam)

    if mode == "mm":
        anchor_dot_contrast = torch.matmul(f2, f2_from_f1.T) / temperature
    elif mode == "l2":
        anchor_dot_contrast = -torch.cdist(f2, f2_from_f1) / temperature
    else:
        raise NotImplementedError

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    mask = torch.eye(bsz, device=f.device)
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()
    return loss


class IPGPNetwork(pl.LightningModule):
    """ """

    def __init__(
        self,
        model_name: str = "WideResNet_28_10",
        model_args: dict = {},
        gp_args: dict = {},
        head: str = "mlp",
        head_dim: int = 128,
        optimizer: str = "Adam",
        optimizer_args: dict = {},
        scheduler_params: dict = {},
        loss_args: dict = {},
        train_through_gp: bool = False,
        contrastive: bool = True,
        contrastive_factor: float = 1.0,
        mc_prediction_steps: int = 1,
        pretrained_eval_mode: bool = False,
        initialization_ips: str = "clustering",
        load_weight_file: str = "",
        mixup_cutmix_alphas: list = [0.0, 0.0],
        float32_matmul_precision: str = "",
        n_additional_inducing_points=0,
        extra_samples: int = 0,
        contrastive_linearizing_loss=0.0,
    ):
        super().__init__()
        gp_args["alpha_eps"] = float(gp_args["alpha_eps"])

        if float32_matmul_precision:
            torch.set_float32_matmul_precision(float32_matmul_precision)

        self.pretrained_eval_mode = pretrained_eval_mode
        self.train_through_gp = train_through_gp
        self.contrastive = (
            contrastive
            if isinstance(contrastive, bool)
            else (True if contrastive == "true" else False)
        )
        loss_name = loss_args.pop("name")
        self.gensupcon = False
        if loss_name.lower().__contains__("supcon"):
            if loss_name.lower() == "gensupcon":
                self.cont_loss = GenSupConLoss(**loss_args)
                self.gensupcon = True
            assert contrastive, "Contrastive loss requires contrastive learning"
        self.loss_args = loss_args
        self.cce_loss = torch.nn.CrossEntropyLoss()
        self.n_additional_inducing_points = n_additional_inducing_points
        self.model = NetworkWrapper(
            model_name,
            model_args,
            gp_args,
            contrastive,
            head,
            head_dim,
            initialization_ips,
            load_weight_file,
            n_additional_inducing_points,
        )
        self.optimizer = optimizer
        self.opt_args = optimizer_args
        self.scheduler_params = scheduler_params
        self.scheduler_requires_batch_idx = False
        self.current_test_loader = 0
        self.n_test_loader = 5 * 5 + 1
        if dataset.lower().__contains__("imagenet"):
            self.n_test_loader += 2
        if dataset.lower().__contains__("tiny"):
            self.n_test_loader -= 5
        self.mc_prediction_steps = mc_prediction_steps
        self.temperature_scale = False
        self.temperature_scale_fn = None
        self.extra_samples = int(extra_samples)
        self.test_run = "best"
        self.contrastive_linearizing_loss = float(contrastive_linearizing_loss)
        self.contrastive_factor = float(contrastive_factor)

        n_classes = model_args["num_classes"]
        self.add_aug = any(mixup_cutmix_alphas)
        if self.add_aug:
            mixup_alpha, cutmix_alpha = mixup_cutmix_alphas
            if mixup_alpha and not cutmix_alpha:
                self.add_aug = v2.MixUp(alpha=mixup_alpha, num_classes=n_classes)
            if not mixup_alpha and cutmix_alpha:
                self.add_aug = v2.CutMix(alpha=cutmix_alpha, num_classes=n_classes)
            if cutmix_alpha and mixup_alpha:
                self.add_aug = v2.RandomChoice(
                    [
                        v2.MixUp(alpha=mixup_alpha, num_classes=n_classes),
                        v2.CutMix(alpha=cutmix_alpha, num_classes=n_classes),
                    ]
                )

        self.metrics = tm.MetricCollection(
            {
                "val/mcc": tm.MatthewsCorrCoef(
                    task="multiclass", num_classes=n_classes
                ),
                "val/accuracy": tm.Accuracy(task="multiclass", num_classes=n_classes),
                "val/f1-score": tm.F1Score(task="multiclass", num_classes=n_classes),
                "val/auroc": tm.AUROC(task="multiclass", num_classes=n_classes),
                "val/ece": tm.CalibrationError(
                    task="multiclass", num_classes=n_classes, norm="l1"
                ),
                "val/mce": tm.CalibrationError(
                    task="multiclass", num_classes=n_classes, norm="max"
                ),
                "val/adaptive_ece": AdaptiveECE(n_bins=15),
                "val/classwise_ece": ClasswiseECE(num_classes=n_classes, n_bins=15),
                "val/ment": MeanEntropy(),
            }
        )

        self.metrics2 = tm.MetricCollection(
            {
                k.replace("val", "val_gp"): v
                for k, v in copy.deepcopy(self.metrics).items()
            }
        )

        self.test_metrics = torch.nn.ModuleList(
            [
                tm.MetricCollection(
                    {
                        k.replace("val", "test"): v
                        for k, v in copy.deepcopy(self.metrics).items()
                    }
                )
                for _ in range(self.n_test_loader)
            ]
        )
        self.test_metrics2 = torch.nn.ModuleList(
            [
                tm.MetricCollection(
                    {
                        k.replace("val", "test_gp"): v
                        for k, v in copy.deepcopy(self.metrics).items()
                    }
                )
                for _ in range(self.n_test_loader)
            ]
        )

        self.automatic_optimization = False
        self.save_hyperparameters()

    def configure_optimizers(self) -> Any:
        params = self.model.get_params_for_optimizer()
        if self.pretrained_eval_mode:
            self.model.feature_extractor.eval()
        optimizer = get_pytorch_optimizer_from_name(
            self.optimizer, params, self.opt_args
        )
        if len(self.scheduler_params) > 0:
            scheduler, self.scheduler_requires_batch_idx = get_lr_scheduler(
                self.scheduler_params, optimizer
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        else:
            return [optimizer]

    def lr_scheduler_step(self, scheduler, batch_idx):
        scheduler.step(epoch=self.current_epoch, batch_idx=batch_idx)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        bsz = y.shape[0]
        opt = self.optimizers()
        opt.zero_grad()

        if self.add_aug:
            if self.gensupcon:
                x1, y1 = self.add_aug(x[0], y)
                x2, y2 = self.add_aug(x[1], y)
                x = [x1, x2]
                y = [y1, y2]
            if not self.contrastive:
                if self.contrastive_linearizing_loss > 0.0:
                    x_new, y_new = self.add_aug(x, y)
                    x = torch.cat([x, x_new])
                    y = torch.cat(
                        [
                            torch.nn.functional.one_hot(
                                y, num_classes=self.model.n_classes
                            ),
                            y_new,
                        ]
                    )
                else:
                    x, y = self.add_aug(x, y)
        else:
            if self.contrastive:
                y = [y, y]

        def closure():
            if self.contrastive:
                images = torch.cat([x[0], x[1]], dim=0)
                if self.pretrained_eval_mode:
                    with torch.no_grad():
                        z, features = self.model(images)
                else:
                    z, features = self.model(images)

                gp_loss = self.model.gp_classifier.get_loss(features, torch.cat(y))

                y_sc = [y]

                fs = torch.nn.functional.normalize(z, dim=1)
                f1, f2 = torch.split(fs, [bsz, bsz], dim=0)
                if self.gensupcon:
                    fs = [f1, f2]
                else:
                    fs = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    y_sc = [y[0] for y in y_sc]

                main_loss = torch.zeros(1, device=f1.device)
                for i in range(1, len(y_sc) + 1):
                    main_loss = main_loss + self.contrastive_factor * self.cont_loss(
                        fs, y_sc[i - 1]
                    )  # i**-1 *

                features = features.detach()
                class_loss = self.cce_loss(
                    self.model.classify_linear(features), torch.cat(y)
                )
                classifier_loss = class_loss + gp_loss
            else:
                if self.extra_samples > 0:
                    extra_samples = torch.randn(
                        self.extra_samples,
                        x.size(1),
                        x.size(2),
                        x.size(3),
                        device=x.device,
                    )
                    xhat = torch.cat([x, extra_samples], dim=0)
                    yhat = torch.cat(
                        [y, -torch.ones(self.extra_samples, device=y.device).long()],
                        dim=0,
                    )
                    if self.contrastive_linearizing_loss > 0.0:
                        raise NotImplementedError
                else:
                    xhat, yhat = x, y

                if self.pretrained_eval_mode:
                    with torch.no_grad():
                        features = self.model(xhat)
                else:
                    features = self.model(xhat)

                if self.train_through_gp:
                    if self.contrastive_linearizing_loss > 0.0:
                        cont_loss = cla_constrastive_loss(
                            features, yhat, bsz, mode="l2"
                        )
                    main_loss = self.model.gp_classifier.get_loss(features, yhat)
                    if self.contrastive_linearizing_loss > 0.0:
                        main_loss = (
                            1 - self.contrastive_linearizing_loss
                        ) * main_loss + self.contrastive_linearizing_loss * cont_loss
                else:
                    pred = self.model.gp_classifier.linear_with_ip(features)
                    main_loss = self.cce_loss(pred, yhat)

                features = features.detach()
                if self.extra_samples > 0:
                    classifier_loss = self.cce_loss(
                        self.model.classify_linear(features[: -self.extra_samples]), y
                    )
                else:
                    classifier_loss = self.cce_loss(
                        self.model.classify_linear(features), y
                    )

                if not self.train_through_gp:
                    gp_loss = self.model.gp_classifier.gp_loss(features, yhat)
                    classifier_loss = classifier_loss + gp_loss

            loss = main_loss + classifier_loss

            self.manual_backward(loss)

            return loss, main_loss, classifier_loss

        loss, main_loss, classifier_loss = closure()

        opt.step()
        on_step = False
        self.log(
            "loss/all",
            float(loss.item()),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "loss/main",
            main_loss.item(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "loss/classifier",
            classifier_loss.item(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        if self.model.gp_classifier.debug_mode:
            for k, v in self.model.gp_classifier.debug_values.items():
                self.log(
                    f"debug/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        if self.scheduler_requires_batch_idx:
            sch = self.lr_schedulers()
            sch.step(self.current_epoch, batch_idx)

    def on_train_epoch_end(self) -> None:
        if len(self.scheduler_params) > 0 and not self.scheduler_requires_batch_idx:
            sch = self.lr_schedulers()
            sch.step()

    def prediction_step(self, batch, m1, m2):
        x, y = batch
        # Why the hell was this commented out until 1085?
        if self.contrastive:
            z, features = self.model(x)
            # fs = torch.nn.functional.normalize(z, dim=1)
            # mean, cov = self.model.gp_classifier(fs)
        else:
            features = self.model(x)
        mean, cov = self.model.gp_classifier(features)

        if self.model.gp_classifier.loss_type.__contains__("dirichlet_pll"):
            alphas = torch.exp(mean - np.log(1 / self.model.n_classes)) / torch.sqrt(
                cov + 1e-7
            )
            marginal = Dirichlet(alphas)
            batched_marginal = Independent(marginal, 0)
            if self.mc_prediction_steps > 1:
                probs = batched_marginal.sample((self.mc_prediction_steps,)).mean(0)
            else:
                probs = alphas / alphas.sum(dim=-1, keepdim=True)
        else:
            if self.mc_prediction_steps > 1:
                tmp = mean + torch.randn(
                    self.mc_prediction_steps,
                    mean.size(0),
                    self.model.n_classes,
                    device=x.device,
                ) * cov.unsqueeze(0)
                if self.temperature_scale:
                    tmp = torch.vmap(self.temperature_scale_fn)(tmp)
                probs = softmax(tmp, dim=-1).mean(0)
            else:
                if self.temperature_scale:
                    mean = self.temperature_scale_fn(mean)
                probs = softmax(mean, dim=-1)

        logits = self.model.classify_linear(features)
        logits = softmax(logits, dim=1)

        m1.update(logits, y)
        m2.update(probs, y)

    # def on_validation_epoch_start(self) -> None:
    #     self.model.gp_classifier.costly_gp_setup()

    def validation_step(self, batch, batch_idx):
        self.prediction_step(batch, self.metrics, self.metrics2)

    def on_validation_epoch_end(self) -> None:
        metric_dict = self.metrics.compute()

        # loop over log to put some values in progress bar
        for k, v in metric_dict.items():
            include = k in ["val/f1-score"]
            self.log(
                k,
                v.item(),
                on_step=False,
                on_epoch=True,
                prog_bar=include,
                sync_dist=True,
            )
        self.log_dict(
            self.metrics2.compute(), on_step=False, on_epoch=True, sync_dist=True
        )

        self.metrics.reset()
        self.metrics2.reset()

    def on_train_end(self) -> None:
        self.model.eval()

        def model_forward(x):
            if self.contrastive:
                z, features = self.model(x)
            else:
                features = self.model(x)

            mean, cov = self.model.gp_classifier(features)
            if self.mc_prediction_steps > 1:
                return mean, cov
            else:
                return mean

        if self.mc_prediction_steps > 1:
            self.temperature_scale_fn = compute_temperature_scale_mc(
                model_forward,
                self.trainer.datamodule.val_dataloader(),
                self.mc_prediction_steps,
            )
        else:
            self.temperature_scale_fn = compute_temperature_scale(
                model_forward, self.trainer.datamodule.val_dataloader()
            )

    # def on_test_epoch_start(self) -> None:
    #     self.model.gp_classifier.costly_gp_setup()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.prediction_step(
            batch, self.test_metrics[dataloader_idx], self.test_metrics2[dataloader_idx]
        )

        self.log_dict(
            self.test_metrics[dataloader_idx],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            self.test_metrics2[dataloader_idx],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    @staticmethod
    def metrics_to_dict(metric):
        return {k: v.item() for k, v in metric.compute().items()}

    def on_test_end(self) -> None:
        name_list = list(self.test_metrics[0].keys()) + list(
            self.test_metrics2[0].keys()
        )
        metrics_summary = pd.DataFrame(
            data=np.zeros((self.n_test_loader, len(name_list))),
            columns=name_list,
            index=["test"] + self.trainer.datamodule.corrupted_testSets_names,
        )

        for j in range(self.n_test_loader):
            fd, sd = (
                self.metrics_to_dict(self.test_metrics[j]),
                self.metrics_to_dict(self.test_metrics2[j]),
            )
            print(fd, sd)

            metrics_summary.iloc[j] = pd.concat([pd.Series(fd), pd.Series(sd)])
            # print(metrics_summary)

            if not os.path.exists(self.logger.log_dir + f"/{self.test_run}"):
                os.makedirs(self.logger.log_dir + f"/{self.test_run}")

            ece = self.test_metrics[j]["test/ece"]
            name = f"{self.test_run}/calibration_plot_test_{j}_{int(self.temperature_scale)}"
            get_calibration_plot_from_ece_metric(ece, self.logger.log_dir + "/" + name)
            ece = self.test_metrics2[j]["test_gp/ece"]
            name = f"{self.test_run}/calibration_plot_test_{j}_probs_{int(self.temperature_scale)}"
            get_calibration_plot_from_ece_metric(ece, self.logger.log_dir + "/" + name)

            self.test_metrics[j].reset()
            self.test_metrics2[j].reset()

        metrics_summary.to_csv(
            self.logger.log_dir
            + f"/metrics_summary_{int(self.temperature_scale)}_{self.test_run}.csv"
        )

        if not os.path.exists(self.logger.log_dir + "/model_stored_manually.pth"):
            torch.save(
                self.model.state_dict(),
                self.logger.log_dir + "/model_stored_manually.pth",
            )

        if hasattr(self.trainer.datamodule, "ood_dataloader"):
            if isinstance(self.trainer.datamodule.test_dataloader(), list):
                dl = self.trainer.datamodule.test_dataloader()[0]
            else:
                dl = self.trainer.datamodule.test_dataloader()
            mean_ent_test, total_unc_test = get_ood_scores(
                dl,
                self.model,
                "cuda" if torch.cuda.is_available() else "cpu",
                self.mc_prediction_steps,
                self.logger.log_dir,
                temperature_scale=self.temperature_scale,
                ts_fn=self.temperature_scale_fn,
                name=f"test_{self.test_run}",
            )
            mean_ent_ood, total_unc_ood = get_ood_scores(
                self.trainer.datamodule.ood_dataloader(),
                self.model,
                "cuda" if torch.cuda.is_available() else "cpu",
                self.mc_prediction_steps,
                self.logger.log_dir,
                self.temperature_scale,
                ts_fn=self.temperature_scale_fn,
                name=f"ood_{self.test_run}",
            )

            with open(
                os.path.join(
                    self.logger.log_dir,
                    f"ood_mean_scores_{int(self.temperature_scale)}_{self.test_run}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "mean_ent_test": float(mean_ent_test),
                        "total_unc_test": float(total_unc_test),
                        "mean_ent_ood": float(mean_ent_ood),
                        "total_unc_ood": float(total_unc_ood),
                    },
                    f,
                )


def cli_main(args: ArgsType = None):
    global dataset
    dataset = sys.argv.pop(1)
    global DataModule
    DataModule, _ = get_datamodule_by_name(dataset)
    cli = LightningCLI(
        IPGPNetwork,
        DataModule,
        args=args,
        run=False,
        trainer_defaults=dict(
            callbacks=[
                ModelCheckpoint(monitor="val/accuracy", mode="max", save_last=True)
            ]
        ),
    )
    if USE_COMPILATION:
        model = torch.compile(cli.model)
    else:
        model = cli.model
    cli.trainer.fit(model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)  # , ckpt_path="best"
    cli.model.temperature_scale = True
    cli.trainer.test(cli.model, cli.datamodule)
    # cli.model.test_run = 'last'
    # cli.model.temperature_scale = False
    # cli.trainer.test(cli.model, cli.datamodule, ckpt_path="last")
    # cli.model.temperature_scale = True
    # cli.trainer.test(cli.model, cli.datamodule, ckpt_path="last")


class HelperModule(torch.nn.Module):
    def __init__(self, model, contrastive, temperature_scale=False, temp_fn=None):
        super().__init__()
        self.model = model
        self.contrastive = contrastive
        self.temperature_scale = temperature_scale
        self.temp_fn = temp_fn

    def forward(self, x):
        if self.contrastive:
            z, features = self.model(x)
        else:
            features = self.model(x)

        mean, cov = self.model.gp_classifier(features)

        if self.temperature_scale:
            mean = self.temp_fn(mean)
        return mean


if __name__ == "__main__":
    # cli_test()
    cli_main()
