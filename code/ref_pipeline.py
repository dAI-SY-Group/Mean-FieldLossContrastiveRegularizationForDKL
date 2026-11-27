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
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torchvision
from pytorch_lightning.cli import ArgsType, LightningCLI

torchvision.disable_beta_transforms_warning()
from data_modules import get_datamodule_by_name
from model_container import container_chooser
from torchvision.transforms import v2
from utils import (
    AdaptiveECE,
    ClasswiseECE,
    MeanEntropy,
    compute_temperature_scale,
    entropy,
    get_calibration_plot_from_ece_metric,
    get_defaults,
    get_lr_scheduler,
    get_pytorch_optimizer_from_name,
)

dict_or_none = Union[dict, None]
USE_COMPILATION = False


@torch.no_grad()
def get_ood_scores(ood_loader, model, device, log_dir="", temp_scale=False, ts_fn=None):
    model.to(device)
    probabilities = []
    variance_scores = []
    for x, y in ood_loader:
        x, y = x.to(device), y.to(device)
        yhat = model.predict(x)
        if len(yhat.shape) > 2:
            variance_scores.append(yhat.std(0))
            yhat = yhat.mean(0)
        if temp_scale:
            yhat = ts_fn(yhat)
        if model.name not in ["SVDKL", "DUE", "DKL", "PosteriorNetwork"]:
            yhat = torch.nn.functional.softmax(yhat, dim=-1)
        probabilities.append(yhat)

    probabilities = torch.cat(probabilities)
    if log_dir:
        torch.save(probabilities.cpu(), log_dir + "/ood_probabilities.pth")

    entropy_scores = entropy(probabilities)
    if len(variance_scores) > 0:
        variance_scores = torch.cat(variance_scores)
        if log_dir:
            np.savez(
                log_dir + f"/ood_scores_{int(temp_scale)}.npz",
                entropy_scores=entropy_scores.clone().detach().cpu().numpy(),
                variance_scores=variance_scores.clone().detach().cpu().numpy(),
            )
        return entropy_scores.mean(), variance_scores.mean()
    else:
        if log_dir:
            np.savez(
                log_dir + f"/ood_scores_{int(temp_scale)}.npz",
                entropy_scores=entropy_scores.clone().detach().cpu().numpy(),
            )
        return entropy_scores.mean(), torch.zeros(1)


class PipelineModule(pl.LightningModule):
    """Pipeline supports different DKL and DNN models served as a model_container"""

    def __init__(
        self,
        name: str,
        use_defaults: bool = False,
        model_params: dict = {"name": "WideResNet_28_10"},
        gaussian_process_params: dict_or_none = {},
        loss_params: dict = {},
        optim_params: dict = {},
        scheduler_params: dict = {},
        n_train_samples: int = 47500,
        mixup_cutmix_alphas: list = [0.0, 0.0],
        float32_matmul_precision: str = "",
    ):
        super().__init__()
        assert "name" in model_params, "No model specified!"
        self.name = name

        if float32_matmul_precision:
            torch.set_float32_matmul_precision(float32_matmul_precision)

        model_class = container_chooser(name)

        if gaussian_process_params is not None:
            if len(gaussian_process_params) < 1:
                gaussian_process_params = None

        if use_defaults:
            (
                model_params,
                gaussian_process_params,
                loss_params,
                optim_params,
                scheduler_params,
            ) = get_defaults(name, dataset, model_params["name"])

        if gaussian_process_params is not None:
            self.model = model_class(
                name,
                model_params,
                gaussian_process_params,
                loss_params,
                optim_params,
                dataset,
                n_train_samples,
                num_classes,
                device=self.device,
            )
        else:
            # print(name, model_params, loss_params, optim_params)
            loss_params["name"] = "CCE"
            self.model = model_class(name, model_params, loss_params, optim_params)

        self.optimizer_name = optim_params["name"]
        self.optimizer_args = {} if "args" not in optim_params else optim_params["args"]

        self.scheduler_params = scheduler_params
        self.scheduler_requires_batch_idx = False

        self.automatic_optimization = False
        self.temperature_scale = False
        self.temperature_scale_fn = None

        self.add_aug = any(mixup_cutmix_alphas)
        if self.add_aug:
            mixup_alpha, cutmix_alpha = mixup_cutmix_alphas
            if mixup_alpha and not cutmix_alpha:
                self.add_aug = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
            if not mixup_alpha and cutmix_alpha:
                self.add_aug = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
            if cutmix_alpha and mixup_alpha:
                self.add_aug = v2.RandomChoice(
                    [
                        v2.MixUp(alpha=mixup_alpha, num_classes=num_classes),
                        v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes),
                    ]
                )

        # separate metrics for dataloaders
        self.current_test_loader = 0
        self.n_test_loader = 5 * 5 + 1
        if dataset.lower().__contains__("imagenet"):
            self.n_test_loader += 2
        if dataset.lower().__contains__("tiny"):
            self.n_test_loader -= 5
        if dataset.lower().__contains__("mnist"):
            self.n_test_loader = 6
        self.metrics = tm.MetricCollection(
            {
                "val/mcc": tm.MatthewsCorrCoef(
                    task="multiclass", num_classes=num_classes
                ),
                "val/accuracy": tm.Accuracy(task="multiclass", num_classes=num_classes),
                "val/f1-score": tm.F1Score(task="multiclass", num_classes=num_classes),
                "val/auroc": tm.AUROC(task="multiclass", num_classes=num_classes),
                "val/ece": tm.CalibrationError(
                    task="multiclass", num_classes=num_classes, norm="l1"
                ),
                "val/mce": tm.CalibrationError(
                    task="multiclass", num_classes=num_classes, norm="max"
                ),
                "val/adaptive_ece": AdaptiveECE(n_bins=15),
                "val/classwise_ece": ClasswiseECE(num_classes=num_classes, n_bins=15),
                "val/ment": MeanEntropy(),
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

        self.cce = torch.nn.CrossEntropyLoss()
        self.has_trained = False
        self.track_variables_storage = []
        self.store_test_logits = True
        self.test_logits = []
        self.save_hyperparameters()

    def configure_optimizers(self):
        parameters, opt_args = self.model.parameter_list()
        optimizer = get_pytorch_optimizer_from_name(
            self.optimizer_name, parameters, opt_args
        )
        if len(self.scheduler_params) > 0:
            scheduler, self.scheduler_requires_batch_idx = get_lr_scheduler(
                self.scheduler_params, optimizer
            )
        else:
            return [optimizer]
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def lr_scheduler_step(self, scheduler, batch_idx):
        scheduler.step(epoch=self.current_epoch, batch_idx=batch_idx)

    def on_train_epoch_start(self) -> None:
        if self.name.__contains__("SVGP"):
            self.model.gaussian_process.reset_precision_matrix()

    def training_step(self, batch, batch_idx):
        if len(batch) > 2:
            x, y, inds = batch
        else:
            x, y = batch
            inds = None
        tensorboard = self.logger.experiment

        if self.add_aug:
            x, y = self.add_aug(x, y)

        opt = self.optimizers()
        opt.zero_grad()

        def closure():
            loss = self.model.forward_loss(x, y, tensorboard, inds)
            self.manual_backward(loss)
            return loss

        loss = closure()

        self.log(
            "loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        opt.step()

        if len(self.scheduler_params) > 0 and self.scheduler_requires_batch_idx:
            sch = self.lr_schedulers()
            sch.step(self.current_epoch, batch_idx)
            for m, lr in enumerate(sch.get_last_lr()):
                self.log(f"learning_rate_{m}", lr, on_step=True, sync_dist=True)
        self.has_trained = True
        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.scheduler_params) > 0 and not self.scheduler_requires_batch_idx:
            # log epoch wise learning rate
            sch = self.lr_schedulers()
            sch.step()
            for m, lr in enumerate(sch.get_last_lr()):
                self.log(f"learning_rate_{m}", lr, sync_dist=True)
        if self.name.__contains__("NoiseEst"):
            self.track_variables_storage.append(self.model.finalize_stats_buffers())

    def on_train_end(self) -> None:
        self.model.eval()

        def model_forward(x):
            yhat = self.model.predict(x)
            if len(yhat.shape) > 2:
                yhat = yhat.mean(0)
            return yhat

        self.temperature_scale_fn = compute_temperature_scale(
            model_forward, self.trainer.datamodule.val_dataloader()
        )

        if self.name.__contains__("NoiseEst"):
            n_vals = len(self.track_variables_storage[0])
            for n in range(n_vals):
                vectors = [t[n] for t in self.track_variables_storage]
                vectors = torch.stack(vectors).detach().cpu().numpy()
                np.save(self.logger.log_dir + "/BufVariable" + str(n) + ".npy", vectors)
            self.model.finalize_stats_buffers(self.logger.log_dir)
            if self.model.track_features_of_n_samples:
                samples = self.model.tracked_samples.astype("int")
                ds = self.trainer.train_dataloader.dataset
                labels = [ds[sample][1] for sample in samples]
                labels = np.stack(labels)
                np.savez(
                    self.logger.log_dir + "/samples_labels.npz",
                    samples=samples,
                    labels=labels,
                )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.has_trained:
            yhat = self.model.predict(x)
            if len(yhat.shape) > 2:
                yhat = yhat.mean(0)
            # val_cce = self.model.loss(yhat, y)
            self.metrics.update(yhat, y)
            # self.log('val_cce', val_cce, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if self.has_trained:
            metric_dict = self.metrics.compute()
            self.log(
                "val_accuracy",
                metric_dict["val/accuracy"],
                prog_bar=True,
                sync_dist=True,
            )
            self.metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        yhat = self.model.predict(x)

        if len(yhat.shape) > 2:
            yhat = yhat.mean(0)

        if self.temperature_scale:
            yhat = self.temperature_scale_fn(yhat)

        if self.store_test_logits and (dataloader_idx == 0 or dataloader_idx == 5):
            self.test_logits.append(
                (dataloader_idx, y.clone().cpu(), yhat.clone().cpu())
            )

        test_cce = self.cce(yhat, y)
        if self.name not in ["SVDKL", "DUE", "DKL", "PosteriorNetwork"]:
            yhat = torch.nn.functional.softmax(yhat, dim=-1)
        self.test_metrics[dataloader_idx].update(yhat, y)

        self.log(
            f"test_cce",
            test_cce,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            self.test_metrics[dataloader_idx],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    @staticmethod
    def metrics_to_dict(metric):
        return {k: v.item() for k, v in metric.compute().items()}

    def on_test_end(self) -> None:
        name_list = list(self.test_metrics[0].keys())
        metrics_summary = pd.DataFrame(
            data=np.zeros((self.n_test_loader, len(name_list))),
            columns=name_list,
            index=["test"] + self.trainer.datamodule.corrupted_testSets_names,
        )

        for j in range(self.n_test_loader):
            fd = self.metrics_to_dict(self.test_metrics[j])

            metrics_summary.iloc[j] = pd.Series(fd)
            # print(metrics_summary)

            ece = self.test_metrics[j]["test/ece"]
            name = f"calibration_plot_test_{j}_{int(self.temperature_scale)}"
            get_calibration_plot_from_ece_metric(ece, self.logger.log_dir + "/" + name)

            self.test_metrics[j].reset()

        metrics_summary.to_csv(
            self.logger.log_dir + f"/metrics_summary_{int(self.temperature_scale)}.csv"
        )

        # store test logits
        if self.store_test_logits:
            tmp_0 = tmp_5 = []
            for dlidx, y, yhat in self.test_logits:
                tmp_0.append((y, yhat)) if dlidx == 0 else tmp_5.append((y, yhat))
            y0 = torch.cat([y for y, yhat in tmp_0]).numpy()
            yhat0 = torch.cat([yhat for y, yhat in tmp_0]).numpy()
            np.savez(self.logger.log_dir + "/test_logits_0", y=y0, yhat=yhat0)
            y5 = torch.cat([y for y, yhat in tmp_5]).numpy()
            yhat5 = torch.cat([yhat for y, yhat in tmp_5]).numpy()
            np.savez(self.logger.log_dir + "/test_logits_5", y=y5, yhat=yhat5)

        torch.save(
            self.model.state_dict(), self.logger.log_dir + "/model_stored_manually.pth"
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
                temp_scale=self.temperature_scale,
                ts_fn=self.temperature_scale_fn,
            )
            mean_ent_ood, total_unc_ood = get_ood_scores(
                self.trainer.datamodule.ood_dataloader(),
                self.model,
                "cuda" if torch.cuda.is_available() else "cpu",
                self.logger.log_dir,
                self.temperature_scale,
                self.temperature_scale_fn,
            )

            with open(
                os.path.join(
                    self.logger.log_dir,
                    f"ood_mean_scores_{int(self.temperature_scale)}.json",
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "mean_ent_test": mean_ent_test.item(),
                        "total_unc_test": total_unc_test.item(),
                        "mean_ent_ood": mean_ent_ood.item(),
                        "total_unc_ood": total_unc_ood.item(),
                    },
                    f,
                )


def cli_main(args: ArgsType = None):
    global dataset
    dataset = sys.argv.pop(1)
    global DataModule, num_classes
    DataModule, num_classes = get_datamodule_by_name(dataset)
    cli = LightningCLI(PipelineModule, DataModule, args=args, run=False)
    if USE_COMPILATION:
        model = torch.compile(cli.model)
    else:
        model = cli.model
    cli.trainer.fit(model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)
    cli.model.temperature_scale = True
    cli.trainer.test(cli.model, cli.datamodule)


# def cli_test(args: ArgsType = None):
#     cli = LightningCLI(PipelineModule, DataModule, args=args, run=False)
#     cli.model.load_from_checkpoint('lightning_logs/aleatoric_cip_conflow/Imagenet100/Base/ResNet50/lightning_logs/version_9/checkpoints/epoch=1-step=316.ckpt')
#     cli.trainer.test(cli.model, cli.datamodule)
#     cli.model.temperature_scale = True
#     cli.trainer.test(cli.model, cli.datamodule)


# def cli_test():
#     trainer = pl.Trainer()
#     model = PipelineModule.load_from_checkpoint('lightning_logs/version_39/checkpoints/epoch=349-step=98700.ckpt')
#     data = DataModule(batch_size=200, num_workers_train=12, num_workers_test=12, normalize=True, transform='base')
#     trainer.test(model, data)


if __name__ == "__main__":
    cli_main()
