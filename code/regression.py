# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24, 2026

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


"""

import json

import gpytorch
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
import torch
from layer import SpectralBatchNorm1d
from models import linear
from pipeline import NetworkWrapper
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torchmetrics import NormalizedRootMeanSquaredError
from ucimlrepo import fetch_ucirepo
from utils import get_gp


class Tensors(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.data = torch.tensor(x, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        target = self.targets[idx]
        return sample, target


def batch_norm(features, spectral_norm=0.0):
    if spectral_norm:
        return SpectralBatchNorm1d(features, spectral_norm, track_running_stats=True)
    else:
        return torch.nn.BatchNorm1d(features)


class Module(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        num_classes,
        has_batch_norm=False,
        nonlin=torch.nn.ReLU,
        dropout=0.0,
        spectral_norm=0.0,
        use_residuals=False,
        residual_start_idx=0,
    ):
        super().__init__()
        self.use_residuals = use_residuals
        self.dropout = dropout
        self.hidden_sizes = hidden_sizes
        layers = []
        hs = hidden_sizes[0]
        layers.append(torch.nn.Sequential(linear(input_size, hs), nonlin()))
        if has_batch_norm:
            layers[0].append(batch_norm(hs, spectral_norm))
        for j, hsnext in enumerate(hidden_sizes[1:]):
            layers.append(torch.nn.Sequential(linear(hs, hsnext), nonlin()))
            if has_batch_norm:
                layers[j + 1].append(batch_norm(hsnext, spectral_norm))
            hs = hsnext

        self.layers = torch.nn.ModuleList(layers)
        self.drop = torch.nn.Dropout(dropout)
        self.fc = linear(hs, num_classes)
        self.residual_start_idx = residual_start_idx

    def forward(self, X, contrastive=False):
        idx = 0
        for s, block in zip(self.hidden_sizes, self.layers):
            if self.use_residuals and idx > self.residual_start_idx:
                if X.size(1) > s:
                    X_res = torch.nn.functional.max_pool1d(
                        X, X.size(1) // s, stride=X.size(1) // s
                    )
                else:
                    X_res = X
                X = block(X) + X_res
            else:
                X = block(X)
            idx += 1

        if contrastive:
            return X

        if self.dropout > 0.0:
            X = self.drop(X)
        X = self.fc(X)
        return X

    def final_forward(self, X):
        if self.dropout > 0.0:
            X = self.drop(X)
        X = self.fc(X)
        return X


def generate_1d_test_data(N=200):
    f1 = lambda x: np.exp(-0.3 * x**2) * (1.0 * np.sin(1.5 * x))
    f2 = lambda x: -1.0 * np.exp(-2.0 * x**2) * (0.5 * np.cos(4.0 * x))
    f = lambda x: f1(x) + f2(x - 1.0)

    # x_sample = np.random.rand(N) * 7 - 3.5
    x_sample = np.random.rand(N - int(0.05 * N)) * 7 - 3.5
    x_sample = np.concatenate([x_sample, np.random.rand(int(0.05 * N)) - 8])
    y_sample = f(x_sample) + 0.05 * np.random.randn(N)

    x_val = np.random.rand(N // 4) * 7 - 3.5
    y_val = f(x_val)

    x_test = np.linspace(-12.0, 8.0, 2000)
    y_test = f(x_test)

    # normalization:
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    x_sample_n = input_scaler.fit_transform(x_sample.reshape(-1, 1))
    y_sample_n = output_scaler.fit_transform(y_sample.reshape(-1, 1))
    train_set = Tensors(x_sample_n, y_sample_n)
    test_set = {
        "normalized_input": torch.tensor(
            input_scaler.transform(x_test.reshape(-1, 1)), dtype=torch.float32
        ),
        "input": x_test,
        "output": y_test,
        "x_sample": x_sample,
        "y_sample": y_sample,
        "output_scaler": output_scaler,
    }
    val_set = {
        "normalized_input": torch.tensor(
            input_scaler.transform(x_val.reshape(-1, 1)), dtype=torch.float32
        ),
        "normalized_output": torch.tensor(
            output_scaler.transform(y_val.reshape(-1, 1)), dtype=torch.float32
        ),
    }
    return train_set, val_set, test_set


def c_mixup(x):
    with torch.no_grad():
        batch_size = x.size(0)
        mixing_values = torch.rand(batch_size, device=x.device).unsqueeze(-1)
        pairing_idx = torch.roll(torch.arange(batch_size, device=x.device), 1)
        x_new = x * mixing_values + x[pairing_idx] * (1 - mixing_values)
        return torch.cat([x, x_new]), mixing_values


def reg_constrastive_loss(f, mixing_values, mode="mm"):
    f = torch.nn.functional.normalize(f, dim=-1)
    temperature = 0.07
    base_temperature = 0.07
    bsz = f.size(0) // 2
    f1, f2 = torch.split(f, [bsz, bsz], dim=0)
    pairing_idx = torch.roll(torch.arange(bsz, device=f.device), 1)
    f1c = f1  # .clone().detach()
    f2_from_f1 = f1c * mixing_values + f1c[pairing_idx] * (1 - mixing_values)

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
    return loss, f1


def pytorch_training_loop(
    train_set, model, optimizer, loss_fn, args, val_set=None, iph_model=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"])

    contrastive_args = args.get("contrastive_args", {})
    contrastive = contrastive_args.get("contrastive", False)
    contrastive_factor = contrastive_args.get("contrastive_factor", 0.5)
    contrastive_mode = contrastive_args.get("contrastive_mode", "mm")
    contrastive_head = torch.nn.Linear(50, 10).to(device)

    def update_lr(curr_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

    early_stopping = np.inf
    has_not_improved = 0

    for epoch in range(args["epochs"]):
        model.train()
        accumulated_loss = 0.0
        steps = 0
        separated_losses = [0.0, 0.0]

        if hasattr(model, "gp_classifier"):
            if hasattr(model.gp_classifier, "reset_precision_matrix"):
                model.gp_classifier.reset_precision_matrix()

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            if contrastive:
                inputs, mixing_vals = c_mixup(inputs)

            def closure():
                optimizer.zero_grad()
                if contrastive:
                    if isinstance(model, NetworkWrapper):
                        f = model(inputs)
                        cont_loss, pred = reg_constrastive_loss(
                            f, mixing_vals, mode=contrastive_mode
                        )
                    else:
                        f = model(inputs, contrastive=True)
                        fp = f  # contrastive_head(f)
                        cont_loss, _ = reg_constrastive_loss(
                            fp, mixing_vals, mode=contrastive_mode
                        )
                        pred = model.final_forward(f[: f.size(0) // 2])
                    loss = loss_fn(pred, targets)
                    add_loss = (
                        contrastive_factor * cont_loss + (1 - contrastive_factor) * loss
                    )
                    add_loss.backward()
                    separated_losses[0] += loss.item()
                    separated_losses[1] += cont_loss.item()
                    return add_loss.item()
                else:
                    pred = model(inputs)
                    loss = loss_fn(pred, targets)
                    loss.backward()
                    return loss.item()

            l = closure()

            optimizer.step()
            accumulated_loss += l
            steps += 1
            if epoch == 50:
                update_lr(1e-3)

            if epoch == 200:
                update_lr(0.2 * 1e-3)

            if has_not_improved >= args["early_stopping"] // 2:
                update_lr(0.2 * 1e-3)

        if val_set is not None:
            model.eval()
            val_inputs, val_targets = (
                val_set["normalized_input"].float().to(device),
                val_set["normalized_output"].float().to(device),
            )
            with torch.no_grad():
                val_pred = model(val_inputs)
                if iph_model:
                    val_pred = model.gp_classifier(val_pred)[0]  # take only mean
                val_loss = torch.sqrt(
                    torch.nn.functional.mse_loss(val_pred, val_targets)
                )
            print(
                f"Epoch {epoch + 1}/{args['epochs']}, Loss: {accumulated_loss / steps:.4f}, Val MSE: {val_loss.item():.4f}, Separated: {separated_losses[0] / steps:.4f}, {separated_losses[1] / steps:.4f}"
            )
            if args["early_stopping"]:
                if iph_model and epoch < 100:
                    continue
                if epoch < 50:
                    continue
                if val_loss.item() + 1e-4 >= early_stopping:
                    has_not_improved += 1
                else:
                    has_not_improved = 0
                if has_not_improved > args["early_stopping"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            early_stopping = min(early_stopping, val_loss.item())
        else:
            print(
                f"Epoch {epoch + 1}/{args['epochs']}, Loss: {accumulated_loss / steps:.4f}"
            )
    return device


def setup_model(module_args, gp_args):
    gp_args = gp_args.copy()
    feature_extractor = Module(**module_args)
    if len(list(gp_args.keys())) > 0:
        init_ips = gp_args.pop("initialization_ips", "random_normal")
        n_additional = gp_args.pop("n_additional_inducing_points", 0)
        init_forward_data = gp_args.pop("init_forward_data", None)
        model = NetworkWrapper(
            feature_extractor=feature_extractor,
            fe_args=module_args,
            gp_args=gp_args,
            use_contrastive=False,
            head="mlp",
            head_dim=20,
            initialization_ips=init_ips,
            n_additional_inducing_points=n_additional,
            init_forward_data=init_forward_data,
        )
        optimizer = torch.optim.AdamW(
            model.get_params_for_optimizer(), lr=1e-2, weight_decay=0.003
        )
        loss_fn = lambda x, y: model.gp_classifier.get_loss(x, y)

    else:
        model = feature_extractor
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.003)
        loss_fn = torch.nn.MSELoss()

    return model, optimizer, loss_fn


def setup_ref_model(module_args, gp_args, regression=True):
    gp_args = gp_args.copy()
    feature_extractor = Module(**module_args)
    feature_extractor.fc = torch.nn.Identity()

    class RefModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.gp_classifier, self.scaler = get_gp(
                gp_args,
                "",
                self.feature_extractor,
                "cuda" if torch.cuda.is_available() else "cpu",
            )

        def forward(self, x, contrastive=False):
            x_feat = self.feature_extractor(x)
            if contrastive:
                return x_feat
            x_feat = self.scaler(x_feat)
            return self.gp_classifier(x_feat)

        def final_forward(self, X):
            x_feat = self.scaler(X)
            return self.gp_classifier(x_feat)

    model = RefModel()

    if isinstance(model.gp_classifier, gpytorch.models.ApproximateGP):
        if regression:
            gll = gpytorch.likelihoods.GaussianLikelihood()
        else:
            gll = gpytorch.likelihoods.SoftmaxLikelihood()
        mll = gpytorch.mlls.VariationalELBO(
            gll, model.gp_classifier, num_data=gp_args["num_data"]
        )  # num_data might not work every time.
        loss_fn = lambda x, y: -mll(x, y).mean()
        params = [
            {"params": model.parameters()},
            {"params": gll.parameters(), "weight_decay": 0.0},
        ]
        loss_fn = (loss_fn, gll)
    else:
        if regression:
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        params = model.parameters()

    optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.003)
    return model, optimizer, loss_fn


NFOLDS = 5


def get_data(name):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=52)
    if name == "boston":
        file = ds.fetch_openml(data_id=531)
        X = file.frame.iloc[:, :-1].to_numpy()
        y = file.frame.iloc[:, -1].to_numpy().reshape(-1, 1)
    elif name == "energy":
        file = fetch_ucirepo(id=242)
        X = file.data.features.to_numpy()
        # use first output only
        y = file.data.targets.to_numpy()[:, :1]
    elif name == "power":
        file = fetch_ucirepo(id=294)
        X = file.data.features.to_numpy()
        y = file.data.targets.to_numpy()
    elif name == "protein":
        file = ds.fetch_openml(data_id=46588)
        X = file.data.to_numpy()
        y = file.target.to_numpy().reshape(-1, 1)
    elif name == "house_sales":
        file = ds.fetch_openml(data_id=44144)
        X = file.data.to_numpy()
        y = file.target.to_numpy().reshape(-1, 1)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    kf.get_n_splits(X)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i + 1}/{NFOLDS} on dataset {name}")
        Xtrain = X[train_index]
        Xtest = X[test_index]
        ytrain = y[train_index]
        ytest = y[test_index]

        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        Xtrain = input_scaler.fit_transform(Xtrain)
        Xtest = input_scaler.transform(Xtest)
        ytrain = output_scaler.fit_transform(ytrain)
        ytest = output_scaler.transform(ytest)
        yield Tensors(Xtrain, ytrain), Tensors(Xtest, ytest)


def metrics(y_true, y_pred):
    nrmse = NormalizedRootMeanSquaredError(normalization="mean")
    mse = torch.mean((y_true - y_pred) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    nrmse = nrmse(y_pred, y_true).item()
    return dict(mse=mse, rmse=rmse, mae=mae, nrmse=nrmse)


def create_dataloader(train_set, test_set, batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, test_loader


def ref_dkl_training(train_set, test_set, args):
    module_args, gp_args, training_args = args
    test_set = {
        "normalized_input": test_set.data,
        "normalized_output": test_set.targets,
    }
    model, optimizer, loss_fn = setup_ref_model(module_args, gp_args)

    if isinstance(loss_fn, tuple):
        loss_fn, gll = loss_fn
    else:
        gll = None
    device = pytorch_training_loop(
        train_set, model, optimizer, loss_fn, training_args, test_set
    )

    def test(data_in):
        model.eval()
        if hasattr(model, "gp_classifier"):
            if hasattr(model.gp_classifier, "reset_precision_matrix"):
                model.gp_classifier.mean_field_factor = None
        with torch.no_grad():
            y_hat = model(data_in.to(device))
            if gll is None:
                if isinstance(y_hat, tuple):
                    pred = [
                        model(data_in[i].unsqueeze(0).to(device))
                        for i in range(len(data_in))
                    ]
                    y_mean = torch.cat([p[0].cpu() for p in pred])
                    y_var = torch.cat([p[1].cpu() for p in pred])
                else:
                    raise ValueError(
                        "Expected tuple output from model when no likelihood is provided."
                    )
            else:
                gll.eval()
                y_dist = gll(y_hat)
                y_mean = y_dist.mean
                y_var = y_dist.variance
        return y_mean, y_var

    # TODO: Check why the results might be different
    return (test(test_set["normalized_input"]), test_set["normalized_output"]), (
        test(train_set.data),
        train_set.targets,
    )


def our_dkl_training(train_set, test_set, args):
    module_args, gp_args, training_args = args
    test_set = {
        "normalized_input": test_set.data,
        "normalized_output": test_set.targets,
    }
    iph_model = False
    if len(list(gp_args.keys())) > 0:
        gp_args["init_forward_data"] = train_set.data[
            : (gp_args["n_additional_inducing_points"] + 1), :
        ].clone()
        iph_model = True
    model, optimizer, loss_fn = setup_model(module_args, gp_args)
    device = pytorch_training_loop(
        train_set,
        model,
        optimizer,
        loss_fn,
        training_args,
        test_set,
        iph_model=iph_model,
    )

    def test(data_in):
        model.eval()
        with torch.no_grad():
            y_hat = model(data_in.to(device))
            if len(list(gp_args.keys())) > 0:
                y_hat = model.gp_classifier(y_hat)
            if isinstance(y_hat, tuple):
                y_mean = y_hat[0].cpu()
                y_std = y_hat[1].cpu().squeeze()
            else:
                y_mean = y_hat.cpu()
                y_std = None
        return y_mean, y_std

    return (test(test_set["normalized_input"]), test_set["normalized_output"]), (
        test(train_set.data),
        train_set.targets,
    )


def get_args_and_trainfn_for_method(method, dataset, input_size, ds_size):
    hidden_sizes = (
        [200, 100, 50] if dataset in ["boston", "energy"] else [1000, 500, 100, 50]
    )
    training_args = {
        "batch_size": 100 if dataset in ["boston", "energy"] else 1000,
        "epochs": 600 if dataset in ["boston", "energy"] else 500,
        "sam": False,
        "early_stopping": 50,
    }
    if method.lower().__contains__("cont"):
        training_args["contrastive_args"] = {
            "contrastive": True,
            "contrastive_factor": 0.3
            if dataset in ["boston", "energy", "power"]
            else 0.7,
            "contrastive_mode": "l2" if method.__contains__("IP-DKL") else "mm",
        }
    module_args = {
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "num_classes": 1,
        "has_batch_norm": False,
        "dropout": 0.0,
        "spectral_norm": 3.0 if method[-2:] == "SN" else 0.0,
        "use_residuals": True if not method == "NN" else False,
        "residual_start_idx": 1 if method[-2:] == "PR" else 0,
    }
    gp_args_laplace = {
        "name": "Laplace",
        "num_deep_features": 50,
        "num_gp_features": 50,
        "normalize_gp_features": True,
        "num_random_features": 512,
        "num_outputs": 1,
        "train_batch_size": training_args["batch_size"],
        "num_data": ds_size,
        "mean_field_factor": 25,
        "ridge_penalty": 1,
        "feature_scale": 2,
    }
    gp_args = {
        "initialization_ips": "random_gaussian",
        "n_additional_inducing_points": 49 if dataset in ["boston", "energy"] else 99,
        "init_forward_data": None,
        "alpha_eps": 0.1,
        "kernel": "scaled_l2_difference",
        "learnable_kernel_variance": False,
        "fixed_kernel_var": 0.2,
        "extra_linear": 0,
        "diagonal_q_u_covar": False,
        "reg": 1.0 / 20 if dataset in ["boston", "energy"] else 1.0 / 400,
        "whiten": True,
        "loss_type": "gaussian_nll_regression",
        "inducing_points_method": "vectors",
        "inducing_points_reg": 1e-5,
        "manifold_regularization": False,
        "direct_kernel_regularization": False,
        "direct_kernel_scale": 1e2,
        "orthogonal_ips": False,
        "normalize_gp_features": True,
    }
    if method.__contains__("NN"):
        return (module_args, {}, training_args), our_dkl_training
    elif method.__contains__("RFF-DKL"):
        return (module_args, gp_args_laplace, training_args), ref_dkl_training
    elif method.__contains__("IP-DKL"):
        return (module_args, gp_args, training_args), our_dkl_training
    else:
        raise ValueError(f"Unknown method: {method}")


def methods_iterator(dataset, input_size, ds_size):
    methods = ["IP-DKL", "IP-DKL-Cont"]  # 'NN', 'RFF-DKL-FR'   'NN', 'RFF-DKL-SN',
    # in 2 different batch-size settings?, with depth *2?
    for method in methods:
        yield (
            method,
            get_args_and_trainfn_for_method(method, dataset, input_size, ds_size),
        )


def uci_crossvalidation_plot():
    # ds_size always plus one, in case not devidable by NFOLDS
    datasets = [
        ("boston", 13, 405),
        ("energy", 8, 615),
        ("power", 4, 7655),
        ("protein", 9, 36585),
        ("house_sales", 15, 17291),
    ]
    results = {dataset[0]: {} for dataset in datasets}

    resfile = h5py.File("dkl_uci_results.h5", "w")

    for dataset, input_size, ds_size in datasets:
        print(f"Dataset: {dataset}, Dim: {input_size}, Num data: {ds_size}")
        ds_group = resfile.create_group(dataset)
        for method, (args, train_fn) in methods_iterator(dataset, input_size, ds_size):
            results[dataset][method] = {
                "train_results": {"mse": [], "rmse": [], "mae": [], "nrmse": []},
                "test_results": {"mse": [], "rmse": [], "mae": [], "nrmse": []},
            }
            results[dataset]["N"] = ds_size - 1
            results[dataset]["D"] = input_size
            print(f"Training method {method} on dataset {dataset}")
            idx = 1
            for kfold in get_data(dataset):
                train_set, test_set = kfold
                fold_group = ds_group.create_group(f"{method}_fold_{idx}")
                (out_test, yt_test), (out_train, yt_train) = train_fn(
                    train_set, test_set, args
                )
                yp_test, ypv_test = out_test
                yp_train, ypv_train = out_train
                test_metrics = metrics(yp_test, yt_test)
                train_metrics = metrics(yp_train, yt_train)
                for key in test_metrics.keys():
                    results[dataset][method]["test_results"][key].append(
                        test_metrics[key]
                    )
                    results[dataset][method]["train_results"][key].append(
                        train_metrics[key]
                    )

                fold_group.create_dataset(f"pY_test", data=yp_test.numpy())
                if ypv_test is not None:
                    fold_group.create_dataset(f"pYv_test", data=ypv_test.numpy())
                fold_group.create_dataset(f"Ytest", data=yt_test.numpy())
                fold_group.attrs["test_rmse"] = test_metrics["rmse"]

                fold_group.create_dataset(f"pY_train", data=yp_train.numpy())
                if ypv_train is not None:
                    fold_group.create_dataset(f"pYv_train", data=ypv_train.numpy())
                fold_group.create_dataset(f"Ytrain", data=yt_train.numpy())
                fold_group.attrs["train_rmse"] = train_metrics["rmse"]

                idx += 1

    # store results with json
    with open("dkl_uci_results.json", "w") as f:
        json.dump(results, f, indent=4)

    resfile.close()


def create_plot_for_uci_experiments(path="dkl_uci_results.json"):
    with open(path, "r") as f:
        results = json.load(f)

    datasets = list(results.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(4.2 * n_datasets, 4), sharey=True)

    # Use methods from the first dataset for consistent y-axis alignment
    global_methods = list(results[datasets[0]].keys())
    y_pos = np.arange(len(global_methods))
    y_shift_test = 0.2
    y_shift_train = -0.2

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        data = results[dataset]

        # Alternate background shading
        for j in range(len(global_methods)):
            if j % 2 == 0:
                ax.axhspan(j - 0.5, j + 0.5, color="lightgray", alpha=0.3, zorder=0)

        test_values = [
            np.mean(data[method]["test_results"]["rmse"]) for method in global_methods
        ]
        test_errors = [
            np.std(data[method]["test_results"]["rmse"]) for method in global_methods
        ]

        train_values = [
            np.mean(data[method]["train_results"]["rmse"]) for method in global_methods
        ]
        train_errors = [
            np.std(data[method]["train_results"]["rmse"]) for method in global_methods
        ]

        # testing points
        ax.errorbar(
            test_values,
            y_pos + y_shift_test,
            xerr=test_errors,
            fmt="o",
            color="blue",
            ecolor="blue",
            elinewidth=1.5,
            capsize=3,
            markersize=5,
            zorder=3,
        )
        ax.scatter(test_values, y_pos + y_shift_test, color="blue", s=25, zorder=4)

        # training points
        ax.errorbar(
            train_values,
            y_pos + y_shift_train,
            xerr=train_errors,
            fmt="o",
            color="orange",
            ecolor="orange",
            elinewidth=1.5,
            capsize=3,
            markersize=5,
            zorder=3,
        )
        ax.scatter(train_values, y_pos + y_shift_train, color="orange", s=25, zorder=4)

        # connect test_and_train
        for j in range(len(global_methods)):
            ax.plot(
                [train_values[j], test_values[j]],
                [y_pos[j] + y_shift_train, y_pos[j] + y_shift_test],
                color="gray",
                alpha=0.6,
                linewidth=1,
                zorder=2,
            )

        # Titles and labels
        ax.set_title(f"{dataset}\nN={data['N']}, D={data['D']}", fontsize=12)
        ax.set_yticks(y_pos)
        ax.tick_params(left=False)

        if i == 0:
            ax.set_yticklabels(global_methods, fontsize=12, ha="right")
            ax.set_ylabel("RMSEs", fontsize=13)

        # X-axis formatting
        # ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("dkl_uci_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # Example usage
    # dataset = get_data('house_sales')
    uci_crossvalidation_plot()
