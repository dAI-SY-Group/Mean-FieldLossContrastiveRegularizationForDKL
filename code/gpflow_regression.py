# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24, 2026

@author: Matthias Schmal
@tui: Data Intensive Systems and Visualization Group (dAI.SY)


install environment with:
conda create --name gpflow python=3.10
conda activate gpflow
pip install gpflow tensorflow~=2.10.0 tensorflow-probability~=0.18.0
pip3 install -U scikit-learn
pip3 install -U ucimlrepo
pip install matplotlib
pip install tqdm

"""

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import time
import itertools
import h5py
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


NFOLDS = 5
rng = np.random.RandomState(123)
tf.random.set_seed(42)



def get_data(name):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=52)
    if name == 'boston':
        file = ds.fetch_openml(data_id=531)
        X = file.frame.iloc[:, :-1].to_numpy()
        y = file.frame.iloc[:, -1].to_numpy().reshape(-1, 1)
    elif name == 'energy':
        file = fetch_ucirepo(id=242)
        X = file.data.features.to_numpy()
        # use first output only
        y = file.data.targets.to_numpy()[:, :1]
    elif name == 'power':
        file = fetch_ucirepo(id=294)
        X = file.data.features.to_numpy()
        y = file.data.targets.to_numpy()
    elif name == 'protein':
        file = ds.fetch_openml(data_id=46588)
        X = file.data.to_numpy()
        y = file.target.to_numpy().reshape(-1, 1)
    elif name == 'house_sales':
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
        yield (Xtrain, ytrain), (Xtest, ytest)


def check_elbo(model: gpflow.models.GPModel, train_dataset: tf.data.Dataset, N: int):
    # Evaluate objective for different minibatch sizes
    elbo = tf.function(model.elbo)

    minibatch_proportions = np.logspace(-2, 0, 10)
    times = []
    objs = []
    for mbp in minibatch_proportions:
        batchsize = int(N * mbp)
        train_iter = iter(train_dataset.batch(batchsize))
        start_time = time.time()
        objs.append([elbo(minibatch) for minibatch in itertools.islice(train_iter, 20)])
        times.append(time.time() - start_time)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(minibatch_proportions, times, "x-")
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")

    ax2.plot(minibatch_proportions, np.array(objs), "kx")
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    plt.show()


def train_model(
    model: gpflow.models.GPModel, train_dataset: tf.data.Dataset
) -> None:
    # gpflow.set_trainable(model.inducing_variable, False)
    elbo = tf.function(model.elbo)
    minibatch_size = 100

    def run_adam(model, iterations):
        """
        Utility function running the Adam optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(train_dataset.batch(minibatch_size))
        training_loss = model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam()

        @tf.function
        def optimization_step():
            tfp.math.minimize(training_loss, 1, optimizer, trainable_variables=model.trainable_variables)

        for step in tqdm(range(iterations)):
            optimization_step()
            if step % 10 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)
        return logf

    maxiter = 20000
    logf = run_adam(model, maxiter)
    plt.plot(np.arange(maxiter)[::10], logf)
    plt.xlabel("iteration")
    _ = plt.ylabel("ELBO")
    plt.show()


def main():
    datasets = [('boston', 13, 404), ('energy', 8, 614), ('power', 4, 7654), ('protein', 9, 36584),
                ('house_sales', 15, 17290)]

    resfile = h5py.File("gpflow_svgp_results.h5", "w")

    for dataset, dim, num_data in datasets:
        print(f"Dataset: {dataset}, Dim: {dim}, Num data: {num_data}")
        ds_group = resfile.create_group(dataset)
        idx = 1
        for fold in get_data(dataset):
            print(f"Fold {idx}/{NFOLDS}")
            fold_group = ds_group.create_group(f"SVGP_fold_{idx}")

            (Xtrain, Ytrain), (Xtest, Ytest) = fold

            M = 100 if dataset in ['boston', 'energy'] else 1000
            model = gpflow.models.SVGP(
                kernel=gpflow.kernels.SquaredExponential(),
                likelihood=gpflow.likelihoods.Gaussian(),
                inducing_variable=Xtrain[:M, :].copy(),
                num_data=num_data
            )
            train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain)).repeat().shuffle(num_data)
            # check_elbo(model, train_dataset, num_data)
            train_model(model, train_dataset)

            # test model
            tensor_data = tuple(map(tf.convert_to_tensor, (Xtest, Ytest)))
            pY, pYv = model.predict_y(tensor_data[0])
            rmse = tf.sqrt(tf.reduce_mean((pY - tensor_data[1]) ** 2))
            print(f"Test RMSE: {rmse.numpy():.4f}")

            fold_group.create_dataset(f"pY_test", data=pY.numpy())
            fold_group.create_dataset(f"pYv_test", data=pYv.numpy())
            fold_group.create_dataset(f"Ytest", data=Ytest)
            fold_group.attrs['test_rmse'] = rmse.numpy()

            tensor_data_train = tuple(map(tf.convert_to_tensor, (Xtrain, Ytrain)))
            pYtrain, pYvtrain = model.predict_y(tensor_data_train[0])
            rmsetrain = tf.sqrt(tf.reduce_mean((pYtrain - tensor_data_train[1]) ** 2))

            fold_group.create_dataset(f"pY_train", data=pYtrain.numpy())
            fold_group.create_dataset(f"pYv_train", data=pYvtrain.numpy())
            fold_group.create_dataset(f"Ytrain", data=Ytrain)
            fold_group.attrs['train_rmse'] = rmsetrain.numpy()

            idx += 1

            # plt.figure()
            # plt.plot(tensor_data[1], pY, "kx")
            # plt.plot([-3, 3], [-3, 3], "r--")
            # plt.xlabel("True values")
            # plt.ylabel("Predicted values")
            # plt.title(f"Test RMSE: {rmse.numpy():.4f}")
            # plt.show()
    resfile.close()


if __name__ == "__main__":
    main()
