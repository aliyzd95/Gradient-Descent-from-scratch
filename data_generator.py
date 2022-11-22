import random

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs, make_regression, \
    make_s_curve, make_swiss_roll, make_classification

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_dataset(d):
    scaler = StandardScaler()
    if d == 0:
        X, y = make_regression(n_samples=980, n_features=1, noise=15)
        X = scaler.fit_transform(X)
        y = y.reshape(len(y), 1)
        y = scaler.fit_transform(y)
        df = pd.DataFrame(dict(x=X[:, 0], y=y[:, 0], label=''))
        df[['x', 'y']].to_csv('dataset/data0.csv', index=False)
        # plt.scatter(X[:, 0], y, marker='.')

    elif d == 1:
        X, y = make_moons(n_samples=1000, noise=0.1)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('dataset/data1.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 2:
        X, y = make_circles(n_samples=1000, noise=0.1)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('dataset/data2.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 3:
        X, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=0.5)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('dataset/data3.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 4:
        X, y = make_swiss_roll(n_samples=1000, noise=0.5)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 2], label=y))
        df[['x', 'y']].to_csv('dataset/data4.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 2], marker='.')


    elif d == 5:
        X, y = make_s_curve(n_samples=1000, noise=0.2)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 2], label=y))
        df[['x', 'y']].to_csv('dataset/data5.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 2], marker='.')

    elif d == 6:
        X, y = make_blobs(n_samples=1000, centers=4, n_features=2, cluster_std=0.5)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('dataset/data6.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 7:
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('dataset/data7.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 8:
        X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5)
        X = scaler.fit_transform(X)
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
        df[['x', 'y']].to_csv('dataset/data8.csv', index=False)
        # plt.scatter(X[:, 0], X[:, 1], marker='.')

    elif d == 9:
        X, y = make_blobs(n_samples=1000, centers=3, n_features=2)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X_aniso = np.dot(X, transformation)
        X_aniso = scaler.fit_transform(X_aniso)
        df = pd.DataFrame(dict(x=X_aniso[:, 0], y=X_aniso[:, 1], label=y))
        df[['x', 'y']].to_csv('dataset/data9.csv', index=False)
        # plt.scatter(X_aniso[:, 0], X_aniso[:, 1], marker='.')

    else:
        raise Exception('not true!')




