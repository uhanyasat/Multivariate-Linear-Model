from typing import Union

import numpy as np
import sklearn.datasets as datasets


def load_regression_iris():
    '''
    Load the regression iris dataset that contains N
    input features of dimension F-1 and N target values.

    Returns:
    * features (np.ndarray): A [N x F-1] array of input features
    * targets (np.ndarray): A [N,] array of target values
    '''
    iris = datasets.load_iris()
    return iris.data[:, 0:3], iris.data[:, 3]


def split_train_test(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8
) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    np.random.seed(42)
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
        targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)
