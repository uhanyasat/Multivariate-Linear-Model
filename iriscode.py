# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 22:19:54 2021

@author: Sathish
"""

from typing import Union
from numpy import *
import math
import numpy as np
import sklearn.datasets as datasets
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def load_regression_iris():
    iris = datasets.load_iris()
    return iris.data[:, 0:3], iris.data[:, 3]

X, t = load_regression_iris()
N, D = X.shape

M, sigma, i = 10, 10, 0
mu = np.zeros((M, D))
while i < D:
    mmin = np.min(X[i, :])
    mmax = np.max(X[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)
    i += 1

def mvn_basis(X, mu=None, sigma=None):
    m = X.shape[0]
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0, ddof=1)
    # don't change the intercept term
    mu[0] = 0.0
    sigma[0] = 1.0
    for i in range(m):
        X[i, :] = (X[i, :] - mu) / sigma
    return X, mu, sigma
x,m,s=mvn_basis(X, mu=None, sigma=None)
print(x)
print(m)
print(s)
y = multivariate_normal.pdf(X, mean=m, cov=s)
print(y)

plt.plot(y)
plt.show()

lamda = 0.001
def likelihood_linear_model(y, yhat,lamda):
	return yhat * y + (1 - yhat) * (1 - y)*lamda
mx=likelihood(y, t,lamda)
print(mx)

