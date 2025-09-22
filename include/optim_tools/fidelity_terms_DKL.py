import numpy as np


def DKL_no_outlier(X, Y, alpha):
    """
    Used in CP_covid_4_graph.py file.
    :param X: ndarray with shape (1, days), should be estimated R
    :param Y: ndarray with shape (1, days), should be data
    :param alpha: ndarray with shape (days,), should be the convolution between data and Phi (infectiousness)
    :return: float
    Returns the Kullback-Leibler divergence.
    Note: for an observation z in R degraded by a Poisson noise with parameter
    alpha, the KL divergence reads
    KL(x,y,alpha) = alpha*x - y + y log(y/alpha*x)           if y>0 & x>0,
    KL(x,y,alpha) = alpha*x                                  if y=0 & x>=0,
    KL(x,y,alpha) = +Inf                                     otherwise.
    """
    r = X
    zData = Y

    x = r * alpha

    j = (zData > 0) * (x > 0)  # first sum in each dKL
    k = (zData == 0) * (x >= 0)  # second sum in each dKL

    Ytildj = x[j] - zData[j] * np.log(x[j])  # (outside constant computation)
    Ytildk = x[k]
    return np.sum(Ytildk) + np.sum(Ytildj)


def DKLw_outlier(X, Y, alpha):
    """
    Used in CP_covid_5_outlier and multivariate ver.
    :param X: ndarray with shape (2, dep, days), should be estimated R and estimated O
    :param Y: ndarray with shape (dep, days), should be data
    :param alpha: ndarray with shape (dep, days), should be the convolution between data and Phi (infectiousness)
    :return: float
    Returns the Kullback Leibler divergence.
    Note: for an observation z in R degraded by a Poisson noise with parameter
    alpha, the KL divergence reads
    KL(x,y,alpha) = alpha*x - y + y log(y/alpha*x)           if y>0 & x>0,
    KL(x,y,alpha) = alpha*x                                  if y=0 & x>=0,
    KL(x,y,alpha) = +Inf                                     otherwise.
    """
    r = X[0]
    o = X[1]
    zData = Y

    x = alpha * r + o

    j = (zData > 0) * (x > 0)
    k = (zData == 0) * (x >= 0)

    Ytildj = x[j] - zData[j] * np.log(x[j])  # (outside constant computation)
    Ytildk = x[k]
    return np.sum(Ytildj) + np.sum(Ytildk)

# ASSOCIATED PROX OPERATORS -------------------------------------------------------------------------------------------


def prox_DKL_no_outlier(x, data, alpha, gamma):
    """
    Computes the proximal operator associated to gamma * Kullback-Leibler divergence between 'alpha * x' and 'data'.
    We denote days = len(data).
    :param x: ndarray of shape (1, days)
    :param data: ndarray of same (1, days)
    :param alpha: ndarray of shape (days,) equivalent to (1, days) shape,  or float == 1 when not used
    :param gamma: float
    :return: prox ndarray of shape (1, days)

    """
    x = x.astype('float64')
    prox = (x - gamma * alpha + np.sqrt(np.abs(x - gamma * alpha) ** 2 + 4 * gamma * data)) / 2

    prox[(alpha == 0) * (data == 0)] = 0
    return prox


def prox_DKLw_outlier(X, data, alpha, tau):
    """
    Computes the proximal operator associated to gamma * Kullback-Leibler divergence between 'alpha * x' and 'data'.
    We denote days = len(data). This function is not used in practice, see version below.
    :param X: ndarray of shape (3, 1, days)
    :param data: ndarray of shape (1, days)
    :param alpha: ndarray of shape (days,) equivalent to (1, days) shape
    :param tau: float
    :return: prox ndarray of shape (2, 1, days)
    """
    X1 = X[0, :]
    X2 = X[1, :]
    tmp = alpha * X1 + X2
    prox_DKL = prox_DKL_no_outlier(tmp, data, 1, tau * (alpha ** 2 + 1))

    preprox1 = alpha * (tmp - prox_DKL) / (alpha ** 2 + 1)
    preprox2 = (tmp - prox_DKL) / (alpha ** 2 + 1)
    prox = X - np.array([preprox1, preprox2])
    return prox


def prox_DKLw_outlier_0cas(X, data, alpha, tau):
    """
    Returns the proximal operator associated to gamma * Kullback-Leibler divergence between 'alpha * x' and 'data' and
    that takes into account the zero values in 'y'. y represents data. We denote days = len(data).
    :param X: ndarray of shape (3, 1, days)
    :param data: ndarray of shape (1, days)
    :param alpha: ndarray of shape (days,) equivalent to (1, days) shape
    :param tau: float
    :return: prox ndarray of shape (2, 1, days)

    """
    X1 = X[0, :]
    X2 = X[1, :]
    RPhiZO = alpha * X1 + X2
    prox_DKL = prox_DKL_no_outlier(RPhiZO, data, 1, tau * (alpha ** 2 + 1))

    preprox1 = alpha * (RPhiZO - prox_DKL) / (alpha ** 2 + 1)
    preprox2 = (RPhiZO - prox_DKL) / (alpha ** 2 + 1)
    prox1 = X1 - preprox1
    prox2 = X2 - preprox2
    prox1[(data == 0) * (alpha == 0)] = 0
    prox2[(data == 0) * (alpha == 0)] = 0
    return np.array([prox1, prox2])
