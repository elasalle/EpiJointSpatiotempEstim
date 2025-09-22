import numpy as np


def prox_L2(wx, gamma):
    """
    :param wx: ndarray of shape (1, days)
    :param gamma: float
    :return: ndarray of shape (1, days) = np.shape(wx) computing prox_L2(wx)
    Proximity operator of the squared l2 norm wp = prox_{gamma 1/2*|| .||_2^2}(wx)
    """

    return wx / (1+gamma)