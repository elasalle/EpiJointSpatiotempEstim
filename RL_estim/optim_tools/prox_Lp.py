# from https://github.com/juliana-du/Covid-R-estim/blob/main/include/optim_tools/prox_L1.py
# downloaded in 2025-03

# modifications by Etienne Lasalle:
# - modif1, 2025-03: add the prox of the squared L2 norm


import numpy as np


def positive_mask(array):
    """
    Also serves as projector on non negative values.
    :param array: any ndarray
    """
    return np.array([val > 0 for val in array])


def prox_L1(wx, gamma):
    """
    :param wx: ndarray of shape (1, days)
    :param gamma: float
    :return: ndarray of shape (1, days) = np.shape(wx) computing prox_L1(wx)
    Proximity operator of the l1 norm wp = prox_{gamma || .||_1}(wx)
    From MATLAB's implementation N. PUSTELNIK, CNRS, ENS Lyon
    June 2019
    """

    tmp = np.abs(wx) - gamma * np.ones(np.shape(wx))  # based on MATLAB's code
    signs = np.sign(wx)
    # Previous python version that is slower:
    # prev = tmp * positive_mask(tmp) * signs  # * np.sign(wx)
    return np.maximum(tmp, np.zeros(np.shape(tmp))) * signs

# modif1
def prox_L2(wx, gamma):
    """
    :param wx: ndarray of shape (1, days)
    :param gamma: float
    :return: ndarray of shape (1, days) = np.shape(wx) computing prox_L2(wx)
    Proximity operator of the squared l2 norm wp = prox_{gamma 1/2*|| .||_2^2}(wx)
    """

    return wx / (1+gamma)