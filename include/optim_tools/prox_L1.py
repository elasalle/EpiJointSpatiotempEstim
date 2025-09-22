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
