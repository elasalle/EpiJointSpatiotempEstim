import numpy as np


def meanAndError(resByDraw, precDecim=2):
    """
    Averaging over draws and Monte Carlo simulation error (using std).
    :param resByDraw: ndarray of shape (nbDraws)
    :param precDecim: (optional) int
    :return:
    """
    nbDraws = len(resByDraw)
    return 1 / nbDraws * np.sum(resByDraw), np.round((1.96 / np.sqrt(nbDraws)) * np.std(resByDraw), decimals=precDecim)
