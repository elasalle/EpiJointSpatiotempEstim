import numpy as np


class struct(object):
    """
    Mimics the mutiple types struct in MATLAB.
    Usually used to contain parameters of different types.
    """
    pass


def pyvec2matvec(arr):
    """
    Increase dimension from ndarray vector of shape (len(arr),) to MATLAB-shaped vector that is (1, len(arr))
    :param arr: ndarray of shape (len(arr),)
    :return: ndarray of shape (1, len(arr))
    """
    # arr = arr.reshape((1, len(arr)))
    arr = np.reshape(arr, (1, len(arr)))
    return arr


def matvec2pyvec(arr):
    """
    Flattens MATLAB-shaped vector that is (1, len(arr)) into ndarray vector of shape (len(arr),)
    :param arr: ndarray of shape (1, len(arr))
    :return: ndarray of shape (len(arr),)
    """
    shape = np.shape(arr)
    arr = arr.flatten()
    assert(len(arr) == np.max(shape))
    return arr

