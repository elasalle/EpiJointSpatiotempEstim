import numpy as np
from include import settings


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))  # python version


def sliding_median(data, alpha, nbDaysMed=settings.slidingMADDays):
    """
    Computes smoothened data with a sliding median using previous function mad.
    For exemple if alpha = 0, for each t, Z^{denoised}_t = MAD(Z_{t-width,..., t+width}).
    NON-CAUSAL VERSION.
    :param data: ndarray of shape (days,) in MATLAB format
    :param alpha: threshold for denoising. Explicitly alpha =max(|data_t - median(Z_{[t-w, t+w]})|) with w = nbDaysMed/2
    :param nbDaysMed: number of days on which this function is applied
    :return: data, data
    """

    totalDays = len(data)
    width = int((nbDaysMed - 1) / 2)  # if nbDaysMed is odd
    dataMed = np.zeros(np.shape(data))

    dataMedians = np.zeros(np.shape(data))
    dataMADs = np.zeros(np.shape(data))
    for k in range(0, totalDays):
        dataWindowed = data[max(k - width, 0): min(k + width + 1, totalDays)]  # 0 index for 1D version only
        # print(len(dataWindowed))
        dataWindowedMedian = np.median(dataWindowed)
        dataWindowedMAD = mad(dataWindowed)

        dataMedians[k] = dataWindowedMedian
        dataMADs[k] = dataWindowedMAD

        if abs(data[k] - dataWindowedMedian) > alpha * dataWindowedMAD:
            # windowedMAD[k] = dataWindowedMAD
            dataMed[k] = dataWindowedMedian
        else:
            dataMed[k] = data[k]

    return dataMed
