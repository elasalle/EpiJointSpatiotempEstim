from include import settings
from include.optim_tools import crafting_phi

# Common libraries for computation
import numpy as np
import time


def Rt_MLE(data, options=None):
    """
    Computes the evolution of the reproduction number R for the chosen country and between dates 'fday' and 'lday'
    using the explicit Maximum-Likelihood Estimator
    :param data ndarray of shape (days, )
    :param options: dictionary containing
            - dates ndarray of shape (days, )
    :return: REstimate : ndarray of shape (days - 1, ), daily estimation of Rt
             OEstimate : ndarray of shape (days - 1, ), daily estimation of Outliers (none here)
             options: dictionary containing at least:
             - dates: ndarray of shape (counties, days -1) representing dates
             - data: ndarray of shape (counties, days - 1) representing processed data
    """
    dates = options['dates']
    # Preprocess : ONLY get rid of negative values
    data[data < 0] = 0

    # Compute Phi and convolution Phi * Z (here every vector is cropped from 1 day)
    Phi = crafting_phi.buildPhi(settings.phiBeta, settings.phiAlpha, settings.phiDays)
    timestamps, ZDataProc, ZPhi = crafting_phi.buildZPhi(dates, data, Phi)

    print("Computing Maximum Likelihood Estimator (MLE) ...")
    start = time.time()
    Rt = np.zeros(len(ZDataProc))
    Rt[ZPhi > 0] = ZDataProc[ZPhi > 0] / ZPhi[ZPhi > 0]
    executionTime = time.time() - start
    print("Done in %.4f seconds ---" % executionTime)

    optionsMLE = {'dates': timestamps,
                  'data': ZDataProc,
                  'method': 'MLE'}
    return Rt, optionsMLE
