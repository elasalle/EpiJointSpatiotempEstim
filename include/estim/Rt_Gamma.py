from include import settings
from include.optim_tools import crafting_phi

# Common libraries for computation
import numpy as np
import time


def Rt_Gamma(data, tau=settings.tauWindow, display=True, options=None):
    """
    Computes a Bayesian estimator (R_Gamma) of the reproduction number R for the chosen country and between dates 'fday'
    and 'lday' using a prior assuming that Rt is constant on time windows of length 'tau' days.
    :param data ndarray of shape (days, )
    :param tau : (optional) integer, number of days for which prior distribution is supposed piecewise constant
    :param options: dictionary containing at least
        - dates ndarray of shape (days, )
    :param display : (optional) bool whether displaying execution timle or not
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
    days = len(ZDataProc)
    Rt = np.zeros(days)
    if display:
        print("Computing Bayesian estimator ...")
    start = time.time()
    for t in range(days):
        posteriorA = settings.priorA + np.sum(ZDataProc[max(t - tau + 1, 0):t + 1])
        posteriorB = 1 / settings.priorB + np.sum(ZPhi[max(t - tau + 1, 0):t + 1])
        if posteriorB > 0:
            Rt[t] = posteriorA / posteriorB
        else:
            Rt[t] = 0
    executionTime = time.time() - start
    if display:
        print("Done in %.4f seconds ---" % executionTime)

    optionsGamma = {'dates': timestamps,
                    'data': ZDataProc,
                    'method': 'Gamma'}
    return Rt, optionsGamma

