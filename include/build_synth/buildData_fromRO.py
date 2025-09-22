import numpy as np

from include import settings
from include.optim_tools import crafting_phi as phi

import pandas as pd


def randomDates(firstDay, days):
    """
    Returns a list of dates in format 'YYYY-MM-DD' from firstDay on until 'days' days later.
    :param firstDay: str in format 'YYYY-MM-DD'
    :param days: integer of days starting from firstDay. Also equals to len(dates).
    :return: dates: list of str in format 'YYYY-MM-DD'
    """
    randomDays = pd.date_range(firstDay, periods=days, freq='D')
    dates = [day.strftime("%Y-%m-%d") for day in randomDays]
    return dates


def firstCasesCorrection(data, REstim, OEstim):
    """
    Return corrected first cases of data so that firstCases does not have outliers effect.
    :param data: ndarray of shape (nbDeps, days)
    :param REstim: ndarray of shape (nbDeps, days)
    :param OEstim: ndarray of shape (nbDeps, days)
    :return: firstCases: ndarray of shape (nbDeps,)
             dates_crop: ndarray of shape (days - 1, )
             REstim_crop: ndarray of shape (nbDeps, days - 1)
             OEstim_crop: ndarray of shape (nbDeps, days - 1)
    """
    nbDeps, days = np.shape(REstim)
    firstCases = np.zeros(nbDeps)
    for k in range(nbDeps):
        if REstim[k][0] == 0:
            firstCases[k] = 0
        else:
            firstCases[k] = max((data[k][0] - OEstim[k][0]) / REstim[k][0], data[k][0])

    return firstCases, REstim[:, 1:], OEstim[:, 1:]


def buildData_anyRO(R, Outliers, firstCases, firstDay='2020-01-23', threshold=settings.thresholdPoisson, gamma=1.):
    """
    Build data Z drawn from Poisson distribution with mean (R * Phi Z + Out) with the given firstCases (1 day)
    :param R: ndarray of shape (days,)
    :param firstCases : number of cases of the original data, only used for initialization.
    :param Outliers: ndarray of shape (days, )
    :param firstDay: (optional) str in format 'YYYY-MM-DD' to indicate first day of random dates drawn
    :param threshold: (optional) inferior limit for Poisson parameter : float
    :param gamma: (optional) float, coefficient to modify the variance of poisson variables by sampling according to gamma*P(mean/gamma) instead of P(mean). 
    :return datesBuilt: list of str in format 'YYYY-MM-DD' random dates associated with ZDataBuilt
            ZDataBuilt: ndarray of shape (days, ) built following Cori's epidemiological model
    """
    days = len(R)  # should be one day less than the original data it was computed from
    assert (days == len(Outliers))

    Phi = phi.buildPhi()

    if days <= len(Phi) - 1:
        DataSizeErr = ValueError("Number of samples (days) for ground truth R too small. Should be greater than %d"
                                 % (len(Phi) - 1))
        raise DataSizeErr
    
    # Initialization with known values for the first day only
    ZData = np.zeros(days + 1)
    ZData[0] = firstCases
    # scale = firstCases / (np.max(Outliers[0]) * 150)  # WIP to ensure outliers and firstCases are on the same scale
    scale = 1
    OutliersRescaled = scale * Outliers
    realR = np.ones(days + 1)  # realR[0] not relevant since it will be cropped out

    tauPhi = len(Phi) - 1  # wrong explanation in associated papers (tauPhi = len(Phi) -1 = 25)
    # Modified convolution : normalized convolution for the first tauPhi days.
    for k in range(1, tauPhi):
        daysIterK = len(ZData[:k + 1])
        assert (daysIterK > 1)  # if there's only one day of data, can not compute ZPhi
        PhiNormalizedIterK = Phi[1:k + 1] / np.sum(Phi[1:k + 1])  # careful to use non-normalized Phi here !!
        fZ = np.flip(ZData[:k])  # 1st value of Phi is always 0 : we do not need data on day 0
        realR[k] = R[k-1] * np.sum(fZ * PhiNormalizedIterK)  # R is already cropped of day 1
        ZData[k] = np.random.poisson(max((realR[k] + OutliersRescaled[k-1])/gamma, threshold))*gamma   # Outliers are cropped of day 1

    PhiNormalized = Phi / np.sum(Phi)
    for k in range(tauPhi, days + 1):
        fZ = np.flip(ZData[k - len(Phi) + 1:k])
        realR[k] = R[k-1] * np.sum(fZ * PhiNormalized[1:])  # 1st value of Phi is always 0 : we don't need data on day 0
        ZData[k] = np.random.poisson(max((realR[k] + OutliersRescaled[k-1])/gamma, threshold))*gamma

    # options = {'dates': randomDates(firstDay, len(ZData[1:])),
    #            'data': ZData[1:]}  # cropped from the initialization with 'real' firstCases
    # return ZData[1:], options
    options = {'dates': randomDates(firstDay, len(ZData)),
               'data': ZData}  # cropped from the initialization with 'real' firstCases
    return ZData, options


def buildDataMulti_anyRO(R, Outliers, options=None, firstDay='2020-01-23', threshold=settings.thresholdPoisson):
    """
    Build data Z drawn from Poisson distribution with mean (R * Phi Z + Out) with the given firstCases for one day,
    by county.
    :param R: ndarray of shape (deps, days)
    :param options: dictionary containing at least:
        - B_matrix: ndarray of shape ([E|, |V|) transposed incidence matrix of the associated connectivity structure,
    represented by a graph G = (V, E) where each node corresponds to a territory/county.
        - firstCases: ndarray of shape (deps,) indicating number of cases of the original data, only used for init.
    :param Outliers: ndarray of shape (deps, days)
    :param firstDay: (optional) str in format 'YYYY-MM-DD' to indicate first day of random dates drawn
    :param threshold: (optional) inferior limit for Poisson parameter : float
    :return datesBuilt: list of str in format 'YYYY-MM-DD' random dates associated with ZDataBuilt
            ZDataBuilt: ndarray of shape (days, ) built following Cori's epidemiological model
    """
    firstCases = options['firstCases']

    deps, days = np.shape(R)
    ZData = np.zeros((deps, days))

    for d in range(deps):
        ZData[d], optionsTmp = buildData_anyRO(R[d], Outliers[d], firstCases[d], firstDay=firstDay, threshold=threshold)

    options_M = {'dates': randomDates(firstDay, len(ZData[0])),
                 'data': ZData,
                 'B_matrix': options['B_matrix']}
    return ZData, options_M
