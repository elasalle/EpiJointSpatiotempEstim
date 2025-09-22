import os

import numpy as np
from scipy.io import loadmat, savemat


from include.build_synth.Tikhonov_method import Tikhonov_spat_corr


def compute_delta_withG(REstimates, B_matrix, firstPower=-15, lastPower=5, step=1, prec=10**(-3), fileSuffix='Last'):
    """
    Returns delta_min, delta_max that ensure that over for delta > delta_max, there is the same estimation on every node
    (total diffusion) and below delta_min, the diffusion is not effective (no difference between diffusion or not) by
    computing the squared l2 norm of the matrix-vector product between B_matrix and REstimates (regularization term).
    :param REstimates: ndarray of shape (counties, days)
    :param B_matrix: ndarray of shape (edges, counties) transposed incidence matrix of the associated B_matrix
    :param firstPower: (optional)
    :param lastPower: (optional)
    :param step: (optional)
    :param prec: (optional)
    :param fileSuffix: (optional)
    :return: delta : ndarray of shape (20,)
    """

    path = 'include/build_synth/deltaFiles/deltaMinMaxSpatCorr%s.mat' % fileSuffix

    if os.path.exists(path) == 0:

        initSpatReg = np.sum(np.abs(np.dot(B_matrix, REstimates)) ** 2)

        dissMin = 0
        powerMin = firstPower - step

        dissMax = 0
        powerMax = lastPower + step

        minSearch = True
        maxSearch = False

        while minSearch or maxSearch :
            if dissMin < prec:
                powerMin += step
                power = powerMin
            else:
                minSearch = False
                maxSearch = True
                if dissMax < prec:
                    powerMax -= step
                    power = powerMax
                else:
                    maxSearch = False
                    continue
            print("power=", power)
            deltaS = 10 ** power
            print('Computing diffusion with delta = 10 ** (%.1f) ----' % power)
            RDiff = Tikhonov_spat_corr(REstimates, B_matrix, deltaS)
            if minSearch:
                dissMin = np.abs(initSpatReg - np.sum(np.abs(np.dot(B_matrix, RDiff)) ** 2))
            if maxSearch:
                dissMax = np.sum(np.abs(np.dot(B_matrix, RDiff)) ** 2)

        if dissMin > prec:
            powerMin -= step  # border effect because we computed the first deltaS s.t. dissMin > prec
        if dissMax > prec:
            powerMax += step  # border effect because we computed the first deltaS s.t. dissMax > prec
        print("Saving delta max and min in include/deltaFiles/deltaMinMaxSpatCorr%s.mat" % fileSuffix)
        savemat(path, {'deltaSmax': 10 ** powerMax,
                       'deltaSmin': 10 ** powerMin,
                       'powerMax': powerMax,
                       'powerMin': powerMin,
                       'REstimates': REstimates,
                       'B_matrix': B_matrix,
                       'firstPower': firstPower,
                       'lastPower': lastPower,
                       'step': step,
                       'prec': prec})
    else:
        file = loadmat(path, squeeze_me=True)
        powerMin = file['powerMin']
        powerMax = file['powerMax']

    return 10 ** powerMin, 10 ** powerMax, powerMin, powerMax


def deltaGrid(number=500, fileSuffix='Last'):
    """
    Create a range of parameters deltaS used in diffusion computation for multivariate synthetic data generation.
    :param fileSuffix : string (optional)
    :param number : int (optional)
    :return: list of `number` delta that span regularly increasing levels of correlation.
    """
    delta_file = loadmat('include/build_synth/deltaFiles/deltaMinMaxSpatCorr%s.mat' % fileSuffix, squeeze_me=True)

    alphaMin = delta_file['powerMin']
    betaMax = delta_file['powerMax']

    if alphaMin > 0:
        decimals = int(alphaMin)
    else:
        decimals = - int(alphaMin) + 1
    return np.round(np.logspace(alphaMin, betaMax, num=number), decimals)
