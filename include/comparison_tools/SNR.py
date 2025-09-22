import numpy as np


def SignaltoNoiseRatio(groundTruth_init, estimation_init):
    """
    Computes the Signal-to-Noise-Ratio (SNR in dB) which stands for the quadratic error between
    the ground truth and estimation.
    :param groundTruth_init: ndarray of shape (deps, days) -- or (days,)
    :param estimation_init: ndarray of shape (deps, days) -- or (days,)
    :return: SNR(groundTruth_init, estimation)
    """
    normOrder = 2
    if len(np.shape(groundTruth_init)) == 1:
        if len(np.shape(estimation_init)) != 1 or len(groundTruth_init) != len(estimation_init):
            ShapeError = TypeError("Jaccard index cannot be computed between two arrays of different shapes.")
            raise ShapeError
        groundTruth = np.reshape(groundTruth_init, (1, len(groundTruth_init)))
        estimation = np.reshape(estimation_init, (1, len(estimation_init)))
        nbCounties = 1
        days = len(groundTruth_init)
    else:
        nbCounties, days = np.shape(groundTruth_init)
        if nbCounties != np.shape(estimation_init)[0] or days != np.shape(estimation_init)[1]:
            ShapeError = TypeError("Jaccard index cannot be computed between two arrays of different shapes.")
            raise ShapeError
        groundTruth = groundTruth_init
        estimation = estimation_init

    errorMean = (np.sum(np.abs(estimation[:, 1:] - groundTruth[:, 1:])) / (days - 1))
    burn_in_correction = False
    for k in range(nbCounties):
        if (estimation[k, 0] - groundTruth[k, 0]) / errorMean > 100000:
            burn_in_correction = True
            break

    if burn_in_correction:
        SquaredError = np.sum(np.abs(estimation[:, 1:] - groundTruth[:, 1:]) ** normOrder)
    else:
        SquaredError = np.sum(np.abs(estimation - groundTruth) ** normOrder)
    return 10 * np.log10(np.sum(np.abs(groundTruth) ** normOrder) / SquaredError)


def SignaltoNoiseRatioMC(groundTruth, estimations):
    """
    Compute the Signal-to-Noise-Ratio (SNR in dB) for multiple draws in Monte-Carlo simulations, which R estimations are
    gathered in 'estimations'.
    Returns meanSNR, errorSNR such that SNR(estimations) = meanSNR +/- errorSNR.
    :param groundTruth: ndarray of shape (deps, days) --- or (days,)
    :param estimations: ndarray of shape (nbDraws, deps, days) --- or (nbDraws, days)
    :return: meanSNR: float
             errorSNR: float
    """
    nbDraws, days = np.shape(estimations)
    assert (days == len(groundTruth))

    SNREstim = np.zeros(nbDraws)
    for draw in range(nbDraws):
        SNREstim[draw] = SignaltoNoiseRatio(groundTruth, estimations[draw])
    return (1 / nbDraws) * np.sum(SNREstim), (1.96 / np.sqrt(nbDraws)) * np.std(SNREstim)


def SNRByDep_indic(groundTruth, estimation):
    """
    :param estimation: ndarray of shape (nbDeps, days)
    :param groundTruth: ndarray of shape (nbDeps, days)
    :return:
    """
    nbDeps, days = np.shape(estimation)
    assert (nbDeps == np.shape(groundTruth)[0])
    assert (days == np.shape(groundTruth)[1])

    indicators = np.zeros(nbDeps)
    for d in range(nbDeps):
        indicators[d] = SignaltoNoiseRatio(groundTruth[d], estimation[d])
    return indicators  # we should maximize this criteria


def SNRMean_indic(groundTruth, estimation):
    """
    :param estimation: ndarray of shape (nbDeps, days)
    :param groundTruth: ndarray of shape (nbDeps, days)
    :return:
    """
    nbDeps, days = np.shape(estimation)
    assert (nbDeps == np.shape(groundTruth)[0])
    assert (days == np.shape(groundTruth)[1])

    indicatorsByDep = SNRByDep_indic(groundTruth, estimation)
    # assert (extremum == 'minimum')
    assert (len(indicatorsByDep) == nbDeps)
    indicators = 1 / nbDeps * np.sum(indicatorsByDep)

    return indicators  # we should minimize this criteria
