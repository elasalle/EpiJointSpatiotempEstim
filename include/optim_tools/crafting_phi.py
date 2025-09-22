import numpy as np
import scipy.stats as spst


def buildPhi(beta=1.87, alpha=1 / 0.28, nbDays=26):
    """
    Build the gamma pdf to be convoluted with another 26 daily new infections' data.
    By default, we use parameters defined in date_choice.py.
    :return: Phi : ndarray of shape (26,)
    """
    # beta = 1.87
    # alpha = 1 / 0.28
    Phi = spst.gamma.pdf(range(0, nbDays), alpha, scale=beta)
    return Phi


def buildZPhi(timestamps, ZData, Phi):
    """
    Given the timestamps, data (ZData) and a distribution Phi over len(Phi) days, computes ZPhi which is the data
    convoluted with distribution (Phi) normalized both for the first len(Phi) days, and any other days.
    :param timestamps: array of shape (days,)
    :param ZData: array of shape (days,)
    :param Phi: array of shape (len(Phi), ) : pdf of some distribution accounting for the infectiousness of the disease,
    over time during len(Phi) days.
    :return: timestamps : array of shape (days -1,) timestamps cropped of the first day
             ZDataCropped : array of shape (days - 1,) ZData cropped of the first day
             ZPhi : array of shape (days - 1,) data on which we apply a "normalized convolution" with Phi
    """
    days = len(ZData)
    assert(days > 1)  # if there's only one day of data, can not compute ZPhi
    PhiNormalized = Phi / np.sum(Phi)
    ZPhi = np.convolve(ZData, PhiNormalized)
    ZPhi = ZPhi[:days]

    # Modified convolution : normalized convolution for the first tauPhi days.

    tauPhi = len(Phi) - 1  # wrong explanation in associated papers (tauPhi = len(Phi) - 1 = 25)
    # endPhi = min(days, tauPhi)  # mostly tauPhi, unless we compute less than 25 days of data

    ZPhiNormalized = np.copy(ZPhi)
    ZPhiNormalized[:tauPhi] = 0
    for T in range(1, tauPhi):  # at day 0, the convolution stays at 0.
        PhiNormalizedIterT = Phi[:T+1] / np.sum(Phi[:T+1])  # careful to use non-normalized Phi here !!
        fZ = np.flip(ZData[:T+1])  # first value of Phi is always 0 : we do not need ZData[T] to compute ZPhi[T]
        ZPhiNormalized[T] = np.sum(fZ * PhiNormalizedIterT)

    # Crop ZPhi and ZData : first day of computing R is irrelevant since we only have one sample.
    if timestamps is not None:
        timestamps = timestamps[1:]
    ZPhiNormalized = ZPhiNormalized[1:]
    ZDataCropped = ZData[1:]
    return timestamps, ZDataCropped, ZPhiNormalized

