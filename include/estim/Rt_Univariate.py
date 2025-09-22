import numpy as np
import time
from include.optim_tools import conversion_pymat as pymat
from include.optim_tools.Rt_PL_graph import Rt_PL_graph


def Rt_U(data, muR=50, options=None):
    """
    Computes the spatial and temporal evolution of the reproduction number R.
    The method used is detailed in optim_tools/CP_covid_4_graph.py (time regularized optimization scheme solved using
    Chambolle-Pock algorithm), with muS = 0 and no underlying connectivity structure. Can be used for time series.
    (optional) One can choose the regularization parameter muR that sets the penalization for piecewise linearity of Rt
    :param data ndarray of shape (counties, days) or (days,)
    :param options: dictionary containing at least
            - dates ndarray of shape (days, )
    :param muR: regularization parameter for piecewise linearity of Rt
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             options: dictionary containing at least:
             - dates: ndarray of shape (counties, days -1) representing dates
             - data: ndarray of shape (counties, days - 1) representing processed data
    """
    dates = options['dates']
    if len(np.shape(data)) == 1:
        days = len(data)
        counties = 1
        dataProc = pymat.pyvec2matvec(data)
    elif len(np.shape(data)) == 2:
        counties, days = np.shape(data)
        dataProc = data
    else:
        ShapeError = TypeError("data should be of shape (days,) or (counties, days) ")
        raise ShapeError
    assert (days == len(dates))

    B_matrix = np.zeros((2, counties))
    print("Computing Univariate estimator ...")
    start_time = time.time()
    REstimate, datesUpdated, dataCrop = Rt_PL_graph(dates, dataProc, B_matrix, muR, muS=0)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)
    if len(np.shape(data)) == 1:
        assert (np.shape(REstimate)[0] == 1)
        assert (np.shape(REstimate)[1] == days - 1)
        options_U = {'dates': datesUpdated,
                     'data': options['data'][1:],
                     'OEstim': options['data'][1:] - pymat.matvec2pyvec(dataCrop),
                     'method': 'U'}
        return pymat.matvec2pyvec(REstimate), options_U

    options_U = {'dates': datesUpdated,
                 'data': options['data'][:, 1:],
                 'method': 'U'}
    return REstimate, options_U


def myRt_U(data, muR=50, options=None, verbose=False):
    """
    Computes the spatial and temporal evolution of the reproduction number R.
    The method used is detailed in optim_tools/CP_covid_4_graph.py (time regularized optimization scheme solved using
    Chambolle-Pock algorithm), with muS = 0 and no underlying connectivity structure. Can be used for time series.
    (optional) One can choose the regularization parameter muR that sets the penalization for piecewise linearity of Rt
    :param data ndarray of shape (counties, days) or (days,)
    :param options: dictionary containing at least
            - dates ndarray of shape (days, )
    :param muR: regularization parameter for piecewise linearity of Rt
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             options: dictionary containing at least:
             - dates: ndarray of shape (counties, days -1) representing dates
             - data: ndarray of shape (counties, days - 1) representing processed data
    """
    dates = options['dates']
    if len(np.shape(data)) == 1:
        days = len(data)
        counties = 1
        dataProc = pymat.pyvec2matvec(data)
    elif len(np.shape(data)) == 2:
        counties, days = np.shape(data)
        dataProc = data
    else:
        ShapeError = TypeError("data should be of shape (days,) or (counties, days) ")
        raise ShapeError
    assert (days == len(dates))

    B_matrix = np.zeros((2, counties))
    if verbose:
        print("Computing Univariate estimator ...")
    start_time = time.time()
    REstimate = Rt_PL_graph(dates, dataProc, B_matrix, muR, muS=0)["R"]
    executionTime = time.time() - start_time
    if verbose:
        print("Done in %.4f seconds ---" % executionTime)
    
    return REstimate
