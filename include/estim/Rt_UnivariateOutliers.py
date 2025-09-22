import time
import numpy as np

from include.optim_tools import conversion_pymat as pymat
from include.optim_tools.Rt_Joint_graph import Rt_Jgraph


def Rt_U_O(data, lambdaR=3.5, lambdaO=0.02, options=None):
    """
    Computes the spatial and temporal evolution of the reproduction number R and erroneous counts.
    Can be used for time series.
    The method used is detailed in optim_tools/CP_covid_5_outlier_graph.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm) but with neither spatial regularization nor explicit underlying connectivity structure.
     Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    :param data ndarray of shape (counties, days) or (days, )
    :param lambdaR: regularization parameter for piecewise linearity of Rt
    :param lambdaO: regularization parameters for sparsity of O
    :param options: dictionary containing at least
            - dates ndarray of shape (days, )
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

    print("Computing Univariate estimation with O misreported counts modelisation ...")
    start_time = time.time()
    REstimate, OEstimate, datesUpdated, dataCrop = Rt_Jgraph(dates, dataProc, B_matrix, lambdaR, lambdaO, lambdaS=0)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)

    if len(np.shape(data)) == 1:
        assert (np.shape(REstimate)[0] == 1)
        assert (np.shape(REstimate)[1] == days - 1)
        output = {'dates': datesUpdated,
                  'data': pymat.matvec2pyvec(dataCrop),
                  'method': 'U-O',
                  'OEstim': pymat.matvec2pyvec(OEstimate)}
        return pymat.matvec2pyvec(REstimate), pymat.matvec2pyvec(OEstimate), output

    options_UO = {'dates': datesUpdated,
                  'data': dataCrop,
                  'method': 'U-O',
                  'OEstim': OEstimate}
    return REstimate, OEstimate, options_UO
