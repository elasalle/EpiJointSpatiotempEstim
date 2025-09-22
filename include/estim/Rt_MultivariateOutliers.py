import time
from include.optim_tools.Rt_Joint_graph import Rt_Jgraph


def Rt_M_O(data, lambdaR=3.5, lambdaO=0.02, lambdaS=0.005, options=None):
    """
    Computes the evolution of the reproduction number R for the indicated country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_5_outlier_graph.py
    (regularized optimization scheme solved using Chambolle-Pock algorithm).
    Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    - lambda S sets total variations regularity on the chosen graph 'G'
    :param data: ndarray of shape (counties, days)x
    :param lambdaR: regularization parameter for piecewise linearity of Rt
    :param lambdaO: regularization parameters for sparsity of O
    :param lambdaS: regularization parameters for spatial coherence
    :param options: dictionary containing
    - 'dates': list of str of length (days, )
    - 'B_matrix': ndarray of shape (|E|, counties) : operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matri
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             OEstimate: ndarray of shape (counties, days - 1), daily estimation of Outliers
             options: dictionary containing at least:
             - dates: ndarray of shape (counties, days -1) representing dates
             - data: ndarray of shape (counties, days - 1) representing processed data
    """
    dates = options['dates']
    B_matrix = options['B_matrix']
    print("Computing Univariate estimator with misreported counts modelisation ...")
    start_time = time.time()
    REstimate, OEstimate, datesUpdated, ZDataDep = Rt_Jgraph(dates, data, B_matrix, lambdaR, lambdaO, lambdaS)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)

    options_MO = {'dates': datesUpdated,
                  'data': ZDataDep,
                  'method': 'M-O',
                  'OEstim': OEstimate}

    return REstimate, OEstimate, options_MO
