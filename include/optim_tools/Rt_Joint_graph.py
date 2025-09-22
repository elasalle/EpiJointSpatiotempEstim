import numpy as np

from include.optim_tools import conversion_pymat as pymat
from include.optim_tools import crafting_phi

from include.optim_tools import CP_covid_5_outlier_graph as cp5g


def Rt_Jgraph(dates, data, B_matrix=np.ones((1, 1)), lambdaR=3.5, lambdaO=0.02, lambdaS=0.005):
    """
    Computes the evolution of the reproduction number R for the indicated country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_5_outlier_graph.py
    (regularized optimization scheme solved using Chambolle-Pock algorithm).
    Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    - lambda S sets total variations regularity on the chosen graph 'G'
    :param dates: list of str of length (days, )
    :param data: ndarray of shape (counties, days)
    :param B_matrix: ndarray of shape (|E|, counties) : operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matrix
    :param lambdaR: regularization parameter for piecewise linearity of Rt
    :param lambdaO: regularization parameters for sparsity of O
    :param lambdaS: regularization parameters for spatial coherence
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             OEstimate: ndarray of shape (counties, days - 1), daily estimation of Outliers
             timestamps: ndarray of shape (counties, days -1) representing dates
             ZDataDep: ndarray of shape (counties, days - 1) representing processed data
    """
    # Gamma pdf
    Phi = crafting_phi.buildPhi()
    
    edges, depG = np.shape(B_matrix)
    
    # Normalize each counts for each vertex
    counties, days = np.shape(data)
    assert (counties == depG)
    ZDataDep = np.zeros((counties, days - 1))
    ZPhiDep = np.zeros((counties, days - 1))
    ZDataNorm = np.zeros((counties, days - 1))
    ZPhiNorm = np.zeros((counties, days - 1))
    datesUpdated = dates[1:]
    for d in range(counties):
        ZDataProc = data[d]
        ZDataProc[ZDataProc < 0] = 0
        tmpDates, ZDataDep[d], ZPhiDep[d] = crafting_phi.buildZPhi(dates, ZDataProc, Phi)
        # Asserting dates are cropped from first day
        assert (len(tmpDates) == len(datesUpdated))  # == days - 1
        for i in range(days - 1):
            assert (tmpDates[i] == datesUpdated[i])
        # Normalizing for each 'département'
        ZDataNorm[d] = ZDataDep[d] / np.std(ZDataDep[d])
        ZPhiNorm[d] = ZPhiDep[d] / np.std(ZDataDep[d])

    # # Run CP covid
    choice = pymat.struct()
    choice.prior = 'laplacian'  # or 'gradient'
    choice.dataterm = 'DKL'  # or 'L2'
    choice.nbiterprint = 10 ** 5
    choice.iter = 7 * choice.nbiterprint
    choice.nbInf = 7 * choice.nbiterprint
    choice.prec = 10**(-6)
    choice.incr = 'R'

    xx, crit2, gap, opout = cp5g.CP_covid_5_outlier_graph(ZDataNorm, lambdaR, lambdaS, lambdaO, ZPhiNorm,
                                                          B_matrix, choice)
    REstimate = xx[0]
    OEstimate = np.zeros(np.shape(xx[1]))
    for d in range(depG):
        OEstimate[d] = xx[1][d] * np.std(ZDataDep[d])

    return REstimate, OEstimate, datesUpdated, ZDataDep

