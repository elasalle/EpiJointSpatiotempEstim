# from https://github.com/juliana-du/Covid-R-estim/blob/main/include/optim_tools/Rt_PL_graph.py
# downloaded in 2025-03

# modifications by Etienne Lasalle:
# - modif1, 2025-03: add the choice of the spatial regularization type: L1 or L2
# - modif2, 2025-03: add the initialization of the R variable
# - modif3; 2025-04: add the initializatino of the dual variable of R
# - modif4; 2025-04: change the renormalization of Z and ZPhi using parameters omegas
# - modif5; 2025-04: change the output format. Return a dictionnary containing variables on interest



import numpy as np

from RL_estim.optim_tools import conversion_pymat as pymat
from RL_estim.optim_tools import crafting_phi

from RL_estim.optim_tools import CP_covid_4_graph as cp4g


def Rt_PL_graph(dates, data, B_matrix, muR=50, muS=0.005, Gregularization="L1", omegas=None, Rinit=None, dualRinit=None): # modif1, modif2, modif3, modif4
    """
    Computes the evolution of the reproduction number R for counties on a graph.
    The method used is detailed in optim_tools/CP_covid_4_graph.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm). Hyperparameters choice has to be as followed :
    - mu R sets piecewise linearity of Rt
    - mu S sets spatial regularity of Rt on the chosen graph which transposed incidence matrix is 'B_matrix'.
    :param dates : ndarray of shape (days, )
    :param data : ndarray of shape (counties, days)
    :param B_matrix: ndarray of shape (|E|, counties) : operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matrix
    :param muR: regularization parameter for piecewise linearity of Rt
    :param muS: regularization parameters for spatial coherence
    :return: results : dictionnary with keys: "R", "Rdual", "objs", "gaps", "op_out"
    """
    # Gamma pdf
    Phi = crafting_phi.buildPhi()
    # print("dtype of phi:", Phi.dtype)

    data[data < 0] = 0

    # Normalize each counts for each vertex
    # modif4
    if omegas is None:
        omegas = 1/np.std(data[:, 1:], axis=1) # first time point is removed from the computation of std
    counties, days = np.shape(data)
    ZDataDep = np.zeros((counties, days - 1))
    ZDataNorm = np.zeros((counties, days - 1))
    ZPhiNorm = np.zeros((counties, days - 1))
    datesUpdated = dates[1:]
    for d in range(counties):
        tmpDates, ZDataDep[d], ZPhiDep = crafting_phi.buildZPhi(dates, data[d], Phi)
        # Asserting dates are cropped from first day
        assert (len(tmpDates) == len(datesUpdated))  # == days - 1
        for i in range(days - 1):
            assert (tmpDates[i] == datesUpdated[i])
        # Normalizing for each 'département'
        ZDataNorm[d] = omegas[d] * ZDataDep[d] # modif4
        ZPhiNorm[d] = omegas[d] * ZPhiDep # modif4

    # Run CP covid
    choice = pymat.struct()
    choice.prior = 'laplacian'  # or 'gradient'
    choice.dataterm = 'DKL'  # or 'L2'
    choice.prec = 10 ** (-7)
    choice.nbiterprint = 10 ** 5
    choice.iter = 7 * choice.nbiterprint
    choice.incr = 'R'
    choice.regularization = Gregularization # modif1
    # modif2
    if Rinit is not None:
        choice.x0 = Rinit
    # modif3
    if dualRinit is not None:
        choice.y0 = dualRinit
    
    REstimate, dualR, crit, gap, op_out = cp4g.CP_covid_4_graph(ZDataNorm, muR, muS, ZPhiNorm, B_matrix, choice)
    
    # modif5
    results = {
        "R" : REstimate,
        "Rdual" : dualR,
        "objs" : crit,
        "gaps" : gap,
        "op_out" : op_out
    }
    return results

