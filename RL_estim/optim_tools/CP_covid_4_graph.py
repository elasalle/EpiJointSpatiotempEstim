# from https://github.com/juliana-du/Covid-R-estim/blob/main/include/optim_tools/CP_covid_4_graph.py
# downloaded in 2025-03

# modifications by Etienne Lasalle:
# - modif1, 2025-03: add L2-type regularization
# - modif2, 2025-03: rewrite for clarity
# - modif3, 2025-04: initialize and return the dual variable y



import numpy as np
from RL_estim.optim_tools import opL
from RL_estim.optim_tools import conversion_pymat as mat2py
from RL_estim.optim_tools import Chambolle_pock_pdm as cppdm
from RL_estim.optim_tools import opLadj
from RL_estim.optim_tools import fidelity_terms_DKL as dkl
from RL_estim.optim_tools import prox_Lp


def set_choice(choice):
    # Default choices
    if not (hasattr(choice, "prior")): choice.prior = "laplacian"
    if not (hasattr(choice, "dataterm")): choice.dataterm = "DKL"
    if not (hasattr(choice, "regularization")): choice.regularization = "L1"
    if not (hasattr(choice, "stop")): choice.stop = "LimSup"
    if not (hasattr(choice, "incr")): choice.incr = 'R'

    if not (hasattr(choice, "prec")): choice.prec = 10 ** (-7)
    if not (hasattr(choice, "nbiterprint")): choice.nbiterprint = 10 ** 6
    if not (hasattr(choice, "iter")): choice.iter = 10 ** 7
    if not (hasattr(choice, "nbInf")): choice.nbInf = 10 ** 7

    return


def CP_covid_4_graph(data, muR, muS, alpha, B_matrix, choice):
    """
    :param data: ndarray of shape (dep,  days) in MATLAB format
    :param muR: float : time regularization parameter on R (rather discrete gradient of R)
    :param muS: float : spatial regularization parameter on R (rather Total Variation for R on graph G)
    :param alpha: ndarray of shape (dep, days)  infectiousness convoluted with the data
    :param B_matrix: ndarray of shape (|E|, dep) : operator matrix for the Graph Total Variations where E are the edges
    :param choice: structure (see below)
    :return: (x, crit, gap, op_out)

    CP_covid_4_graph minimizes the following criterion:
    min_u  L(data, alpha.* u) + muR * R(u) + muS * G(u)
    where L stands either for the Kullback-Leibler divergence or the L2 data term and R(u) stands for the l1 norm
    applied on discrete laplacian, G(u) stands for the l1 norm applied on the Total Variation operator
    which corresponds to the matrix-vector product with the transposed incidence matrix (G).

    Input:  - data: observation
            - muR: regularization parameter
            - alpha: ndarray of shape (days,)
            - choice: a structure to select parameters
            - dataterm: 'DKL' (by default)  or 'L2'
            - type: 'accelerated' for the strong convex L2
            - prec: tolerance for the stopping criterion (1e-6 by default)
            - prior: 'laplacian' by default
            - regularization: 'L1' (by default) or 'L12'

    Output: - x: solution of the minimization problem
            - crit: values of the objective criterion w.r.t iterations
            - gap: relative difference between the objective criterion of successive iterations
            - op_out: structure containing direct operators for debugging sessions
    """
    dep, days = np.shape(data)
    edges, depG = np.shape(B_matrix)
    assert(depG == dep)

    set_choice(choice)

    if not (hasattr(choice, "x0")):
        choice.x0 = data
    else:
        assert (np.shape(choice.x0) == np.shape(data))

    if not (hasattr(choice, "y0")): # modif3
        choice.y0 = None

    filter_def = choice.prior
    computation = 'direct'

    param = mat2py.struct()
    param.sigma = 1
    param.tol = choice.prec
    param.iter = choice.iter
    param.stop = choice.stop
    param.nbiterprint = choice.nbiterprint
    param.nbInf = choice.nbInf
    param.x0 = choice.x0
    param.y0 = choice.y0 # modif3
    param.incr = choice.incr
    param.noOutlier = True

    objective = mat2py.struct()
    prox = mat2py.struct()

    if choice.dataterm == "DKL":
        cst = np.sum(data[data > 0] * (np.log(data[data > 0]) - 1))  # WIP
        param.mu = 0
        objective.fidelity = lambda y_, tempData: dkl.DKL_no_outlier(y_, tempData, alpha) + cst
        prox.fidelity = lambda y_, tempData, tau: dkl.prox_DKL_no_outlier(y_, tempData, alpha, tau)

    if choice.regularization == "L1":
        prox.regularization = lambda y_, tau: np.array([prox_Lp.prox_L1(y_[0], tau), prox_Lp.prox_L1(y_[1], tau)], dtype=object)
        objective.regularization = lambda y_, tau: tau * (np.sum(np.abs(y_[0])) + np.sum(np.abs(y_[1]))) # modif2
    elif choice.regularization == "L2": # modif1
        prox.regularization = lambda y_, tau: np.array([prox_Lp.prox_L1(y_[0], tau), prox_Lp.prox_L2(y_[1], tau)], dtype=object)
        objective.regularization = lambda y_, tau: tau * (np.sum(np.abs(y_[0])) + 1/2*np.sum(np.square(y_[1])) )
        muS = np.sqrt(2*muS)


    paramL = mat2py.struct()
    paramL.lambd = muR
    paramL.type = '1D'
    paramL.op = choice.prior

    op = mat2py.struct()

    def direct_covid_4_graph(R):
        return np.array([opL.opL(R, paramL), muS * np.dot(B_matrix, R)], dtype=object)

    op.direct = direct_covid_4_graph

    def adjoint_covid_4_graph(opEstimates):
        laplacianR = opEstimates[0]  # named according to the expected shape of these arrays
        GTVR = opEstimates[1]
        # depR, days = np.shape(laplacianR)
        # assert (depR == dep)
        # res = np.zeros((dep, days))
        res = opLadj.opLadj(laplacianR, paramL, filter_def, computation) + muS * np.dot(np.transpose(B_matrix), GTVR)
        return res

    op.adjoint = adjoint_covid_4_graph

    param.normL = muR ** 2 + (muS * np.linalg.norm(B_matrix, ord=2)) ** 2  # operator norm
    x, y, crit, gap = cppdm.PD_ChambollePock_primal_BP(data, param, op, prox, objective) # modif3

    # For debugging sessions:
    op_out = mat2py.struct()
    paramL.lambd = 1
    op_out.direct = lambda x_: np.array([opL.opL(x_, paramL), np.dot(B_matrix, x_)], dtype=object)
    op_out.adjoint = lambda x_: opLadj.opLadj(x_[0], paramL, filter_def, computation) + \
                                muS * np.dot(np.transpose(B_matrix), x_[1])

    return x, y, crit, gap, op_out # modif3
