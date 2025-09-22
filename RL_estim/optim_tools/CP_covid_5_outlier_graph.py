# from https://github.com/juliana-du/Covid-R-estim/blob/main/include/optim_tools/CP_covid_5_outlier_graph.py
# downloaded in 2025-03

# modification by Etienne Lasalle:
# - modif1, 2025-03: deal with extra output



import numpy as np
from RL_estim.optim_tools import fidelity_terms_DKL as dkl
from RL_estim.optim_tools import prox_Lp
from RL_estim.optim_tools import opL
from RL_estim.optim_tools import opLadj
from RL_estim.optim_tools import Chambolle_pock_pdm as cppdm
from RL_estim.optim_tools import conversion_pymat as pymat


def set_choice(choice):
    # Default choices
    if not (hasattr(choice, "prior")): choice.prior = "laplacian"
    if not (hasattr(choice, "dataterm")): choice.dataterm = "DKL"
    if not (hasattr(choice, "regularization")): choice.regularization = "L1"
    if not (hasattr(choice, "stop")): choice.stop = "LimSup"

    if not (hasattr(choice, "prec")): choice.prec = 10 ** (-6)
    if not (hasattr(choice, "nbInf")): choice.nbInf = 10 ** 7
    if not (hasattr(choice, "iter")): choice.iter = 10 ** 7
    if not (hasattr(choice, "nbiterprint")): choice.nbiterprint = 10 ** 6
    return


def CP_covid_5_outlier_graph(data, lambdaR, lambdaG, lambdaO, alpha, B_matrix, choice):
    """
    :param data: ndarray of shape (dep, days)
    :param lambdaR: float hyperparameter for piecewise linear time regularization
    :param lambdaG: float hyperparameter for space regularization
    :param lambdaO: float hyperparameter for outliers sparsity regularization
    :param alpha: ndarray size y (supposed to be ZPhi)
    :param B_matrix: ndarray of shape (|E|, dep) : operator matrix for the Graph Total Variations where E are the edges
    :param choice: structure (see below)
    :return: (x, crit, gap, op_out)

    CP_covid_5_outlier_graph minimizes the following criterion:
    min_{R, O}  L(data, alpha.*u) + lambdaR * Pen1(R) + lambdaG * Pen3(G) + lambdaO * Pen2(O)
    where L stands either for the Kullback-Leibler divergence or the L2 data term and Pen1(R) stands either for the l1
    norm applied either on discrete gradient for laplacian applied on R, Pen3 stands for the l1 norm of the Total
    Variations Graph operator applied on G and Pen2 stands for the l1 norm applied on O.

    Input: - data: observation
           - lambdaR, lambdaG, lambdaO: regularization parameters
           - choice: a structure to select parameters
                    - dataterm: 'DKL' (by default)  or 'L2'
                    - type: 'usual' (by default) or 'accelerated', the second one is for the strong convex L2
                    - prec: tolerance for the stopping criterion (1e-6 by default)
                    - prior: 'gradient' (by default) or 'laplacian'
                    - regularization: 'L1' (by default) or 'L12'

    Output: - x: solution of the minimization problem
            - crit: values of the objective criterion w.r.t iterations
            - gap: relative difference between the objective criterions of successive iterations
            - op_out: structure containing direct operators for debugging sessions
    """
    dep, days = np.shape(data)
    edges, depG = np.shape(B_matrix)
    assert (depG == dep)
    set_choice(choice)

    if not (hasattr(choice, "x0")):
        choice.x0 = np.array([data, np.zeros((dep, days))])
    else:
        assert (np.shape(choice.x0[0]) == np.shape(data))  # not data shape
        assert (np.shape(choice.x0[1]) == np.shape(data))

    filter_def = choice.prior
    computation = 'direct'

    param = pymat.struct()
    param.tol = choice.prec
    param.iter = choice.iter
    param.stop = choice.stop
    param.nbiterprint = choice.nbiterprint
    param.nbInf = choice.nbInf
    param.x0 = choice.x0

    objective = pymat.struct()
    prox = pymat.struct()

    if choice.dataterm == "DKL":
        param.mu = 0
        cst = np.sum(data[data > 0] * (np.log(data[data > 0]) - 1))
        objective.fidelity = lambda y_, tempdata: dkl.DKLw_outlier(y_, tempdata, alpha) + cst
        prox.fidelity = lambda y_, tempdata, tau: dkl.prox_DKLw_outlier_0cas(y_, tempdata, alpha, tau)

    if choice.regularization == "L1":
        prox.regularization = lambda y_, tau: \
            np.array([prox_Lp.prox_L1(y_[0], tau), prox_Lp.prox_L1(y_[1], tau), prox_Lp.prox_L1(y_[2], tau),
                      np.maximum(y_[3], np.zeros(np.shape(y_[3])))], dtype=object)
        objective.regularization = lambda y_, tau: tau * np.sum(np.abs(np.concatenate((y_[0], y_[1], y_[2]))))

    # if choice.regularization == "L12":
    #   prox.regularization = lambda : y_, tau: L1.prox_L12(y, tau)
    #   objective.regularization = lambda : y_, tau: tau * np.sum(np.sqrt(np.sum(y ** 2, 1)))

    paramL = pymat.struct()
    paramL.lambd = lambdaR
    paramL.type = '1D'
    paramL.op = choice.prior

    op = pymat.struct()

    def direct_covid_5_outlier_0cas_graph(estimates):
        R = estimates[0]
        outliers = estimates[1]
        return np.array([opL.opL(R, paramL, filter_def, computation), lambdaG * np.dot(B_matrix, R),
                         lambdaO * outliers, R],
                        dtype=object)

    op.direct = direct_covid_5_outlier_0cas_graph

    def adjoint_covid_5_outlier_0cas_graph(opEstimates):
        laplacianR = opEstimates[0]  # named according to the expected shape of these arrays
        GTVR = opEstimates[1]
        outliers = opEstimates[2]
        R = opEstimates[3]
        depR, days = np.shape(R)
        assert (depR == dep)
        res = np.zeros((2, dep, days))
        res[0] = opLadj.opLadj(laplacianR, paramL, filter_def, computation)\
                 + lambdaG * np.dot(np.transpose(B_matrix), GTVR) + R
        res[1] = lambdaO * outliers
        return res

    op.adjoint = adjoint_covid_5_outlier_0cas_graph

    # operator norm
    param.normL = max(lambdaR ** 2 + lambdaG ** 2 * np.linalg.norm(B_matrix, ord=2) ** 2 + 1, lambdaO ** 2)

    x, _, crit, gap = cppdm.PD_ChambollePock_primal_BP(data, param, op, prox, objective) # modif1

    op_out = pymat.struct()
    paramL.lambd = 1
    op_out.direct = direct_covid_5_outlier_0cas_graph
    op_out.adjoint = adjoint_covid_5_outlier_0cas_graph
    return x, crit, gap, op_out
