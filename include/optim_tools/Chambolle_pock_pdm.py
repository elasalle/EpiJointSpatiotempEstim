import numpy as np
from include.optim_tools import conversion_pymat as pymat


def PD_ChambollePock_primal_BP(data, param, op, prox, objective):
    """
    :param data: ndarray of shape (1, days)
    :param param: structure with options
    :param op: structure with operators (lambda functions)
    :param prox: structure with prox operators (lambda functions)
    :param objective: structure with convergence tools (lambda functions)
    :return: x = [R, O] estimates,
             obj : ndarray of shape (iterations, ) objective function evolution w.r.t. iterations,
             gap : ndarray of shape (iterations, ) increments on the objective function w.r.t. iterations
             gapR : ndarray of shape (iterations, ) increments on normalized R estimates w.r.t. iterations
             RinftyQuadErr : ndarray of shape (iterations, ) normalized distance to the NOPREC solution
             (maximum : 7 * 10 ** 5 iterations)

    Primal-dual algorithm by Chambolle and Pock handling strong convexity when possible.
    see Chambolle A., Pock T. : A first-order primal-dual algorithm for convex problems
    with applications to imaging J. Math. Imag. Vis. 40(1), 120 145 (2011)
    Translation of Matlab's code from N. PUSTELNIK, CNRS, ENS Lyon April 2020
    """

    # Default parameters
    if not hasattr(param, 'stop'):
        param.stop = 'LimSup'
    if not hasattr(param, 'stopwin'):
        param.stopwin = 500
    if not hasattr(param, 'incr'):
        param.incr = 'R'
    if not hasattr(param, "y0"):
        param.y0 = None

    # Proximal parameters
    gamma = 0.99
    tau = gamma / np.sqrt(param.normL)
    sig = gamma / np.sqrt(param.normL)
    assert (tau * sig * param.normL < 1)
    theta = 1

    # Initializing variables
    x = param.x0  # x = [R, O] estimates
    if param.y0 is None:
        y = op.direct(x)  # dual variable of x via L operator
    else:
        y = param.y0
    x0 = np.copy(x)  # dual auxiliary variable
    bx = np.copy(x)  # dual auxiliary variable

    # Criterion of convergence
    obj = np.zeros(param.iter)
    realIncr = np.zeros(param.iter)  # intermediate computation of increments
    gap = np.zeros(param.iter)

    if param.incr == 'R':
        if not (hasattr(param, "noOutlier")):
            previousR = np.zeros(np.shape(x0[0]))
        else:
            previousR = np.zeros(np.shape(x0))
    else:
        previousR = None

    stopCondition = np.copy(param.tol) + 1
    objInit = objective.fidelity(x, data) + objective.regularization(op.direct(x), 1)
    objCurrent = objInit+1

    # Main loop
    i = -1
    while (stopCondition > param.tol and i < param.iter - 1) or objCurrent>objInit:
        i += 1
        # Update of primal variable
        tmp = y + sig * op.direct(bx)
        y = tmp - sig * prox.regularization(tmp / sig, 1 / sig)  # Matlab's version

        # Update of the dual variable
        x = prox.fidelity(x0 - tau * op.adjoint(y), data, tau)  # fidelity == KLD
        # # ---------------------------------------------------------------------------------------------
        # # If data is already used in objective.fidelity lambda function
        # x = prox.fidelity(x0 - tau * op.adjoint(y), tau)  # fidelity == KLD
        # # ---------------------------------------------------------------------------------------------

        # Update of the descent steps
        if param.mu >= 0:
            theta = (1 + 2 * param.mu * tau) ** (-0.5)
            tau = theta * tau
            sig = sig / theta


        # Update of the dual auxiliary variable
        bx = x + theta * (x - x0)
        x0 = x
        # Computing the objective function
        obj[i] = objective.fidelity(x, data) + objective.regularization(op.direct(x), 1)
        objCurrent = obj[i]
        # # ---------------------------------------------------------------------------------------------
        # # If data is already used in objective.fidelity lambda function
        # obj[i] = objective.fidelity(x) + objective.regularization(op.direct(x), 1)
        # # ---------------------------------------------------------------------------------------------
        # Computing the stopping criteria
        if i == 0:
            if not (hasattr(param, "noOutlier")):
                previousR = x[0]  # to prepare new increments
            else:
                previousR = x
        if i > 0:
            # Stop criterion on objective function increments
            if param.incr == 'obj':
                previousObj = obj[i - 1]
                realIncr[i - 1] = np.abs(obj[i] - previousObj) / np.abs(previousObj)
                if param.stop == 'primal':
                    gap[i - 1] = realIncr[i - 1]
                elif param.stop == 'LimSup':
                    ind_past = max(0, i - param.stopwin)
                    gap[i - 1] = max(realIncr[ind_past:i])
                    

            # Stop criterion on Rt estimates increments
            if param.incr == 'R':
                if not (hasattr(param, "noOutlier")):
                    newR = x[0]
                else:
                    newR = x
                nonNegPrevR = previousR[previousR > 0]
                realIncr[i - 1] = np.max(np.abs(newR[previousR > 0] - nonNegPrevR)
                                         / np.maximum(10 ** (-2) * np.ones(np.shape(nonNegPrevR)), nonNegPrevR))
                # Normalization when prevR is not too close to 0. Care to the use of np.maximum =/= np.max in Python
                if param.stop == 'primal':
                    gap[i - 1] = realIncr[i - 1]
                elif param.stop == 'LimSup':
                    ind_past = max(0, i - param.stopwin)
                    gap[i - 1] = max(realIncr[ind_past:i])
                previousR = newR
            

            stopCondition = gap[i - 1]

        if stopCondition == np.nan:
            stopCondition = np.inf
        if (i % param.nbiterprint == 0) and (i != 0):
            print("iter %f \t crit=%f \n" % (i, obj[i]))  # print the current nb of iterations and objective function

    obj = obj[:i+1]
    gap = gap[:i]

    return x, y, obj, gap