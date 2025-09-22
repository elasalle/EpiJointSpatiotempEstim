# created by Etienne Lasalle in 2025-03



import numpy as np
from cvxopt.solvers import qp, options
from cvxopt import matrix
from time import time

options['show_progress'] = False

def get_operators_for_vectorization(n):

    indices = np.triu_indices(n)
    convert_indices_to_position = {} # a dict to get the index in the half vectorization of the (i,j) coeff (i<=j)
    offdiagOp = np.zeros((n*(n-1)//2 , n*(n+1)//2)) # the operator extracting the offdiagonal coefficient of the half-vect
    traceOp = np.zeros((1 , n*(n+1)//2)) # the operator that compute the trace from the half-vect
    u = 0
    for i in range(n*(n+1)//2):
        k,l = indices[0][i], indices[1][i]
        convert_indices_to_position[(k,l)] = i
        if k != l:
            offdiagOp[u,i]=1
            u += 1
        else:
            traceOp[0,i]=1

    # duplication operator
    dupliOp = np.zeros((n**2, n*(n+1)//2)) # the operator that compute the full vectorization from the half vectorization
    rowsumOp = np.zeros((n , n*(n+1)//2)) # the operator that the sum of each row from the half-vect

    for i in range(n):
        for j in range(n):
            if i<=j:
                dupliOp [i*n+j, convert_indices_to_position[(i,j)]] = 1
                rowsumOp[i    , convert_indices_to_position[(i,j)]] = 1
            else:
                dupliOp [i*n+j, convert_indices_to_position[(j,i)]] = 1
                rowsumOp[i    , convert_indices_to_position[(j,i)]] = 1

    #offdiagonal_operator

    return dupliOp, offdiagOp, traceOp, rowsumOp



def learningL(lambda_G, lambda_L, R, verbose=False, return_crit=False, inits = None):
    """This function solve the otpimization problem to learn the Laplacian matrix describing spatial relationship from spatio-temporal signal. It solves:
    Argmin_L lambda_G sum_t R_t^T L R_t + lambda_L ||L||_Fro^2 ; subject to Tr(L)=n (L is an nxn matrix), L_ij = L_ji <= 0, and L.1 = 0.
    The optimization problem is cast as a Quadratic Programming problem.

    Args:
        lambda_G (float): Coefficient in front of the signal-related term
        lambda_L (float): Coefficient in front of the Frobenius norm regularization term
        R (array of shape (n,m)): The spatio-temporal signal. (n number of space points, m number of time points). 

    Return:
        L : (n,n)-array, argmin of the above optimization problem.
    """
    
    ti = time()
    n = R.shape[0]
    Mdupli, offdiagOp, traceOp, rowsumOp = get_operators_for_vectorization(n) 
    
    #variable to set for the qp problem
    P = 2 *lambda_L * Mdupli.T @ Mdupli 
    q = lambda_G * Mdupli.T @ (R @ R.T).flatten()[:,None]
    G = offdiagOp
    h = np.zeros((n*(n-1)//2,1))
    A = np.vstack([traceOp, rowsumOp])
    b = np.zeros((n+1,1))
    b[0,0] = n
    P,q,G,h,A,b = matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b)
    tf = time()
    if verbose:
        print("Precomputations done in {:5.3f}".format(tf-ti))

    # potential initialization of the qp variables
    if isinstance(inits, dict):
        initvals = {key:inits[key] for key in ["x", "s", "y", "z"] if key in inits} # keep only a subdict containing "x", "s", "y", "z" keys
        if "x" not in initvals and "L" in inits:      
            # deals with the case where x is not given but L is.
            Linit = inits["L"]
            x0 = matrix(Linit[np.triu_indices((n))][:, None]) # convert to the x-format for cvx
            initvals["x"] = x0
    else:
        initvals = {}

    ti = time()
    res = qp(P,q,G,h,A,b,initvals=initvals)
    # if Linit is not None:
    #     res = qp(P,q,G,h,A,b,initvals=initvals)
    # else:
    #     res = qp(P,q,G,h,A,b)
    tf = time()

    crit = res['primal objective']
    halfvectL = np.array(res['x'])
    if verbose:
        print("QP solution computed in {}".format(tf-ti))
    L = np.zeros((n,n))
    L[np.triu_indices(n)] = halfvectL[:,0]
    L += L.T
    L[range(n), range(n)] /= 2

    results = {
        "L" : L,
        "objs" : crit,
        "x" : res["x"],
        "s" : res["s"],
        "y" : res["y"],
        "z" : res["z"]
    }

    return results