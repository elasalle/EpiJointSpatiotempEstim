# from https://github.com/juliana-du/Covid-R-estim/blob/main/include/estim/Rt_Multivariate.py
# downloaded in 2025-03



import time
import numpy as np
from RL_estim.optim_tools.Rt_PL_graph import Rt_PL_graph

# This function is inspired by the Rt_M function in https://github.com/juliana-du/Covid-R-estim/blob/main/include/estim/Rt_Multivariate.py
# the difference here are:
# - we don't use some incidence matrix B_matrix. We rather compute a square root of the Laplacian matrix L using the Cholesky decomposition. 
# - omegas parameters have been added to allow different renormalization of Z and ZPhi.
# - the primal and dual variables R and dualR can be initialized. 
def Rt_with_laplacianReg(data, L,  lambda_T=50, lambda_S=0.005, Gregularization="L2", dates=None, verbose=False, omegas=None, Rinit = None, dualRinit=None):
    """_summary_

    Args:
        data (_type_): _description_
        L (_type_): _description_
        muR (int, optional): _description_. Defaults to 50.
        muS (float, optional): _description_. Defaults to 0.005.
        Gregularization (str, optional): _description_. Defaults to "L2".
        dates (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
        Rinit (_type_, optional): _description_. Defaults to None.
        dualRinit (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: result dictionnary with keys: "R", "Rdual", "objs", "gaps", "op_out"
    """

    if verbose:
        print("Computing a Cholesky decomposition of L ...")
    ti = time.time()
    try:
        S = np.linalg.cholesky(L + (1e-12)*np.eye(L.shape[0]))
    except:
        minegval = np.linalg.eigvalsh(L)[0]
        print("minimal eignvalue of L was {} \n Warning, we are forcing a regularization of L to get a positive matrix.".format(minegval))
        S = np.linalg.cholesky(L - (1.01*minegval)*np.eye(L.shape[0]))
    tf = time.time()
    if verbose:
        print("Done in {:4.3f} seconds ---".format(tf-ti))
    
    ti = time.time()
    res = Rt_PL_graph(dates, data, S.T, lambda_T, lambda_S, Gregularization, omegas, Rinit, dualRinit)
    tf = time.time() 
    if verbose:
        print("Done in {:4.3f} seconds ---".format(tf-ti))

    return res
