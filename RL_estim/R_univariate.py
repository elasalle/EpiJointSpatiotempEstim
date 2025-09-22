# from https://github.com/juliana-du/Covid-R-estim/blob/main/include/estim/Rt_Univariate.py
# downloaded in 2025-03

# modifications by Etienne Lasalle:
# - modif1, 2025-03: cleaning, not dealing with the option at the end
# - modif2, 2025-03: add verbose parameter
# - modif3, 2025-04: adapt to the new output format of Rt_PL_graph
# - modif4, 2025-04: change muR to lambda_T for consistency
# - modif5, 2025-05: add omega parameter to rescale the data



import numpy as np
import time
from RL_estim.optim_tools import conversion_pymat as pymat
from RL_estim.optim_tools.Rt_PL_graph import Rt_PL_graph


def Rt_U(data, lambda_T=50, options=None, verbose=False, omegas=None): # modif2, modif4, modif5
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
    # modif2
    if verbose:
        print("Computing Univariate estimator ...")
    start_time = time.time()
    REstimate = Rt_PL_graph(dates, dataProc, B_matrix, lambda_T, muS=0, omegas=omegas)["R"] # modif3, modif5
    executionTime = time.time() - start_time
    # modif2
    if verbose:
        print("Done in %.4f seconds ---" % executionTime)

    # modif1
    
    return REstimate # modif1
