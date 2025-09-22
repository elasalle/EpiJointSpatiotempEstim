
import numpy as np

from include.estim.Rt_UnivariateOutliers import Rt_U_O
from include.estim.Rt_Univariate import myRt_U
from include.estim import Rt_Multivariate as RtM
from include.estim import laplacianLearning as LL
from include.estim import Rt_MLE as RtMLE

from include.optim_tools.fidelity_terms_DKL import DKL_no_outlier as DKL
from include.optim_tools import crafting_phi,  opL, conversion_pymat as mat2py

Phi = crafting_phi.buildPhi()

def get_normalized_Zphi_and_Z(Z, omegas=None):
    

    nb_deps, days = Z.shape
    ZDataDep  = np.zeros((nb_deps, days - 1))
    ZPhiDep   = np.zeros((nb_deps, days - 1))
    ZPhiNorm  = np.zeros((nb_deps, days - 1))
    ZDataNorm = np.zeros((nb_deps, days - 1))

    for d in range(nb_deps):
        _, ZDataDep[d], ZPhiDep[d] = crafting_phi.buildZPhi(None, Z[d], Phi)

    if omegas is None:
        std = np.std(ZDataDep, axis=1)
        omegas = 1/std 

    ZDataNorm = ZDataDep * omegas[:,None]
    ZPhiNorm = ZPhiDep * omegas[:,None]
    return ZDataNorm, ZPhiNorm


def obj_function(R, L, ZDataNorm, ZPhiNorm, lambda_pwlin, lambda_GR, lambda_Fro):
    """_summary_

    Args:
        R (_type_): _description_
        L (_type_): _description_
        ZDataNorm (_type_): _description_
        ZPhiNorm (_type_): _description_
        lambda_pwlin (_type_): _description_
        lambda_GR (_type_): _description_
        lambda_Fro (_type_): _description_

    Returns:
        dictionnary: keys are "full", "L", "R", "KL", "pwlin", "GR", "Fro"
    """

    cst = np.sum(ZDataNorm[ZDataNorm > 0] * (np.log(ZDataNorm[ZDataNorm > 0]) - 1))
    KL_term = DKL(R, ZDataNorm, ZPhiNorm) + cst

    param_pwlin = mat2py.struct()
    param_pwlin.lambd = lambda_pwlin
    param_pwlin.type = '1D'
    pwlin_term = np.sum(np.abs(opL.opL(R, param_pwlin)))

    GR_term = lambda_GR * np.sum(R * (L @ R))

    Fro_term = lambda_Fro * np.sum(L**2)

    crit = KL_term + pwlin_term + GR_term + Fro_term
    crit_L = GR_term + Fro_term
    crit_R = KL_term + pwlin_term + GR_term

    objs = {
        "full" : crit,
        "L" : crit_L,
        "R" : crit_R,
        "KL" : KL_term,
        "pwlin" : pwlin_term,
        "GR" : GR_term,
        "Fro" : Fro_term
    }

    return objs

def make_lambda_GR_as_list(max_iter, lambda_GR):
    #make sure that lambda_GR is a list of size max_iter
    if isinstance(lambda_GR, list):
        if len(lambda_GR)<max_iter:
            lambda_GR = lambda_GR + [lambda_GR[-1]]*(max_iter - len(lambda_GR)) #if the initial list is to small, complete with the last value
        else:
            lambda_GR = lambda_GR[:max_iter] #if it is too long, cut it.
    elif isinstance(lambda_GR, (float, int)):
        lambda_GR = [lambda_GR]*max_iter # create a constant list
    else:
        ValueError("lambda_GR should be a list, an int or a float, received {}.".format(type(lambda_GR)))
    return lambda_GR

def initialize_alternate_optim(Z, ndep, options, init_method="U", init_param=None):
    if init_method=="U":
        if init_param is None:
            init_param = {"options":options, "lambdaU_pwlin":50}
        R = myRt_U(Z,  init_param["lambdaU_pwlin"], init_param["options"])
    elif init_method=="1":
        days = Z.shape[1]
        R = np.ones((ndep, days-1))
    elif init_method=="MLE":
        if init_param is None:
            init_param = {"options":options}
        R = []
        for i in range(ndep):
            Ri, _ = RtMLE.Rt_MLE(Z[i], init_param["options"])
            R.append(Ri)
        R = np.array(R)
    elif init_method=="UO":
        if init_param is None:
            init_param = {"options":options, "lambdaU_pwlin":3.5, "lambdaU_O":0.02}
        R, _, _ = Rt_U_O(Z, init_param["lambdaU_pwlin"], init_param["lambdaU_O"], init_param["options"])
    return R


def Rt_L(Z, options, max_iter=10,lambda_pwlin=20, lambda_GR=1., lambda_Fro=0.25, init_method="U", init_param=None, smart_init=True):

    #handle param
    Gregularization="L2"
    dates = options["dates"]
    ndep = Z.shape[0]
    
    #initialize variables        
    Restims = []
    Lestims = []
    extra = {
        "objs" : [],
        "objs_Rupdate" : []
    }


    ZDataNorm, ZPhiNorm = get_normalized_Zphi_and_Z(Z) # we compute once here. it will be usefull to compute the objective function at each iteration

    # initialize
    R = initialize_alternate_optim(Z, ndep, options, init_method, init_param)
    L = LL.learningL(lambda_GR, lambda_Fro, R)
    obj_res = obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_pwlin, lambda_GR, lambda_Fro)
    extra["objs"].append(obj_res)
    Restims.append(R)
    Lestims.append(L)
    
    dualR = None
    for iter in range(max_iter):

        #update R
        if smart_init:
            res = RtM.Rt_with_laplacianReg(Z, L, lambda_pwlin, lambda_GR, Gregularization, dates, Rinit=R, dualRinit=dualR)
        else:
            res = RtM.Rt_with_laplacianReg(Z, L, lambda_pwlin, lambda_GR, Gregularization, dates)
        # the keys of res dict are : "R", "Rdual", "objs", "gaps", "op_out"
        R = res["R"]
        dualR = res["dualR"]
        extra["objs_Rupdate"].append(res["objs"])

        # update L
        L, _ = LL.learningL(lambda_GR, lambda_Fro, R, return_crit=True)

        # get obj
        obj_res = obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_pwlin, lambda_GR, lambda_Fro)
        extra["objs"].append(obj_res)
        Lestims.append(L)
        Restims.append(R)
    return Restims, Lestims, extra