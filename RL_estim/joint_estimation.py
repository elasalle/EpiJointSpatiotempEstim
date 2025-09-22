# created by Etienne Lasalle in 2025-04



from RL_estim import je_tools as jet
from RL_estim import R_wL
from RL_estim import L_estim as LL

import pickle


def RL_wAO(Z, options, max_iter=10, lambda_T=20, lambda_S=1., lambda_Fro=0.25, omegas=None, init_method="1", init_param=None, Linit=None, updateL = True, verbose=True):

    #optim param
    Gregularization="L2" # spatial regularization is || S R ||_2^2 = R^T L R

    ndep = Z.shape[0]
    dates = options["dates"]

    ZDataNorm, ZPhiNorm = jet.get_normalized_Zphi_and_Z(Z, omegas) # we compute once here. it will be usefull to compute the objective function at each iteration

    #define storage lists        
    Restims = []
    Lestims = []
    objs = []

    # initialize 
    if verbose:
        print(("    iteration 1/{}".format(max_iter)))
    # R init
    R = jet.initialize_alternate_optim(Z, ndep, options, init_method, init_param, omegas)
    Restims.append(R)
    Rdual = None
    # Linit
    if Linit is None:
        resL = LL.learningL(lambda_S, lambda_Fro, R)
        L = resL["L"]
    else:
        L = Linit
    Lestims.append(L)

    obj_res = jet.obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_T, lambda_S, lambda_Fro) # keys: "full", "L", "R", "KL", "pwlin", "GR", "Fro"
    objs.append(obj_res["full"])

    if not updateL:
        max_iter = 2

    for iter in range(1,max_iter):
        if verbose:
            print(("    iteration {}/{}".format(iter+1, max_iter)))

        resR = R_wL.Rt_with_laplacianReg(Z, L, lambda_T, lambda_S, Gregularization, dates, omegas=omegas, Rinit=R, dualRinit = Rdual) #keys: "R", "Rdual", "objs", "gaps", "op_out"
        R = resR["R"]
        Rdual = resR["Rdual"]
        obj_res = jet.obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_T, lambda_S, lambda_Fro)
        objs.append(obj_res["full"])

        if updateL:
            resL = LL.learningL(lambda_S, lambda_Fro, R, inits=resL, return_crit=True)
            L = resL["L"]
            obj_res = jet.obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_T, lambda_S, lambda_Fro) 
            objs.append(obj_res["full"])

        Lestims.append(L)
        Restims.append(R)

    extra = {
        "Lestims" : Lestims,
        "Restims" : Restims,
        "objs"    : objs
    }
    return Restims[-1], Lestims[-1], extra

def RL_wAO_noOutput(Z, options, max_iter=10, lambda_T=20, lambda_S=1., lambda_Fro=0.25, omegas=None, init_method="1", init_param=None, Linit=None, updateL = True, verbose=True, folder="/", names = {}):
    
    Rhat, Lhat, extra = RL_wAO(Z, options, max_iter, lambda_T, lambda_S, lambda_Fro, omegas, init_method, init_param, Linit, updateL, verbose)
    
    if "R" not in names.keys():
        names["R"] = "R"
    if "L" not in names.keys():
        names["L"] = "L"
    if "extra" not in names.keys():
        names["extra"] = "extra"
    
    RfullName = folder+names["R"]+".pickle"
    LfullName = folder+names["L"]+".pickle"
    extrafullName = folder+names["extra"]+".pickle"

    with open(RfullName, 'wb') as handle:
        pickle.dump(Rhat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(LfullName, 'wb') as handle:
        pickle.dump(Lhat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(extrafullName, 'wb') as handle:
        pickle.dump(extra, handle, protocol=pickle.HIGHEST_PROTOCOL)





