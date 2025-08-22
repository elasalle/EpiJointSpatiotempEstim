import numpy as np
from itertools import product
from   joblib import Parallel, delayed
from RL_estim import evaluation_tools as et

from RL_estim import joint_estimation as je
from RL_estim import evaluation_tools as et
from time import time

import pickle

from importlib import reload
reload, reload(je)

np.random.seed(2025)

########################
# Set path and folders #
########################

path="C:/Users/adminuser/Documents/GitHub/Covid_R_L_estim/" # path to project folder
data_folder = "/".join(__file__.split("\\")[:-2])+"/" + "data/"
expe_folder = "/".join(__file__.split("\\")[:-1])+"/"
res_folder = expe_folder + "expe_res/"

with open(data_folder+"parameters.pickle", 'rb') as handle:
    parameters = pickle.load(handle)
# parameters is the following dictionary
# parameters = {
#     "cluster_sizes" : cluster_sizes, 
#     "firstDay" : firstDay, 
#     "lastDay" : lastDay, 
#     "countries" : countries, 
#     "lambdaU_L" : lambdaU_L, 
#     "lambdaU_O" : lambdaU_O, 
#     "alpha" : alpha,
#     "N_replica" : N_replica,
#     "options": optionsZ
# }


#####################################
# set parameters for the estimation #
#####################################

# data_indices = range(0, parameters["N_replica"])
data_indices = range(15,18)
max_iter=10
omegas=None
init_method="epiEstim"
lambda_Ts = np.logspace(0,2,16)
lambda_Ss = np.logspace(-2,3,16)
lambda_Fros = [0]

params_names = ["lambda_T", "lambda_S", "lambda_Fro"] 
params = [lambda_Ts, lambda_Ss, lambda_Fros] 

parameters2 = {
     "max_iter": max_iter,     
     "omegas": omegas,     
     "init_method": init_method,     
     "params": params,
     "params_names": params_names
}

with open(expe_folder+"parameters.pickle", 'wb') as handle:
        pickle.dump(parameters2, handle, protocol=pickle.HIGHEST_PROTOCOL)

Lref = et.get_Lstar_blured(parameters["cluster_sizes"])

################
# Parallelized #
################

for data_i, data_index in enumerate(data_indices):

    print("Loading data {} ({}/{})".format(data_index, data_i+1, len(data_indices)))

    data_file_name = data_folder+"data_{}.pickle".format(data_index)
    with open(data_file_name, 'rb') as handle:
        data = pickle.load(handle)

    def names_from_indices(i_comb, data_index):
        suffix = '_' + str(data_index) + '_'+'_'.join([str(i) for i in i_comb])
        names = {"R":"R"+suffix, "L":"L"+suffix,  "extra": "extra"+suffix}
        return names

    all_params_comb = list(product(lambda_Ts, lambda_Ss, lambda_Fros))
    nT, nS, nFro = len(lambda_Ts), len(lambda_Ss), len(lambda_Fros)
    all_i_comb = list(product(range(nT), range(nS), range(nFro)))

    print("Start parallel computations...")

    ti = time()
    res = Parallel(n_jobs=-1)(
        delayed(je.RL_wAO_noOutput)(data["Z"], parameters["options"], max_iter, *p_set, omegas, init_method, Linit=Lref, updateL=False, verbose=False, folder=res_folder, names=names_from_indices(i_set, data_index)) for p_set, i_set in zip(all_params_comb, all_i_comb)
        )
    tf = time()

    print("done in {:5.3f}min".format((tf-ti)/60))