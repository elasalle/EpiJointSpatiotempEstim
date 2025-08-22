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

data_indices = range(0, parameters["N_replica"])
max_iter=1
omegas=None
init_method="MLE"

parameters2 = {
     "max_iter": max_iter,     
     "omegas": omegas,     
     "init_method": init_method
}

with open(expe_folder+"parameters.pickle", 'wb') as handle:
        pickle.dump(parameters2, handle, protocol=pickle.HIGHEST_PROTOCOL)


################
# Parallelized #
################

for data_i, data_index in enumerate(data_indices):

    print("Loading data {} ({}/{})".format(data_index, data_i+1, len(data_indices)))

    data_file_name = data_folder+"data_{}.pickle".format(data_index)
    with open(data_file_name, 'rb') as handle:
        data = pickle.load(handle)

    def names_from_indices(data_index):
        suffix = '_' + str(data_index)
        names = {"R":"R"+suffix, "L":"L"+suffix,  "extra": "extra"+suffix}
        return names

    print("Start parallel computations...")

    ti = time()
    je.RL_wAO_noOutput(data["Z"], parameters["options"], max_iter, 0., 0., 0., omegas, 
                       init_method, updateL=True, verbose=False, folder=res_folder, names=names_from_indices(data_index)) 
    tf = time()

    print("done in {:5.3f}min".format((tf-ti)/60))