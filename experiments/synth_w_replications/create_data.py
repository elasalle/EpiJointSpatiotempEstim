import numpy as np
import matplotlib.pyplot as plt


from include.build_synth import buildData_from_countries as generator
from include.load_data.get_counts import get_real_counts

from RL_estim import joint_estimation as je
from RL_estim.R_univartiate_wOutliers import Rt_U_O

import pickle

from importlib import reload
reload, generator, reload(je)

np.random.seed(2025)

########################
# Set path and folders #
########################

path="C:/Users/adminuser/Documents/GitHub/Covid_R_L_estim/" # path to project folder
data_folder = "/".join(__file__.split("\\")[:-1])+"/" + "data/"

###################
# Generate Z data #
###################

print("Generating data")

cluster_sizes = [3,3,3]
nclusters = len(cluster_sizes)

firstDay, lastDay = "2021-04-01", "2021-06-09"
countries = ['France', 'South Africa', 'India']

alpha = 0.01

lambdaU_L=3.5
lambdaU_O=0.03
with_O = False

N_replica = 20 

ZData_by_cluster = []
for i in range(nclusters):
    ZData, options = get_real_counts(countries[i], firstDay, lastDay, 'JHU', path)
    ZData_by_cluster.append(ZData)
ZData_by_cluster = np.array(ZData_by_cluster)
optionsZ = options

# infer the reproduction number for each of these clusters
R_by_cluster, O_by_cluster, _ = Rt_U_O(ZData_by_cluster, lambdaU_L, lambdaU_O, options=optionsZ)

parameters = {
    "cluster_sizes" : cluster_sizes, 
    "firstDay" : firstDay, 
    "lastDay" : lastDay, 
    "countries" : countries, 
    "lambdaU_L" : lambdaU_L, 
    "lambdaU_O" : lambdaU_O, 
    "alpha" : alpha,
    "N_replica" : N_replica,
    "options": optionsZ
}


with open(data_folder+"parameters.pickle", 'wb') as handle:
        pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

for n_rep in range(N_replica):

    ZData_by_country, extra = generator.drawZ_multi(cluster_sizes, R_by_cluster, O_by_cluster, ZData_by_cluster, with_O, firstDay, alpha=alpha)

    data = {
        "R_by_cluster" : R_by_cluster, 
        "Z_by_cluster" : ZData_by_cluster,
        "Z" : ZData_by_country
    }

    data_file_name = data_folder+"data_{}.pickle".format(n_rep)
    with open(data_file_name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)