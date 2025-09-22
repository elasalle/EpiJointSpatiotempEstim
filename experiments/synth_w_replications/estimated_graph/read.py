import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pickle
from RL_estim import evaluation_tools as et


from itertools import product

fs = 16
plt.rcParams.update({
    'figure.figsize': (6,4),
    "text.usetex": True,
    "font.size": fs,
    'axes.labelsize': fs,
    'axes.titlesize': fs*1.2,
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,

})

######################
# gather the results #
######################

expe_folder = "/".join(__file__.split("\\")[:-1])+"/" # path to the parent folder of this file
res_folder = expe_folder+ "expe_res/"
synth_folder = "/".join(__file__.split("\\")[:-2])+"/" # path to the grand parent folder of this file
data_folder = synth_folder + "data/"

#get the parameters:
with open(expe_folder+"parameters.pickle", 'rb') as handle:
    expe_parameters = pickle.load(handle)
with open(data_folder+"parameters.pickle", 'rb') as handle:
    data_parameters = pickle.load(handle)

params = expe_parameters["params"]
params_names = expe_parameters["params_names"]
all_params_comb = list(product(*params))

data_indices = range(0, data_parameters["N_replica"])

# get the R estimations
indices = [range(len(p)) for p in params]
all_i_comb = list(product(*indices))

def names_from_indices(i_comb, data_index):
        suffix = '_' + str(data_index) + '_'+'_'.join([str(i) for i in i_comb])
        names = {"R":"R"+suffix, "L":"L"+suffix,  "extra": "extra"+suffix}
        return names

mRSEs = []
allmRSEs = []

for data_i, data_index in enumerate(data_indices):

    data_file_name = data_folder+"data_{}.pickle".format(data_index)
    with open(data_file_name, 'rb') as handle:
        data = pickle.load(handle)

    Rhats = []
    for i_comb in all_i_comb:
        RfullName = res_folder+names_from_indices(i_comb, data_i)["R"]+".pickle"
        with open(RfullName, 'rb') as handle:
                Rhat = pickle.load(handle)
                Rhats.append(Rhat)

    #compute metrics
    with open(data_folder+"data_0.pickle", 'rb') as handle:
        data = pickle.load(handle)

    R_by_cluster = data["R_by_cluster"]
    cluster_sizes = data_parameters["cluster_sizes"]

    Rstar = et.get_Rstar(R_by_cluster, cluster_sizes, shorten_time=False)

    sa = None #sum axis for the mse
    metrics = np.array(et.compute_rse_nested(Rhats, Rstar, sum_axis=sa, ))
    allmRSEs.append(metrics)

    argmin_index = np.argmin(metrics)
    best_params = all_params_comb[argmin_index]
    mRSEs.append(np.min(metrics))

average = np.mean(mRSEs)
width_CI = 1.96 * np.std(mRSEs) / np.sqrt(len(mRSEs))

print("average mRSE(R) : {:5.3f} (\pm {:5.3f})".format(average*1e4, width_CI*1e4))
    

Lstar = et.get_Lstar(cluster_sizes)
mRSEs_L, f1s = [], []
for data_i, data_index in enumerate(data_indices):

    data_file_name = data_folder+"data_{}.pickle".format(data_index)
    with open(data_file_name, 'rb') as handle:
        data = pickle.load(handle)

    Lhats = []
    for i_comb in all_i_comb:
        LfullName = res_folder+names_from_indices(i_comb, data_i)["L"]+".pickle"
        with open(LfullName, 'rb') as handle:
                Lhat = pickle.load(handle)
                Lhats.append(Lhat)

    sa = "all" #sum axis for the mse
    rse = np.array(et.compute_rse_nested(Lhats, Lstar, sum_axis=sa))
    f1 = np.array(et.compute_rse_nested(Lhats, Lstar, metric=et.F1score))

    mRSEs_L.append(np.min(rse))
    f1s.append(np.max(f1))

average = np.mean(mRSEs_L)
width_CI = 1.96 * np.std(mRSEs_L) / np.sqrt(len(mRSEs_L))

print("average mRSE(L) : {} (\pm {})".format(average, width_CI))

average = np.mean(f1s)
width_CI = 1.96 * np.std(f1s) / np.sqrt(len(f1s))

print("average F1-score : {} (\pm {})".format(average, width_CI))

mmRSEs = np.mean(np.array(allmRSEs), axis=0)
argmin_index = np.argmin(mmRSEs)
best_params = all_params_comb[argmin_index]
print("  Best parameters:")
for best_p, name in zip(best_params, params_names):
    print("  {} : {}".format(name, best_p))