import numpy as np

from include.load_data.get_counts import get_real_counts
from include.estim.Rt_UnivariateOutliers import Rt_U_O
from include.build_synth import buildData_fromRO as build

def drawZ_uni(R, O, Z0, firstDay="2000-01-01", gamma=None, alpha=None):
    if (gamma is None) and (alpha is None):
        gamma=1
    elif gamma is None:
        gamma = alpha * Z0
    firstCases = gamma * np.random.poisson(Z0 / gamma)
    Z, options = build.buildData_anyRO(R, O, firstCases, firstDay, gamma=gamma)
    return Z, gamma

def drawZ_multi(cluster_sizes, R_by_cluster, O_by_cluster, ZData_by_cluster, with_O=True, firstDay="2000-01-01", gamma=None, alpha=None):
    nclusters = len(cluster_sizes)
    ZData_by_country = []
    extra = {
        "gammas" : []
    }
    for i in range(nclusters):
        for _ in range(cluster_sizes[i]):
            if with_O:
                O = O_by_cluster[i,:]
            else:
                O = 0*O_by_cluster[i,:]
            ZData, gamma_used = drawZ_uni(R_by_cluster[i,:], O, ZData_by_cluster[i][0], firstDay, alpha=alpha, gamma=gamma)
            extra["gammas"].append(gamma_used)
            # if alpha is None:
            #     gamma = 1
            # else:
            #     gamma = alpha*ZData_by_cluster[i][0]
            # firstCases = gamma * np.random.poisson(ZData_by_cluster[i][0] / gamma)
            # if not(with_O):
            #     O_by_cluster = 0*O_by_cluster
            # ZData, options = build.buildData_anyRO(R_by_cluster[i,:], O_by_cluster[i,:], firstCases, firstDay, alpha=gamma)

            ZData_by_country.append(ZData)
    ZData_by_country = np.array(ZData_by_country)
    extra["gammas"] = np.array(extra["gammas"])
    return ZData_by_country, extra


def generate_synthZ(countries, cluster_sizes, firstDay, lastDay, gamma=None, alpha=None, lambdaU_L = 3.5, lambdaU_O = 0.03, with_O=True):
    nclusters = len(cluster_sizes)
    
    # get number of new cases for each cluster (i.e. each country)
    ZData_by_cluster = []
    for i in range(nclusters):
        ZData, options = get_real_counts(countries[i], firstDay, lastDay, 'JHU')
        ZData_by_cluster.append(ZData)
    ZData_by_cluster = np.array(ZData_by_cluster)
    optionsZ = options

    # infer the reproduction number for each of these clusters
    R_by_cluster, O_by_cluster, _ = Rt_U_O(ZData_by_cluster, lambdaU_L, lambdaU_O, options=optionsZ)

    # generate the synthetic Z data according to the R and O of their respective cluster
    ZData_by_country, extra = drawZ_multi(cluster_sizes, R_by_cluster, O_by_cluster, ZData_by_cluster, with_O, firstDay, gamma, alpha)

    return ZData_by_country, ZData_by_cluster, R_by_cluster, O_by_cluster, options