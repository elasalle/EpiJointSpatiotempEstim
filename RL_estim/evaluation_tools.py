import numpy as np
from sklearn.metrics import f1_score


def get_Rstar(R_by_cluster, cluster_sizes, shorten_time=True):
    C = np.sum(cluster_sizes)
    T = R_by_cluster.shape[1]
    if shorten_time:
        T -= 1
    Rstar = np.zeros((C,T))
    k = 0
    for i, size in enumerate(cluster_sizes):
        for _ in range(size):
            if shorten_time:
                Rstar[k,:] = R_by_cluster[i, 1:]
            else:
                Rstar[k,:] = R_by_cluster[i, :]
            k += 1
    return Rstar

def get_Lstar(cluster_sizes):
    C = np.sum(cluster_sizes)
    Lstar = np.zeros((C,C))
    start = 0
    for size in cluster_sizes:
        end = start + size
        miniblock = -1/(size-1)*np.ones((size,size))
        np.fill_diagonal(miniblock, 1)
        Lstar[start:end,start:end] = miniblock
        start = end
    return Lstar

def get_Lstar_blured(cluster_sizes, r=0.01, method="cst_weight_nonE", set_diagonal=True):
    C = np.sum(cluster_sizes)
    Lstar = get_Lstar(cluster_sizes)
    max_weight = - np.min(Lstar)
    if method=="cst_weight_nonE":
        # add a minimal weight everywhere
        Lstar -= r*max_weight*np.ones((C,C))
    elif method=="blured_weights":
        blur = np.random.normal(0,r,Lstar.shape)
        blur = (blur + blur.T)/np.sqrt(2)
        Lstar += Lstar*blur
        #ensure nonpositive offdiagonal coefficients
        Lstar = np.clip(Lstar, None, 0) #this modifies the diagonal but it is set correctly below
    if set_diagonal:
        #respect that rows sum to zero by adjusting diagonal values
        new_diag = -np.sum(Lstar, axis=1) + np.diag(Lstar) 
        Lstar[range(C), range(C)] = new_diag
        #scale Lstar so that the trace is C
        Lstar = C / np.sum(new_diag) * Lstar
    return Lstar

def get_fully_connect_L(cluster_sizes):
    C = np.sum(cluster_sizes)
    L = -1/(C-1)*np.ones((C,C))
    np.fill_diagonal(L, 1)
    return L

#some metrics


def meanRSE(x, xref, sum_axis="none"):
    """Gives the mean relative squared error between an array and its reference.

    Args:
        x (array like): the array of interest.
        xref (array like (same shape as x)): the reference array.
        sum_axis (None, str or int, optional): Can be None, "all" or an integer. Defaults to None.
            If None, the mean relative squared error of each entry is returned.
            If "all", the relative squared error in Froebenius norm is returned.
            If an integer, the mean squared error in L2 norm over the sum_axis axis is given. 
                E.g., to compute the mean of the relative squared error of the rows set sum_axis=1,
                s.t. np.mean( np.sum((x-xref)**2, axis=1) / np.sum(xref**2, axis=1) ) is returned.

    Returns:
        float: The mean relative squared error.
    """
    if sum_axis is None:
        return np.mean((x-xref)**2/xref**2)
    elif sum_axis=="all":
        return np.sum((x-xref)**2)/np.sum(xref**2)
    elif isinstance(sum_axis, int):
        return np.mean( np.sum((x-xref)**2, axis=sum_axis) / np.sum(xref**2, axis=sum_axis) )
    else:
        ValueError("'sum_axis' should be either 'none', 'all' or an integer. Received {}".format(sum_axis))
        

def compute_rse_nested(structure, ref, metric=None, sum_axis=None):

    if metric is None:
        metric = lambda a, b : meanRSE(a, b, sum_axis)

    if isinstance(structure, list) or (isinstance(structure, np.ndarray) and len(structure.shape)>2):
        return [compute_rse_nested(item, ref, metric) for item in structure]
    else:
        return metric(structure, ref)
    
def F1score(L, Ltrue, threshold = 1e-14):
    C = L.shape[0]
    indices =  np.triu_indices(C, 1)
    w, wtrue = -L[indices], -Ltrue[indices] #get the weights of edges
    e, etrue = (w>threshold).astype(int), (wtrue>threshold).astype(int) #weight higher than threshold gives proper edge
    if etrue.sum() == 0:
        raise TypeError("True has no edge")
    return f1_score(etrue, e)