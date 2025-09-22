import numpy as np


def Tikhonov_spat_corr_config(REstimates, config, options):
    """
    Compute diffusion of the signal REstimate on the B_matrix associated to its transposed incidence matrix B_matrix.
    :param REstimates: ndarray of shape (|V|, days) R estimates by territory/county.
    :param config: str between '0', 'I', 'II', 'III', 'IV'
    :param options: dictionary containing at least
        - B_matrix: ndarray of shape ([E|, |V|) transposed incidence matrix of the associated connectivity structure,
        represented by a graph G = (V, E) where each node is a territory/county.
        - '0', 'I', 'II', 'III', 'IV' corresponding to inter-county regularization levels.
    associated to a graph G = (V, E) where each node corresponds to a territory/county.
    :return: ndarray of shape (|V|, days) R estimates diffused
    """
    B_matrix = options['B_matrix']
    delta = options[config]
    dep, days = np.shape(REstimates)
    L = np.dot(np.transpose(B_matrix), B_matrix)
    nbChosenDeps, m = np.shape(L)
    assert (m == nbChosenDeps)
    Tikhonov = np.eye(dep) + 2 * delta * L
    return np.linalg.solve(Tikhonov, REstimates)


def Tikhonov_spat_corr(REstimates, B_matrix, delta):
    """
    Compute diffusion of the signal REstimate on the B_matrix associated to its transposed incidence matrix B_matrix.
    :param REstimates: ndarray of shape (dep, days) R estimates by territory
    :param B_matrix: ndarray of shape (edges, dep) transposed incidence matrix of the associated B_matrix
    :param delta: float hyperparameter controlling the diffusion
    :return: ndarray of shape (dep, days) R estimates diffused
    """
    dep, days = np.shape(REstimates)
    L = np.dot(np.transpose(B_matrix), B_matrix)
    nbChosenDeps, m = np.shape(L)
    assert (m == nbChosenDeps)
    Tikhonov = np.eye(dep) + 2 * delta * L
    return np.linalg.solve(Tikhonov, REstimates)
