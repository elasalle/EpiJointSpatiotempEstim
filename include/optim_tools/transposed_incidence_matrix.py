import numpy as np


def transposed_incidence_matrix(matrix):
    """
    Computes the transposed incidence matrix `B_matrix` of a chosen graph G = (V, E), that also corresponds to the Total
     Variations Operator matrix.
    :param matrix: Id - A where A is the adjacency matrix
    :return: GTV_op: ndarray of shape (|E|, |V|).
    """
    nbDep, mDep = np.shape(matrix)  # here nbDep = mDep and should be 2 less than total number of 'départements'
    theoreticalEdges = max(np.shape(np.where(matrix == -1)))  # assuming there are + edges than nodes
    GTV_op = np.zeros((max(np.shape(np.where(matrix == -1))), nbDep))  # assuming there are + edges than nodes
    edges = 0  # will browse all rows of GTV_op and count the edges
    for row in range(0, nbDep):
        # np.where returns a tuple in which there is an array (for vector use)
        indP1 = np.where(matrix[row] == 1)  # browsing through each 'département' once as "starting point"
        indN1 = np.where(matrix[row] == -1)[0]  # browsing through each 'département' linked to the previous one
        for j in range(0, max(np.shape(indN1))):  # assuming there are + edges than nodes
            GTV_op[edges, indP1] = 1
            GTV_op[edges, indN1[j]] = -1
            edges += 1
    assert (edges == theoreticalEdges)

    return GTV_op

