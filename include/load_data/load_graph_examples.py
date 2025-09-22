import numpy as np
import networkx as nx


def connectStruct_choice(example):
    """
    :param example: str between 'Line', 'T-form', 'Hub', 'Caterpillar', 'Fly', 'Frog', 'Butterfly' and 'Flower'.
    :return: strucMat : ndarray of shape (5, 5) : Id - A where A is the adjacency matrix of the chosen directed graph.
             pos : dictionary {k: (i, j)} where k in range(5) and (i, j) euclidian position for displaying B_matrix.

    """
    if example == 'Line':
        strucMat = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0,  0],
                            [ 0, -1,  1, -1,  0],
                            [ 0,  0, -1,  1, -1],
                            [ 0,  0,  0, -1,  1]])
    elif example == 'Hub':
        strucMat = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1, -1, -1],
                            [ 0, -1,  1,  0,  0],
                            [ 0, -1,  0,  1,  0],
                            [ 0, -1,  0,  0,  1]])

    elif example == 'T-form':
        strucMat = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0,  0],
                            [ 0, -1,  1, -1, -1],
                            [ 0,  0, -1,  1,  0],
                            [ 0,  0, -1,  0,  1]])

    elif example == 'Caterpillar':
        strucMat = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0,  0],
                            [ 0, -1,  1, -1, -1],
                            [ 0,  0, -1,  1, -1],
                            [ 0,  0, -1, -1,  1]])
    elif example == 'Fly':
        strucMat = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1,  0, -1],
                            [ 0, -1,  1, -1, -1],
                            [ 0,  0, -1,  1,  0],
                            [ 0, -1, -1,  0,  1]])
    elif example == 'Frog':
        strucMat = np.array([[ 1, -1,  0,  0,  0],
                            [-1,  1, -1, -1, -1],
                            [ 0, -1,  1, -1,  0],
                            [ 0, -1, -1,  1,  0],
                            [ 0, -1,  0,  0,  1]])
    elif example == 'Butterfly':
        strucMat = np.array([[ 1, -1,  0,  0, -1],
                            [-1,  1, -1, -1, -1],
                            [ 0, -1,  1, -1,  0],
                            [ 0, -1, -1,  1,  0],
                            [-1,  0,  0, -1,  1]])
    elif example == 'Flower':
        strucMat = np.array([[ 1, -1, -1,  0, -1],
                            [-1,  1, -1, -1, -1],
                            [-1, -1,  1, -1,  0],
                            [ 0, -1, -1,  1, -1],
                            [-1, -1,  0, -1,  1]])
    elif example == '2-3': 
        #added by E. Lasalle
        # disconnected graph with a 2-clique and a 3-clique
        strucMat = np.array([[ 1, -1,  0,  0,  0],
                            [ -1,  1,  0,  0,  0],
                            [  0,  0,  1, -1, -1],
                            [  0,  0, -1,  1, -1],
                            [  0,  0, -1, -1,  1]])
    else:
        ExampleError = ValueError("See data/Synthetic/Multivariate/load_graph_examples.py for available examples.")
        raise ExampleError

    depsIndexes = range(5)
    # Displaying positions ----
    pos = {}
    if example == 'Line':
        for k in range(5):
            pos[depsIndexes[k]] = (k, 0)
    elif example == 'Hub':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (1, 1)
        pos[depsIndexes[4]] = (1, -1)
    elif example == 'T-form':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (3, 0.5)
        pos[depsIndexes[4]] = (3, -0.5)

    elif example == 'Caterpillar':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (3, 0.5)
        pos[depsIndexes[4]] = (3, -0.5)
    elif example == 'Fly':
        pos[depsIndexes[0]] = (0, 0)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0)
        pos[depsIndexes[3]] = (3, 0)
        pos[depsIndexes[4]] = (1.5, -1)
    elif example == 'Frog':
        pos[depsIndexes[0]] = (0, 0.5)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0.5)
        pos[depsIndexes[3]] = (2, -0.5)
        pos[depsIndexes[4]] = (0, -0.5)
    elif example == 'Butterfly':
        pos[depsIndexes[0]] = (0, 0.5)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0.5)
        pos[depsIndexes[3]] = (2, -0.5)
        pos[depsIndexes[4]] = (0, -0.5)
    elif example == 'Flower':
        pos[depsIndexes[0]] = (0, 0.5)
        pos[depsIndexes[1]] = (1, 0)
        pos[depsIndexes[2]] = (2, 0.5)
        pos[depsIndexes[3]] = (2, -0.5)
        pos[depsIndexes[4]] = (0, -0.5)
    elif example == '2-3': 
        pos[depsIndexes[0]] = (0  ,  0.5)
        pos[depsIndexes[1]] = (0  , -0.5)
        pos[depsIndexes[2]] = (1  ,  0.5)
        pos[depsIndexes[3]] = (1  , -0.5)
        pos[depsIndexes[4]] = (1.7,  0)
    else:
        ExampleError = ValueError("See data/Synthetic/Multivariate/load_graph_examples.py for available examples.")
        raise ExampleError

    labels = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    colorMap = ['#99CCFF', '#339966', '#FFC266', '#FF6666', '#999999']

    return strucMat, pos, labels, colorMap


def set_Graph_fromMatrix(matrix):
    """
    Displays the undirected network associated to Id - A `matrix` is the adjacency matrix of a chosen directed graph.
    Note that edges are doubled if the connectivity structure is initially not directed.
    :param matrix: Id - A where A is the adjacency matrix.
    :returns: Graph: networkx Graph object (G, E) used to display graph

    """
    Graph = nx.Graph()
    n, m = np.shape(matrix)
    for i in range(n):
        for j in range(m):
            if matrix[i, j] == -1:
                Graph.add_edge(i, j)  # duplicates are not drawn nor counted as a different edge
    return Graph
