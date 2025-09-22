from scipy.io import loadmat


def loadROconfig(configuration):
    """
    Load univariate R (resp. O) as in ndarray of shape (days,) available in data/Synthetic/Univariate/.
    :param configuration: str between 'I', 'II', 'III', 'IV'
    return: Rref: ndarray of shape (days, )
            Oref: ndarray of shape (days, )
    """
    inputData = loadmat('data/Synthetic/Univariate/Config_%s.mat' % configuration, squeeze_me=True)
    assert(inputData['configuration'] == configuration)
    return inputData['Rref'], inputData['OutliersRef']


def loadROconfigMulti(example, configuration):
    """
    Load multivariate R (resp. O) as ndarray, available in data/Synthetic/Multivariate/.
    :param example: str to choose the connectivity structure (see include/build_synth/load_graph_examples.py)
    :param configuration: str between '0', 'I', 'II', 'III', 'IV'
    return: Rref: ndarray of shape (days, )
            Oref: ndarray of shape (days, )
            firstCases: ndarray of shape (deps,) that contains a proposition of realistic first cases.
    """
    inputData = loadmat('data/Synthetic/Multivariate/%s_graph/Config_delta_%s.mat' % (example, configuration),
                        squeeze_me=True)
    assert (inputData['graphName'] == example)
    if configuration == '0':
        assert (inputData['deltaName'] == '0')
    else:
        assert (inputData['deltaName'] == '\delta_\mathtt{%s}' % configuration)

    optionsMulti = {'firstCases': inputData['firstCases'],
                    'B_matrix': inputData['B_matrix']}
    return inputData['Rref'], inputData['OutliersRef'], optionsMulti
