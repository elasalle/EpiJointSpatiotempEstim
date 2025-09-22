import numpy as np
from include.optim_tools.conversion_pymat import struct


def opLadj(y, param=struct(), filter_def="laplacian", computation="direct"):
    """
    Computes the adjoint of the operator used in the penalization.
    Translation from N. PUSTELNIK, CNRS, ENS Lyon, MATLAB code implementation in June 2019.
    :param y: ndarray of shape (dep, days)
    :param filter_def: option between 'gradient' and 'laplacian'
    :param computation: option between 'direct' and 'fourier'
    :param param: structure with options
    :return: x <- ndarray of shape same as np.shape(y)
    """
    dim = np.shape(y)
    x = np.zeros(dim)

    if not (hasattr(param, "lambd")):
        param.lambd = 1  # to match op_out
    if not (hasattr(param, "type")):
        param.type = '1D'
    else:
        assert (param.type == '1D')

    OptionError = ValueError("filter_def = %s and computation = %s not implemented yet." % (filter_def, computation))

    if param.type == "1D":
        if filter_def == "gradient":
            raise OptionError
        elif filter_def == "laplacian":
            if isinstance(param.lambd, int) or isinstance(param.lambd, float):
                if computation == "fourier":
                    raise OptionError
                else:
                    x[:, 0] = 0.25 * y[:, 0]
                    x[:, 1] = -0.5 * y[:, 0] + 0.25 * y[:, 1]
                    x[:, 2:-2] = 0.25 * y[:, 2:- 2] - 0.5 * y[:, 1:- 3] + 0.25 * y[:, :- 4]
                    x[:, -2] = 0.25 * y[:, - 4] - 0.5 * y[:, - 3]
                    x[:, -1] = 0.25 * y[:, - 3]
                    x = param.lambd * x
            else:
                xbeg = np.array([0.25 * y[0, 0], -0.5 * y[0, 0] + 0.25 * y[0, 1]])
                xend = np.array([0.25 * y[0, - 3] - 0.5 * y[0, - 2], 0.25 * y[0, - 2]])
                x = param.lambd[:,None] * np.concatenate((xbeg,
                                                                 0.25 * y[0, 2:- 2] - 0.5 * y[0, 1:- 3] + 0.25 * y[0, :- 4],
                                                                 xend))
    else:
        raise OptionError
    return x
