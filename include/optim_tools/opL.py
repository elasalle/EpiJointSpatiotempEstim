import numpy as np
from include.optim_tools.conversion_pymat import struct


# These 2 functions are used to create L and L^*

def opL(x, param=struct(), filter_def="laplacian", computation="direct"):
    """
    Define linear operator associated with the filter in the prior.
    Translation from Nelly Pustelnik Matlab's code, CNRS, ENS Lyon June 2019
    :param x: ndarray of shape (dep, days)
    :param filter_def: option between 'gradient' and 'laplacian'
    :param computation: option between 'direct' and 'fourier'
    :param param: structure with options
    :return: xt <- ndarray of same shape as x (np.shape(x))
    """

    if not (hasattr(param, "lambd")):
        param.lambd = 1  # to match op_out
    if not (hasattr(param, "type")):
        param.type = "1D"
    else:
        assert (param.type == '1D')

    if len(np.shape(x)) == 1:
        days, = np.shape(x)
        x = np.reshape(x, (1, days))

    dim = np.shape(x)
    xt = np.zeros(dim)

    if param.type == "1D":
        if filter_def == "gradient":
            OptionError = ValueError(
                "filter_def = %s and computation = %s not implemented yet." % (filter_def, computation))
            raise OptionError
        elif filter_def == "laplacian":
            if isinstance(param.lambd, int) or isinstance(param.lambd, float):
                if computation == "fourier":
                    print("filter_def = %s and computation = %s not implemented yet." % (filter_def, computation))
                else:
                    xt[:, :-2] = param.lambd * (x[:, 2:] / 4 - x[:, 1:- 1] / 2 + x[:, :- 2] / 4)
            else:
                xt = np.dot(np.diag(param.lambd), np.concatenate((x[:, 2:] / 4 - x[:, 1:- 1] / 2 + x[:, :- 2] / 4,
                                                                  np.zeros(dim[0], 2))))
    else:
        OptionError = ValueError("filter_def = %s, computation = %s not implemented yet for type = %s."
                                 % (filter_def, computation, param.type))
        raise OptionError
    return xt
