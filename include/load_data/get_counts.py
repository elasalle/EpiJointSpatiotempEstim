import numpy as np
from scipy.io import loadmat

from include.optim_tools.transposed_incidence_matrix import transposed_incidence_matrix
from include.load_data import date_choice, load_counts as load


def get_real_counts(country, fday, lday, dataBasis, path_to_main_folder=""):
    """
    Returns real dates and associated new daily Covid19 cases data in the adequate format,
    between the day before fday (for initialization) and lday. Data is opened from either dataBasis='SPF'
    Santé Publique France or 'JHU' Johns Hopkins University.
    :param country: str between all countries available
    :param fday: str in format 'YYYY-MM-DD'
    :param lday: str in format 'YYYY-MM-DD'
    :param dataBasis: str between 'SPF' and 'JHU'. See ./load_counts.py
    :return: dates ndarray of shape (days + 1, ) of str in format 'YYYY-MM-DD'
             data  ndarray of shape (days + 1, ) of float (round numbers)
    """
    # Opening data with chosen country
    if dataBasis == 'JHU':
        print("Opening data from Johns Hopkins University.")
        timestampsInit, ZDataInit = load.loadingData_JHU(country, path_to_main_folder)
    elif dataBasis == 'SPF':
        if country == 'France':
            print("Opening data from Santé Publique France.")
            timestampsInit, ZDataInit = load.loadingData_byDay()
        else:
            CountryError = ValueError("Santé Publique France (SPB) only provides data for France, not %s." % country)
            raise CountryError
    else:
        DataBasisUnknown = ValueError("Data Basis %s unknown." % dataBasis)
        raise DataBasisUnknown

    # Crop to dates choice
    timestampsCropped, ZDataCropped = date_choice.cropDatesPlusOne(fday, lday, timestampsInit, ZDataInit)
    ZDataCropped[ZDataCropped<0] = 0 #modif1

    options = {'dates': timestampsCropped,
               'data': ZDataCropped,
               'country': country,
               'fday': fday,
               'lday': lday,
               'dataBasis': dataBasis}
    return ZDataCropped, options


def get_real_counts_by_county(fday, lday, dataBasis='SPF'):
    """
    :param fday:
    :param lday:
    :param dataBasis:
    """
    if dataBasis == 'SPF':
        timestampsInit, ZDataDepInit, allDeps = load.loadingData_byDep()
    elif dataBasis == 'hosp':
        timestampsInit, ZDataDepInit, allDeps = load.loadingData_hospDep()
    else:
        DataBasisUnknown = ValueError("Data Basis %s unknown." % dataBasis)
        raise DataBasisUnknown

    # Cropping following dates (time cropping)
    timestampsCropped, ZDataDepCropped = date_choice.cropDatesPlusOne(fday, lday, timestampsInit, ZDataDepInit)

    deps = np.array(allDeps[:96])  # spatial cropping to remove DROM-COM

    # Transposed incidence matrix associated
    fileStructConnect = loadmat('data/Real-world/counties_sharing_borders.mat')
    structMat = fileStructConnect['matrice']  # french for matrix
    structMat[-1, -1] = 1  # correction of a mistake : every dep verifies depContMatrix[i, i] = 1

    # Create the G matrix used to define the Graph Total Variation ----------------------------------------------
    B_matrix = transposed_incidence_matrix(structMat)

    output = {'dates': timestampsCropped,
              'counties': deps,
              'structConnect': structMat,
              'B_matrix': B_matrix}
    return ZDataDepCropped[:96], output

