import numpy as np
import pandas as pd
from datetime import date
from collections import OrderedDict


# Loading daily data (essentially for the daily data updates)


def loadingData_byDay():
    """
    (SiDEP data) Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/
    Available only between May 13th, 2020 and June 27th, 2023
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray of integers (daily new infections in France)
    """
    webdata = pd.read_csv('data/Real-world/SiDEP-France-by-day-2023-06-30-16h26.csv', sep=';')
    # Dates
    timestamps = webdata['jour'].to_numpy()  # str format 'year-month-day'
    confirmedWrongFormat = webdata['P'].to_numpy()
    confirmed = np.array([p.replace(',', '.') for p in confirmedWrongFormat])
    return timestamps, np.array(confirmed, dtype=float)


def loadingData_hosp():
    """
    (SiDEP data) Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/
    (daily new hospitalizations per 'départements').
    This data is not maintained since March 31st 2023.
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray of integers (daily new entrances to the hospital in France)
    """
    webdata = pd.read_csv('data/Real-world/SiDEP-France-hosp-2023-03-31-18h01.csv', sep=';')
    # Dates
    days = webdata['jour'].to_numpy()  # str format 'year-month-day'
    totalDays = date.fromisoformat(days[len(days) - 1]) - date.fromisoformat(days[0])  # datetime format
    totalDays = totalDays.days + 1
    timestamps = days[:totalDays]

    # Data
    nbDepartments = int(len(days) / totalDays)
    hospitalized = webdata['incid_hosp'].to_numpy()
    reanimated = webdata['incid_rea'].to_numpy()
    deaths = webdata['incid_dc'].to_numpy()
    recovered = webdata['incid_rad'].to_numpy()

    H = hospitalized.reshape((nbDepartments, totalDays))  # H[:, i] hospitalized by department
    Rea = reanimated.reshape((nbDepartments, totalDays))  # Rea[i] reanimated by date
    D = deaths.reshape((nbDepartments, totalDays))
    Rec = recovered.reshape((nbDepartments, totalDays))

    totalIncid = H + Rea + D + Rec  # we also add the deaths ?
    confirmed = np.sum(totalIncid, axis=0)  # summing over all 'départements'

    return timestamps, np.array(confirmed, dtype=float)


def loadingData_JHU(country, path):
    """
    Opens daily new infections for the chosen country, based on JHU data basis.
    Available between January 23th, 2020 and March 9th, 2023.
    :param country : name of the chosen country in str format
    Loading data from Johns Hopkins University (JHU) website containing worldwide daily new infections.
    (See https://coronavirus.jhu.edu/map.html for more details)
    Processing only data from 'country'.
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray of integers (daily new infections)
    """
    file = path + 'data/Real-world/JHU-worldwide-covid19-daily-new-infections.csv'
    webdata = pd.read_csv(file)

    # Dates start at 5th column of webdata columns names
    timestamps = pd.to_datetime(webdata.columns[4:], format='%x').strftime('%Y-%m-%d')  # strftime to get only Y-m-d
    timestamps = timestamps.to_numpy()

    # Get daily new infections
    dataCountries = webdata['Country/Region']
    # dataProvinces = webdata['Province/State']
    arrWebdata = webdata.to_numpy()
    confirmedByCountry = arrWebdata[:, 4:]  # dates start at 5th column of each row
    # provinces = 0
    confirmedAbs = np.zeros(len(timestamps))
    for iC in range(0, len(dataCountries)):
        if dataCountries[iC] == country:
            confirmedAbs = confirmedAbs + confirmedByCountry[iC, :]
            # provinces += 1
    confirmed = np.diff(confirmedAbs)
    timestamps = timestamps[1:]
    return timestamps, np.array(confirmed, dtype=float)


# Loading daily data by 'départements' (returning matrices) ------------------------------------------------------------


def loadingData_hospDep():
    """
    (Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/
    (daily new hospitalizations by French 'département')
    Will mostly be used in graph version, which is still WIP.
    This data is not maintained since March 31st 2023.
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray matrix of integers (daily new entrances to the hospital in France) by 'département'
                         of shape (totalDeps, totalDays)
    """
    webdata = pd.read_csv('data/Real-world/SiDEP-France-hosp-2023-03-31-18h01.csv', sep=';')

    # Dates
    days = webdata['jour'].to_numpy()  # str format 'year-month-day'
    totalDays = date.fromisoformat(days[len(days) - 1]) - date.fromisoformat(days[0])  # datetime format
    totalDays = totalDays.days + 1  # integer now
    timestamps = days[:totalDays]

    # Retrieving infection counts
    nbDepartments = int(len(days) / totalDays)
    hospitalized = webdata['hosp'].to_numpy()
    reanimated = webdata['rea'].to_numpy()
    deaths = webdata['dc'].to_numpy()
    recovered = webdata['rad'].to_numpy()

    H = hospitalized.reshape((nbDepartments, totalDays))  # H[:, i] hospitalized by department
    Rea = reanimated.reshape((nbDepartments, totalDays))  # Rea[i] reanimated by date
    D = deaths.reshape((nbDepartments, totalDays))
    Rec = recovered.reshape((nbDepartments, totalDays))

    totalIncid = H + Rea + D + Rec  # we also add the deaths ?
    confirmed = totalIncid

    return np.array(timestamps), np.array(confirmed, dtype=float)


def loadingData_byDep():
    """
    (SiDEP data) Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/
    Daily new infections per 'département' for 102 'départements' (not considering 977 and 978)
    Will mostly be used in graph version, which is still WIP.
    Note : data is sorted daily month by month and the first month starts on May 13th 2020.
    :return: timestamps: ndarray of str format 'year-month-day' (dates from 2020-05-13)
             confirmed : ndarray matrix of integers (daily new entrances to the hospital in France) by 'département'
    """
    webdata = pd.read_csv('data/Real-world/SiDEP-France-by-day-by-dep-2023-06-30-16h26.csv', sep=';')
    # Total number of days
    days = webdata['jour'].to_numpy()  # str format 'year-month-day'
    totalDays = date.fromisoformat(days[-1]) - date.fromisoformat(days[0])  # datetime format
    totalDays = totalDays.days + 1  # last day included
    timestamps = list(OrderedDict.fromkeys(days))
    assert (totalDays == len(timestamps))
    assert (timestamps[0] == days[0])
    assert (timestamps[-1] == days[- 1])

    # Total number of 'Départements'
    depsRaw = webdata['dep'].to_numpy()
    allDeps = list(OrderedDict.fromkeys(depsRaw))[:-2]  # cropping the two last indexes : 977 and 977
    totalDeps = len(allDeps)
    assert (totalDeps == np.max(np.shape(np.where(days == days[-1]))) - 2)

    # Retrieving infection counts
    confirmed = np.zeros((totalDeps, totalDays))
    allInfectionsWrongFormat = webdata['P'].to_numpy()
    allInfections = np.array([p.replace(',', '.') for p in allInfectionsWrongFormat])
    for indexDep in np.arange(totalDeps):
        confirmed[indexDep] = allInfections[np.where(depsRaw == allDeps[indexDep])]
    return np.array(timestamps), np.array(confirmed, dtype=float), allDeps


def codeIndexes_dpt():
    # index decreases from 1 if i < 20 then decreases from 2 if 20 < i < 28, then '2A' and '2B' else  == depCodes
    firstLabels = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    labelsBeforeCorsica = [str(k) for k in range(10, 20)]  # 20 excluded
    labelsAfterCorsica = [str(k) for k in range(30, 96)]  # 97 excluded
    return firstLabels + labelsBeforeCorsica + [str(k) for k in range(21, 30)] + ['2A', '2B'] + labelsAfterCorsica


def select_french_county(matrix, codeIndexes):
    allCodeIndexes = codeIndexes_dpt()
    sliceIndexes = [allCodeIndexes.index(x) for x in codeIndexes]
    return matrix[sliceIndexes]


list_dpt = ['Ain', 'Aisne', 'Allier', 'Alpes-de-Haute-Provence', 'Hautes-Alpes',
            'Alpes-Maritimes', 'Ardèche', 'Ardennes', 'Ariège', 'Aube',
            'Aude', 'Aveyron', 'Bouches-du-Rhône', 'Calvados', 'Cantal',
            'Charente', 'Charente-Maritime', 'Cher', 'Corrèze',  # 19
            'Côte-d\'Or', 'Côtes-d\'Armor', 'Creuse', 'Dordogne', 'Doubs',
            'Drôme', 'Eure', 'Eure-et-Loir', 'Finistère',
            'Corse-du-Sud',  # 2A
            'Haute-Corse',  # 2B
            'Gard',
            'Haute-Garonne', 'Gers', 'Gironde', 'Hérault', 'Ille-et-Vilaine',
            'Indre', 'Indre-et-Loire', 'Isère', 'Jura', 'Landes',
            'Loir-et-Cher', 'Loire', 'Haute-Loire', 'Loire-Atlantique', 'Loiret',
            'Lot', 'Lot-et-Garonne', 'Lozère', 'Maine-et-Loire', 'Manche',
            'Marne', 'Haute-Marne', 'Mayenne', 'Meurthe-et-Moselle', 'Meuse',
            'Morbihan', 'Moselle', 'Nièvre', 'Nord', 'Oise',
            'Orne', 'Pas-de-Calais', 'Puy-de-Dôme', 'Pyrénées-Atlantiques', 'Hautes-Pyrénées',
            'Pyrénées-Orientales', 'Bas-Rhin', 'Haut-Rhin', 'Rhône', 'Haute-Saône',
            'Saône-et-Loire', 'Sarthe', 'Savoie', 'Haute-Savoie', 'Paris',
            'Seine-Maritime', 'Seine-et-Marne', 'Yvelines', 'Deux-Sèvres', 'Somme',
            'Tarn', 'Tarn-et-Garonne', 'Var', 'Vaucluse', 'Vendée',
            'Vienne', 'Haute-Vienne', 'Vosges', 'Yonne', 'Territoire de Belfort',
            'Essonne', 'Hauts-de-Seine', 'Seine-Saint-Denis', 'Val-de-Marne', 'Val-d\'Oise']
