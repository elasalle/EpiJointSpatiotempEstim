import numpy as np


def cropDatesPlusOne(fday, lday, dates, dataInit):
    """
    Crops the data and associated dates between **the day before fday** and lday.
    :param fday: First date ; must be 'year-month-day' format
    :param lday : Last date ; must be 'year-month-day' format or None
    :param dates : ndarray of shape (days,) of dates (object=dtype)
    :param dataInit : ndarray of shape either (days,) or (dep, days) of integers and len(dates) == len(data)
    :return : cropDates: ndarray of shape (shortened days, ) : dates between fday and lday (included)
              cropData : ndarray of shape either (shortened days) or (dep, shortened days) : associated cropped data

    """
    if len(np.shape(dataInit)) == 1:
        data = np.reshape(dataInit, (1, len(dataInit)))
    else:
        data = dataInit
    if fday is None:
        first = 0
    else:
        if fday < dates[1] or fday > dates[-1]:
            firstdateErr = ValueError("First day should be between " + dates[1] + " and " + dates[-1])
            raise firstdateErr
        else:
            first = np.argwhere(dates == fday)
            first = first[0, 0] - 1
    if lday is None:
        last = len(dates) - 1
    else:
        if lday < dates[1] or lday > dates[-1]:
            lastdateErr = ValueError("Last day should be between " + dates[1] + " and " + dates[-1])
            raise lastdateErr
        else:
            last = np.argwhere(dates == lday)
            last = last[0, 0]
    cropDates = dates[first:last + 1]
    cropData = data[:, first:last + 1]
    if fday is None:
        print("Warning : due to initialization and no previous infection counts before %s," % dates[0] +
              " data is exactly %d days long, not %d + 1" % (len(cropDates), len(cropDates)))
    else:
        assert (cropDates[1] == fday)
    if lday is not None:
        assert (cropDates[-1] == lday)

    if len(np.shape(dataInit)) == 1:
        return cropDates, cropData.flatten()
    else:
        return cropDates, cropData
