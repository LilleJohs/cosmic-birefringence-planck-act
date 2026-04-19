import numpy as np

# Column layout for unbinned .dat files from the spectrum pipeline:
#   ell TT TE ET EE EB BE BB TB BT
SPEC_ORDER = ("TT", "TE", "ET", "EE", "EB", "BE", "BB", "TB", "BT")

COL_ELL = 0
COL_TT = 1
COL_TE = 2
COL_ET = 3
COL_EE = 4
COL_EB = 5
COL_BE = 6
COL_BB = 7
COL_TB = 8
COL_BT = 9


def load_unbinned_spectrum(path):
    """Load a single unbinned .dat file written by the spectrum pipeline.

    The file has columns: ell TT TE ET EE EB BE BB TB BT
    with header lines starting with '#'.

    Parameters
    ----------
    path : str
        Path to the .dat file.

    Returns
    -------
    dict
        Keys: 'ell' (int array) plus 9 spectrum keys (float arrays, Cl units).
    """
    data = np.loadtxt(path, comments="#")
    result = {"ell": data[:, COL_ELL].astype(int)}
    for key, col in zip(SPEC_ORDER, range(COL_TT, COL_BT + 1)):
        result[key] = data[:, col]
    return result
