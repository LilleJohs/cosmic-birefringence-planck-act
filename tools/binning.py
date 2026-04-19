"""Multipole binning utilities for CMB power spectra."""
import os

import numpy as np

from .data_loading import OUTPUT_DIR


def get_unified_binning_file():
    """Return path to the unified binning file (Δℓ=20 below ℓ=576, then Δℓ=50)."""
    return _get_binning_file("BIN_ACTPOL_50_AMENDED_20_BELOW_576")


def _get_binning_file(name):
    fpath = os.path.join(OUTPUT_DIR, "auxiliary", name)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Binning file not found at {fpath}.")
    return fpath


def load_binning_file(binning_file, lmin=None, lmax=None):
    """Parse a BIN_ACTPOL-format binning file (3 columns: bin_low, bin_high, bin_center).

    Optionally filter bins to an ell range.

    Parameters
    ----------
    binning_file : str
        Path to binning file.
    lmin : int, optional
        Drop bins whose high edge <= lmin.
    lmax : int, optional
        Drop bins whose low edge > lmax.

    Returns
    -------
    bin_lo, bin_hi, bin_center : 1D arrays
    """
    data = np.loadtxt(binning_file)
    bin_lo, bin_hi, bin_center = data[:, 0], data[:, 1], data[:, 2]
    keep = np.ones(len(bin_lo), dtype=bool)
    if lmin is not None:
        keep &= bin_hi > lmin
    if lmax is not None:
        keep &= bin_lo <= lmax
    return bin_lo[keep], bin_hi[keep], bin_center[keep]


def bin_array(ell, values, bin_lo, bin_hi):
    """Bin a 1D array into bins defined by [bin_lo, bin_hi] (inclusive).

    Parameters
    ----------
    ell : array
        Multipole values.
    values : array
        Values to bin (same length as ell).
    bin_lo, bin_hi : arrays
        Bin boundaries (both inclusive).

    Returns
    -------
    binned : array
        Mean of values in each bin. Zero if bin is empty.
    """
    ell = np.asarray(ell)
    values = np.asarray(values)
    n_bins = len(bin_lo)
    binned = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (ell >= bin_lo[b]) & (ell <= bin_hi[b])
        if np.any(mask):
            binned[b] = np.mean(values[mask])
    return binned


def bin_spectrum_with_file(ell, cl_dict, binning_file, lmax=None):
    """Bin per-ell spectra using a BIN_ACTPOL file.

    Simple unweighted mean per bin. Only includes bins whose high edge <= lmax.
    This matches pspy's read_binning_file(binning_file, lmax) truncation.

    Parameters
    ----------
    ell : array
        Multipole values.
    cl_dict : dict
        Spectrum arrays keyed by e.g. "EE", "BB", "EB", etc.
    binning_file : str
        Path to binning file with columns (bin_low, bin_high, bin_center).
    lmax : int, optional
        Maximum multipole. Bins with high edge > lmax are excluded.
        If None, uses max(ell).

    Returns
    -------
    dict
        Has "ell" key (bin centers) and one key per input spectrum.
    """
    if lmax is None:
        lmax = int(np.max(ell))
    bin_lo, bin_hi, bin_c = load_binning_file(binning_file, lmax=lmax)
    # Preserve pspy convention: exclude bins whose high edge > lmax
    keep = bin_hi <= lmax
    bin_lo, bin_hi, bin_c = bin_lo[keep], bin_hi[keep], bin_c[keep]

    binned = {"ell": bin_c}
    for spec, cl in cl_dict.items():
        binned[spec] = bin_array(ell, cl, bin_lo, bin_hi)
    return binned
