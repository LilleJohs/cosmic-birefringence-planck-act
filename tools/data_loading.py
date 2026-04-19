"""Data paths, path resolvers, and map/beam loaders for ACT and NPIPE.

Edit the directory constants below to match your system.
"""
import os

import numpy as np
from astropy.io import fits
from pspy import pspy_utils

# ============================================================
# Project output directories (shared by spectrum_pipeline and likelihood)
# ============================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Root directory for all computed spectra (set by spectrum_pipeline)
OUTPUT_DIR     = os.path.join(_PROJECT_ROOT, "data", "computed")

# Published ACT DR6 spectra for validation
PUBLISHED_ACT_DIR = os.path.join(_PROJECT_ROOT, "data", "act_dr6")

# Number of ACT time-splits per array-band
ACT_N_SPLITS   = 4

# Output subdirectory names
ACT_MASK_SUBDIR  = "act_mask"
JOINT_SUBDIR     = "joint"

# ============================================================
# ACT DR6
# ============================================================

ACT_MAP_DIR    = "insert_path/act_dr6"
ACT_WINDOW_DIR = "insert_path/act_dr6/windows"
ACT_BEAM_DIR   = "insert_path/act_dr6/main_beams/nominal"

PS_MASK_PATH   = "insert_path/source_mask_15mJy_and_dust_rad5.fits"

KSPACE_TF_DIR        = "insert_path/act_dr6/misc/kspace_tf/dr6_binning_50"
KSPACE_TF_DIR_XPLANCK = "insert_path/act_dr6/misc_npipe/kspace_tf"

# ============================================================
# NPIPE
# ============================================================

NPIPE_MAP_DIR_A = "insert_path/npipe6v20A"
NPIPE_MAP_DIR_B = "insert_path/npipe6v20B"
NPIPE_MASK_DIR  = "insert_path/LFI_HFI_ps"
NPIPE_BEAM_DIR  = "insert_path/beam_window_functions/AB"

LEAKAGE_GAMMA_DIR = "insert_path/pspipe_utils/pspipe_utils/data/beams"

# ============================================================
# NPIPE frequency classification
# ============================================================

# Frequencies where A/B splits are time-splits (share one alpha per band)
NPIPE_TIME_SPLIT_FREQS = {30, 44}

# HFI frequencies that get the dust EB model
NPIPE_DUST_FREQS = {100, 143, 217, 353}

# ============================================================
# Path helpers — ACT
# ============================================================

def act_map_path(array_band, split_idx, product="map"):
    """Return path to an ACT split map."""
    arr, freq = array_band.split("_")
    fname = f"act_dr6.02_std_AA_night_{arr}_{freq}_4way_set{split_idx}_{product}.fits"
    return os.path.join(ACT_MAP_DIR, fname)


def act_window_path(array_band, window_type="baseline"):
    """Return path to an ACT PSpipe window file for an array-band."""
    fname = f"window_dr6_{array_band}_{window_type}.fits"
    return os.path.join(ACT_WINDOW_DIR, fname)


def act_beam_path(array_band):
    """Return path to the ACT coadd jitter_cmb beam file for an array-band."""
    arr, freq = array_band.split("_")
    fname = f"coadd_{arr}_{freq}_night_beam_tform_jitter_cmb.txt"
    return os.path.join(ACT_BEAM_DIR, fname)


# ============================================================
# Path helpers — NPIPE
# ============================================================

def npipe_map_path(freq, split):
    """Return path to a NPIPE split map."""
    base_dir = NPIPE_MAP_DIR_A if split == "A" else NPIPE_MAP_DIR_B
    fname = f"npipe6v20{split}_{freq:03d}_map_K.fits"
    return os.path.join(base_dir, fname)


def npipe_mask_path(nside, mask_percent):
    """Return path to a NPIPE analysis mask.

    Args:
        nside: HEALPix resolution (1024 or 2048).
        mask_percent: Galactic cut percentage (e.g. 30 for a 30% cut).
    """
    fname = f"comb_{mask_percent}_mask_hfi_lfi_nside_{nside}.fits"
    return os.path.join(NPIPE_MASK_DIR, fname)


def npipe_beam_path(freq1, split1, freq2, split2):
    """Return path to the NPIPE Bl_TEB beam file for a cross-spectrum pair."""
    fname = f"Bl_TEB_npipe6v20_{freq1:03d}{split1}x{freq2:03d}{split2}.fits"
    return os.path.join(NPIPE_BEAM_DIR, fname)


# ============================================================
# Loaders — ACT
# ============================================================

def load_act_beam(array_band, lmax=None):
    """Load the ACT coadd jitter_cmb beam B_ell for an array-band (e.g. "pa5_f090").

    The same beam is used for intensity and polarization: ACT DR6 provides a
    single B_ell per array-band, applied to T, E and B alike.
    """
    _, bl = pspy_utils.read_beam_file(act_beam_path(array_band), lmax=lmax)
    return bl


# ============================================================
# Loaders — NPIPE
# ============================================================

def load_npipe_beam(freq1, split1, freq2, split2, lmax=None):
    """Load NPIPE T/E/B beam transfer functions for a cross-spectrum pair.

    NPIPE Bl_TEB beam files have HDU 'WINDOW FUNCTIONS' with columns T, E, B.
    The E and B beams differ from T (and from each other) at the ~0.4% level
    at high ell, so all three are returned separately.

    Returns:
        Tuple (bl_T, bl_E, bl_B) of 1D arrays normalized to 1.0 at ell=0,
        truncated to lmax+1 entries if lmax is given.
    """
    with fits.open(npipe_beam_path(freq1, split1, freq2, split2)) as hdul:
        wf = hdul['WINDOW FUNCTIONS'].data
        bl_T = np.array(wf['T'], dtype=np.float64)
        bl_E = np.array(wf['E'], dtype=np.float64)
        bl_B = np.array(wf['B'], dtype=np.float64)

    for bl in (bl_T, bl_E, bl_B):
        if bl[0] > 0:
            bl /= bl[0]

    if lmax is not None:
        bl_T = bl_T[:lmax + 1]
        bl_E = bl_E[:lmax + 1]
        bl_B = bl_B[:lmax + 1]

    return bl_T, bl_E, bl_B
