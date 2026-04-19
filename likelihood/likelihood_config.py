"""Configuration for likelihood .npz generation.

Controls which mask, bands, frequencies, and ell ranges go into
the likelihood data file used by MCMC in beta_mcmc/.
Spectrum computation settings remain in spectrum_pipeline/pipeline_config.py.

Usage: edit this file, then run `python run_likelihood.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.data_loading import (
    OUTPUT_DIR, PUBLISHED_ACT_DIR, ACT_N_SPLITS,
    ACT_MASK_SUBDIR, JOINT_SUBDIR,
)
from tools.binning import get_unified_binning_file

# ============================================================
# Plotting (null tests, PolSpice comparisons, etc.)
# ============================================================

MAKE_PLOTS = True

# ============================================================
# SACC (Eq. C9 of Louis et al. 2025)
# ============================================================

# When True: use official ACT SACC file for ACT×ACT spectra and covariance.
# Skips all per-split AA computation. Cross-block covariance (AA↔AP, AA↔PP)
# still computed from per-split Wick contractions, contracted with averaging weights.
USE_SACC_AA = True

# ============================================================
# Spectrum corrections (applied when loading spectra for likelihood)
# ============================================================

APPLY_KSPACE_TRANSFER_CORRECTION = True
APPLY_LEAKAGE_CORRECTION = True

# Covariance methods:
#   "mk"                 - Minami-Komatsu approximate 1/((2l+1)*f_sky)
#   "homogeneous_master" - homogeneous MASTER coupling kernel (matches ACT DR6 baseline,
#                          Louis et al. 2025)
COV_METHOD_ACT = "homogeneous_master"  # AA×AA, AP×AP, AA×AP, and cross-mask blocks
COV_METHOD_PP  = "mk"                  # Cov(3,3) PP×PP

# NPIPE galactic mask cut percentage (0 = full sky, 30 = 30% galactic cut)
NPIPE_MASK_PERCENT = 0
NPIPE_MASK_SUBDIR = "npipe_full_sky_mask" if NPIPE_MASK_PERCENT == 0 else f"npipe_{NPIPE_MASK_PERCENT}_mask"

# Precomputed coupling kernel .npz files (from spectrum_pipeline/compute_coupling.py)
COUPLING_DIR = os.path.join(OUTPUT_DIR, "coupling")
ACT_COUPLING_PATH = os.path.join(COUPLING_DIR, "act_mask.npz")
_coupling_suffix = f"_{NPIPE_MASK_PERCENT}" if NPIPE_MASK_PERCENT != 0 else ""
ACT_X_NPIPE_COUPLING_PATH = os.path.join(COUPLING_DIR, f"act_x_npipe{_coupling_suffix}.npz")
ACTPS_X_NPIPE_COUPLING_PATH = os.path.join(COUPLING_DIR, f"actps_x_npipe{_coupling_suffix}.npz")

# ============================================================
# Mask — determines which spectrum folder to read from
# ============================================================

# Options: "joint"
MASK = "joint"

# PP source mask: where NPIPE×NPIPE spectra come from
# "act_window"     — PP on ACT footprint (same mask as AA/AP)
# "npipe_mask"     — PP on NPIPE mask (separate from ACT footprint)
PP_MASK = "npipe_mask"


# ============================================================
# Band / frequency selection
# ============================================================

# ACT bands to include (subset of pipeline_config.ACT_ARRAY_BANDS)
# Set to [] to exclude ACT (NPIPE-only likelihood)
ACT_BANDS = ["pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
#ACT_BANDS = []
# NPIPE frequencies to include (subset of pipeline_config.NPIPE_FREQS)
# Set to [] to exclude NPIPE (ACT-only likelihood)
NPIPE_FREQS = [30, 44, 70, 100, 143, 217, 353]
#NPIPE_FREQS = [70, 100, 143, 217]
NPIPE_SPLITS = ["A", "B"]

# ============================================================
# Multipole ranges
# ============================================================

# ACT×ACT: maximum ell (last bin center, matches Table I of Diego-Palazuelos & Komatsu 2025)
ACT_X_ACT_ELL_MAX = 7525.5

# ACT×NPIPE: maximum ell (last bin center)
ACT_X_NPIPE_ELL_MAX = 2000.5

# Binning for NPIPE-only likelihood (no ACT bands):
#   "uniform"     — uniform bins from NPIPE_ELL_MIN to NPIPE_ELL_MAX with NPIPE_DELTA_ELL
#   "act_binning" — use the unified ACT binning file (delta_ell=20 below 576, then 50)
NPIPE_ONLY_BINNING = "act_binning"

# NPIPE×NPIPE (all masks): ell range and bin width
# With "act_binning", ELL_MIN and ELL_MAX filter bins via active_bins
# (bins with center < ELL_MIN or > ELL_MAX are masked out).
NPIPE_ELL_MIN = 56.5
NPIPE_ELL_MAX = 2000.5
NPIPE_DELTA_ELL = 20

# NPIPE×NPIPE on ACT footprint: minimum ell (bin center of first amended bin)
NPIPE_ACT_MASK_ELL_MIN = 80.5

# ============================================================
# Dust EB model
# ============================================================

# Compute psi_ell from 353 GHz and include in likelihood .npz
COMPUTE_PSI_ELL = True

# Per-ACT-band minimum ell for active_bins mask
ACT_ELL_MIN_PER_BAND = {
    "pa4_f220": 1000.5,
    "pa5_f090": 1000.5,
    "pa5_f150": 800.5,
    "pa6_f090": 1000.5,
    "pa6_f150": 600.5,
}
