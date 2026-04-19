"""Configuration constants for the birefringence analysis."""
import os
import multiprocessing

# ============================================================
# Analysis configuration — edit these to change what runs
# ============================================================

# Analysis mask — which likelihood to load
# "joint" or "act_dr6" (official ACT DR6 SACC with full covariance)
MASK = "joint"  # toggle by editing this line

# PP source mask: where NPIPE×NPIPE spectra come from
# "act_window"    — PP on ACT footprint (same mask as AA/AP)
# "npipe_mask"     — PP on NPIPE mask (separate from ACT footprint)
PP_MASK = "npipe_mask"
NPIPE_MASK_PERCENT = 0

# Band / frequency selection (must match what was used in likelihood_config.py)
ACT_BANDS = ["pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
NPIPE_FREQS = [30, 44, 70, 100, 143, 217, 353]  # GHz

USE_SACC_AA = True  # Use official SACC for ACT×ACT block
AP_ONLY = False      # If True, use only ACT×NPIPE cross-spectra from unified .npz

# NPIPE-only binning (only used when ACT_BANDS is empty):
#   "uniform"     — uniform bins with NPIPE_DELTA_ELL width
#   "act_binning" — use unified ACT binning file
NPIPE_ONLY_BINNING = "act_binning"
NPIPE_ELL_MIN = 51
NPIPE_ELL_MAX = 1990
NPIPE_DELTA_ELL = 20

# ACT DR6: include off-diagonal bin-to-bin covariance correlations
# (Diego-Palazuelos & Komatsu 2025, arXiv:2509.13654)
ACT_DR6_OFFDIAG_COV = False

# Sampling mode — which parameters to explore
#   "all"        — sample individual alphas + global beta via MCMC
#   "beta_only"  — fix all alphas=0, grid-search beta only
#   "alpha_only" — fix beta=0, sample alphas via MCMC
SAMPLING_MODE = "all"  # toggle by editing this line

# MCMC parameters
N_MCMC_WORKERS = int(multiprocessing.cpu_count()/2)
MCMC_N_STEPS = 70000
MCMC_N_BURN = 1000

# Filamentary dust EB model (E&K 2022, Eqs. 10-18)
# When True, adds F matrix to rotation formalism for NPIPE blocks only.
# Introduces len(DUST_ELL_BINS) free A_l amplitude parameters with flat positive priors.
USE_DUST_EB_MODEL = True
DUST_ELL_BINS = [(56, 135), (136, 215), (216, 575)]
N_DUST_BINS = len(DUST_ELL_BINS)

# ============================================================
# Priors
# ============================================================

# Beta prior half-width (degrees); flat prior on [-BETA_PRIOR_MAX_DEG, +BETA_PRIOR_MAX_DEG]
BETA_PRIOR_MAX_DEG = 10.0

# Alpha prior standard deviations (in degrees) for ACT
ACT_ALPHA_PRIOR_STD = {
    "pa4_f220": 0.09,
    "pa5_f090": 0.09,
    "pa5_f150": 0.09,
    "pa6_f090": 0.11,
    "pa6_f150": 0.11,
}

# For ACT: enable correlation between alphas within same optical tube
ACT_ALPHA_CORRELATED = True  # toggle by editing this line
ACT_ALPHA_CORRELATION_COEFF = 0.9  # correlation coefficient within same tube

# For NPIPE: alpha prior (Rosset et al. 2010, arXiv:1004.2595)
# HFI: sigma = sqrt(sigma_abs^2 + sigma_rel^2/N_PSB) = 0.31 deg
#   sigma_abs = 0.3/sqrt(3) = 0.17 deg (focal plane reference, Eq. 41)
#   sigma_rel = 0.9/sqrt(3) = 0.52 deg per PSB (ground calibration, Eq. 44)
#   N_PSB = 4 per detector split
#   rho = sigma_abs^2 / sigma_alpha^2 = 0.31 between all HFI splits
NPIPE_HFI_ALPHA_PRIOR_STD = 0.31   # degrees, per detector split
NPIPE_HFI_ALPHA_CORRELATION = 0.31 # correlation between all HFI split pairs
# LFI: 1.0 deg, independent (conservative; Leahy et al. 2010)
NPIPE_LFI_ALPHA_PRIOR_STD = 1.0    # degrees, per time/detector split

# ============================================================
# Physical constants and special treatments
# ============================================================

# ACT DR6 (official SACC) ell cut
ACT_DR6_ELL_MAX = 8500   # maximum bin center; None = use all bins


# ============================================================
# Directory paths (derived from above settings)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

COMPUTED_DIR = os.path.join(DATA_DIR, "computed")
JOINT_COMPUTED_DIR = os.path.join(COMPUTED_DIR, "joint")

