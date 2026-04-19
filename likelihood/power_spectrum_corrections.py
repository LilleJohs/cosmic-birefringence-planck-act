"""K-space filter and T→P leakage corrections for ACT spectra.

Implements:
1. K-space filter (Eq. C3 of 2503.14452): source add-back + filter + F_b⁻¹
2. T→P leakage correction (Sec 3.3.2): subtract γ-dependent theory residual
"""
import os
import sys
from pathlib import Path

import numpy as np
from pspy import so_spectra
from pspipe_utils.kspace import deconvolve_kspace_filter_matrix
from pspipe_utils import leakage

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.spectra import camb_full_theory
from tools.data_loading import load_act_beam, KSPACE_TF_DIR, KSPACE_TF_DIR_XPLANCK, LEAKAGE_GAMMA_DIR
import likelihood_config as lcfg


def _match_ell_indices(target_ell, source_ell, tol=0.5):
    """Match multipole bins between two ell grids by proximity.

    For each source bin center, find the closest target bin within tol.

    Returns:
        (target_idx, source_idx) arrays of matched indices.
    """
    target_idx = []
    source_idx = []
    for j, ell_s in enumerate(source_ell):
        match = np.where(np.abs(target_ell - ell_s) < tol)[0]
        if len(match) == 1:
            target_idx.append(match[0])
            source_idx.append(j)
    return np.array(target_idx), np.array(source_idx)


def apply_kspace_transfer_correction(lb, ps_dict, band1, band2):
    """Apply F_b⁻¹ transfer correction to Δℓ=50 binned spectra.

    Wraps pspipe_utils.kspace.deconvolve_kspace_filter_matrix().

    Args:
        lb: 1D array of bin centers.
        ps_dict: Dict with spectrum arrays keyed by "TT", "EE", etc.
        band1: ACT array-band (e.g. "pa5_f090") or "Planck_f100".
        band2: ACT array-band or "Planck_f{freq}".

    Returns:
        Tuple (lb, corrected_ps_dict).
    """
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    # Determine which directory and file naming to use
    is_xplanck = band1.startswith("Planck") or band2.startswith("Planck")
    if is_xplanck:
        tf_dir = KSPACE_TF_DIR_XPLANCK
        # Naming: kspace_matrix_dr6_{act_band}xPlanck_f{freq}.npy
        if band1.startswith("Planck"):
            spec_name = f"dr6_{band2}x{band1}"
        else:
            spec_name = f"dr6_{band1}x{band2}"
    else:
        tf_dir = KSPACE_TF_DIR
        spec_name = f"dr6_{band1}xdr6_{band2}"

    matrix_path = f"{tf_dir}/kspace_matrix_{spec_name}.npy"
    te_corr_path = f"{tf_dir}/TE_correction_{spec_name}.dat"

    # Try reversed ordering if file not found
    if not os.path.exists(matrix_path):
        if is_xplanck:
            spec_name_rev = spec_name  # already canonical
        else:
            spec_name_rev = f"dr6_{band2}xdr6_{band1}"
        matrix_path = f"{tf_dir}/kspace_matrix_{spec_name_rev}.npy"
        te_corr_path = f"{tf_dir}/TE_correction_{spec_name_rev}.dat"

    kspace_matrix = np.load(matrix_path)
    n_kspace_bins = kspace_matrix.shape[0] // 9

    # The kspace matrix was built with a specific binning (e.g. BIN_ACTPOL_50)
    # that may differ from lb (e.g. amended binning with extra low-ell bins,
    # or different ell_max). Read kspace bin centers from TE correction file.
    # Note: TE correction entries for EE/EB/BE/BB are identically zero — we
    # only use this file to obtain the bin centers for alignment.
    lb_tf, _ = so_spectra.read_ps(te_corr_path, spectra=spectra)

    # Match spectrum bins to kspace bins by bin center value
    spec_idx, kspace_idx = _match_ell_indices(lb, lb_tf)

    if len(spec_idx) == 0:
        return lb, {s: ps_dict[s].copy() for s in spectra}

    # Fast path: all bins match 1:1 in order — no sub-blocking needed
    if (len(spec_idx) == len(lb) == n_kspace_bins and
            np.array_equal(spec_idx, np.arange(len(lb))) and
            np.array_equal(kspace_idx, np.arange(n_kspace_bins))):
        return deconvolve_kspace_filter_matrix(lb, ps_dict, kspace_matrix, spectra)

    # Extract matched sub-block from kspace matrix (9×n block structure)
    block_idx = np.concatenate([kspace_idx + s * n_kspace_bins
                                for s in range(9)])
    kspace_matrix_sub = kspace_matrix[np.ix_(block_idx, block_idx)]

    ps_matched = {s: ps_dict[s][spec_idx] for s in spectra}

    _, ps_corrected = deconvolve_kspace_filter_matrix(
        lb[spec_idx], ps_matched, kspace_matrix_sub, spectra)

    # Stitch: unmatched bins unchanged, matched bins corrected
    ps_out = {}
    for s in spectra:
        ps_out[s] = ps_dict[s].copy()
        ps_out[s][spec_idx] = ps_corrected[s]
    return lb, ps_out


def apply_kspace_transfer_correction_xplanck(lb, ps_dict, act_band, npipe_freq):
    """Apply F_b⁻¹ for ACT × NPIPE cross-spectra.

    Args:
        lb: 1D array of bin centers.
        ps_dict: Dict with spectrum arrays.
        act_band: ACT array-band (e.g. "pa5_f090").
        npipe_freq: NPIPE frequency in GHz (100, 143, 217, 353).

    Returns:
        Tuple (lb, corrected_ps_dict).
    """
    # Only 100/143/217 have dedicated TF files; LFI and 353 fall back to 100
    # (the kspace TF is ACT-only and identical across all Planck frequencies)
    freq_for_tf = npipe_freq if npipe_freq in (100, 143, 217) else 100
    planck_label = f"Planck_f{freq_for_tf}"
    return apply_kspace_transfer_correction(lb, ps_dict, act_band, planck_label)


# ============================================================
# T→P leakage correction
# ============================================================

def _load_leakage_model(array_band, lmax):
    """Load leakage model (γ and error modes) for an ACT array-band.

    Returns:
        (gamma_TE, err_modes_TE, gamma_TB, err_modes_TB):
            gamma_TE/TB: 1D arrays of shape [n_ell].
            err_modes_TE/TB: 2D arrays of shape [n_ell, n_modes].
    """
    te_file = os.path.join(LEAKAGE_GAMMA_DIR, f"{array_band}_gamma_t2e.txt")
    tb_file = os.path.join(LEAKAGE_GAMMA_DIR, f"{array_band}_gamma_t2b.txt")
    _, gamma_TE, err_TE, gamma_TB, err_TB = leakage.read_leakage_model(
        te_file, tb_file, lmax=lmax)
    return gamma_TE, err_TE, gamma_TB, err_TB


def _load_leakage_gammas(array_band, lmax):
    """Load γ_TE and γ_TB for an ACT array-band.

    Returns:
        (gamma_dict, var_dict) where gamma_dict has keys "TE", "TB"
        and var_dict has keys "TETE", "TBTB", "TETB".
    """
    gamma_TE, err_TE, gamma_TB, err_TB = _load_leakage_model(array_band, lmax)

    gamma_dict = {"TE": gamma_TE, "TB": gamma_TB}
    cov_TETE = leakage.error_modes_to_cov(err_TE)
    cov_TBTB = leakage.error_modes_to_cov(err_TB)
    var_dict = {
        "TETE": np.diag(cov_TETE),
        "TBTB": np.diag(cov_TBTB),
        "TETB": np.zeros(len(gamma_TE)),
    }
    return gamma_dict, var_dict



def _build_theory_ps_dict(lmax):
    """Build 9-spectrum theory dict in D_ℓ for leakage_correction().

    CAMB gives TT, EE, BB, TE in C_ℓ. ΛCDM has TB=BT=EB=BE=0.
    Converts to D_ℓ = ℓ(ℓ+1)/(2π) C_ℓ.
    """
    theory = camb_full_theory(ell_max=lmax)
    ells = theory["ell"]
    dl_factor = ells * (ells + 1) / (2 * np.pi)
    dl_factor[0] = 0.0

    ps_dict = {}
    for spec in ["TT", "EE", "BB", "TE"]:
        ps_dict[spec] = theory[spec] * dl_factor
    ps_dict["ET"] = ps_dict["TE"].copy()
    for spec in ["TB", "BT", "EB", "BE"]:
        ps_dict[spec] = np.zeros(len(ells))

    return ells, ps_dict


def compute_leakage_residual(band1, band2, lmax, binning_file):
    """Compute leakage correction ΔD_ℓ for a band pair.

    Uses pspipe_utils.leakage.leakage_correction() with return_residual=True.

    Spectra are always beam-convolved, so convolves theory with b_ℓ1 × b_ℓ2.
    For ACT×NPIPE: pass gamma_beta with zeros for the NPIPE side.

    Args:
        band1, band2: ACT array-bands, or "npipe_{freq}" for NPIPE.
        lmax: Maximum multipole.
        binning_file: Path to binning file for output binning.

    Returns:
        (lb, residual_dict) with binned ΔD_ℓ residuals.
    """
    lth, ps_dict_th = _build_theory_ps_dict(lmax)

    # Always beam-convolved: convolve theory with beams
    bl1 = load_act_beam(band1, lmax=lmax) if not band1.startswith("npipe") else None
    bl2 = load_act_beam(band2, lmax=lmax) if not band2.startswith("npipe") else None
    if bl1 is not None and bl2 is not None:
        beam_prod = bl1[:lmax] * bl2[:lmax]
    elif bl1 is not None:
        beam_prod = bl1[:lmax]
    elif bl2 is not None:
        beam_prod = bl2[:lmax]
    else:
        beam_prod = 1.0
    for spec in ps_dict_th:
        ps_dict_th[spec] = ps_dict_th[spec][:lmax] * beam_prod

    # Load gammas for each side
    is_npipe_1 = band1.startswith("npipe")
    is_npipe_2 = band2.startswith("npipe")

    if is_npipe_1:
        gamma_alpha = {"TE": np.zeros(lmax), "TB": np.zeros(lmax)}
        var_alpha = {"TETE": np.zeros(lmax), "TBTB": np.zeros(lmax), "TETB": np.zeros(lmax)}
    else:
        gamma_alpha, var_alpha = _load_leakage_gammas(band1, lmax)

    if is_npipe_2:
        gamma_beta = {"TE": np.zeros(lmax), "TB": np.zeros(lmax)}
    else:
        gamma_beta = {"TE": gamma_alpha["TE"], "TB": gamma_alpha["TB"]} if band2 == band1 else _load_leakage_gammas(band2, lmax)[0]

    lb, residual = leakage.leakage_correction(
        lth, ps_dict_th, gamma_alpha, var_alpha, lmax,
        gamma_beta=gamma_beta,
        return_residual=True,
        binning_file=binning_file,
    )
    return lb, residual


def apply_leakage_correction(lb, ps_dict, leakage_lb, leakage_residual):
    """Subtract leakage residual from D_ℓ spectra.

    If the leakage residual covers fewer bins than the data (e.g. due to
    lmax truncation), only bins present in the residual are corrected.

    Args:
        lb: Bin centers of observed spectra.
        ps_dict: Dict of observed D_ℓ arrays.
        leakage_lb: Bin centers of leakage residual.
        leakage_residual: Dict of ΔD_ℓ residual arrays.

    Returns:
        Dict of corrected D_ℓ arrays.
    """
    # Match leakage bins to spectrum bins by bin center value
    data_idx, leak_idx = _match_ell_indices(lb, leakage_lb)

    corrected = {}
    for spec in ps_dict:
        corrected[spec] = ps_dict[spec].copy()
        if spec in leakage_residual and len(data_idx) > 0:
            corrected[spec][data_idx] -= leakage_residual[spec][leak_idx]
    return corrected
