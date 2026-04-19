"""Compute and save pspy coupling kernels and MCM inverses to disk.

Moves the heavy pspy computations (coupling kernel via so_cov and MCM inverse
via so_mcm) out of the likelihood code, so they can be precomputed once and
loaded at analysis time.
"""
import sys
from pathlib import Path

import numpy as np
from pspy import so_cov, so_mcm

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.binning import load_binning_file


def compute_and_save_coupling(window, binning_file, lmax, output_path,
                              niter=0):
    """Compute coupling kernel and MCM inverse for same-mask (homogeneous) case.

    When a single so_map is passed to so_cov, all 32 coupling dict entries
    point to the same array. We extract and save that single unique Xi.

    Parameters
    ----------
    window : so_map
        Window function (used for both T and pol).
    binning_file : str
        Path to BIN_ACTPOL-format binning file.
    lmax : int
        Analysis maximum multipole.
    output_path : str
        Path for the output .npz file.
    niter : int
        Number of map2alm iterations.
    """
    pspy_lmax = lmax + 1
    l_exact, l_band, l_toep = 800, 2000, 2750

    print("Computing coupling kernel (same-mask)...")
    coupling = so_cov.cov_coupling_spin0and2_simple(
        window, pspy_lmax, niter=niter,
        l_exact=l_exact, l_band=l_band, l_toep=l_toep)

    # All entries are identical for single-map input; extract one.
    coupling_Xi = coupling[next(iter(coupling))]

    print("Computing MCM inverse...")
    win_tuple = (window, window)
    mbb_inv, _ = so_mcm.mcm_and_bbl_spin0and2(
        win_tuple, binning_file, pspy_lmax, niter=niter, type='Cl',
        l_exact=l_exact, l_band=l_band, l_toep=l_toep)

    bin_lo, bin_hi, bin_center = load_binning_file(binning_file, lmax=lmax)

    print(f"Saving to {output_path}")
    np.savez(output_path,
             coupling_Xi=coupling_Xi,
             mbb_inv=mbb_inv,
             lmax=lmax,
             bin_lo=bin_lo,
             bin_hi=bin_hi,
             bin_center=bin_center)


def compute_and_save_cross_coupling(window_ab, window_cd, binning_file, lmax,
                                    output_path, niter=0):
    """Compute coupling kernel and MCM inverses for cross-mask case.

    When two different masks are used, the coupling dict entries PaPcPbPd and
    PaPdPbPc differ (first and second Wick contractions). We save both, along
    with separate MCM inverses for each mask.

    Parameters
    ----------
    window_ab : so_map
        Window function for the ab group.
    window_cd : so_map
        Window function for the cd group.
    binning_file : str
        Path to BIN_ACTPOL-format binning file.
    lmax : int
        Analysis maximum multipole.
    output_path : str
        Path for the output .npz file.
    niter : int
        Number of map2alm iterations.
    """
    pspy_lmax = lmax + 1
    l_exact, l_band, l_toep = 800, 2000, 2750

    # Build window dict for cross-mask coupling
    coupling_win = {
        "Ta": window_ab, "Tb": window_ab, "Pa": window_ab, "Pb": window_ab,
        "Tc": window_cd, "Td": window_cd, "Pc": window_cd, "Pd": window_cd,
    }

    print("Computing coupling kernel (cross-mask)...")
    coupling = so_cov.cov_coupling_spin0and2_simple(
        coupling_win, pspy_lmax, niter=niter,
        l_exact=l_exact, l_band=l_band, l_toep=l_toep)

    coupling_Xi_1 = coupling["PaPcPbPd"]
    coupling_Xi_2 = coupling["PaPdPbPc"]

    print("Computing MCM inverse (window ab)...")
    win_ab_tuple = (window_ab, window_ab)
    mbb_inv_ab, _ = so_mcm.mcm_and_bbl_spin0and2(
        win_ab_tuple, binning_file, pspy_lmax, niter=niter, type='Cl',
        l_exact=l_exact, l_band=l_band, l_toep=l_toep)

    print("Computing MCM inverse (window cd)...")
    win_cd_tuple = (window_cd, window_cd)
    mbb_inv_cd, _ = so_mcm.mcm_and_bbl_spin0and2(
        win_cd_tuple, binning_file, pspy_lmax, niter=niter, type='Cl',
        l_exact=l_exact, l_band=l_band, l_toep=l_toep)

    bin_lo, bin_hi, bin_center = load_binning_file(binning_file, lmax=lmax)

    print(f"Saving to {output_path}")
    np.savez(output_path,
             coupling_Xi_1=coupling_Xi_1,
             coupling_Xi_2=coupling_Xi_2,
             mbb_inv_ab=mbb_inv_ab,
             mbb_inv_cd=mbb_inv_cd,
             lmax=lmax,
             bin_lo=bin_lo,
             bin_hi=bin_hi,
             bin_center=bin_center)


