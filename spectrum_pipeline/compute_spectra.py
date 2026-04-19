"""Cross-split pseudo-Cl computation with mode coupling.

Follows the MASTER method as implemented in pspy (for ACT small-sky)
with cross-split averaging to avoid noise bias.

Observed spectra retain beam convolution; theory spectra are beam-convolved
separately for numerical stability at high multipoles.
"""
import numpy as np
import healpy as hp
import pymaster as nmt
from pspy import so_map, so_spectra, so_mcm, sph_tools

import pipeline_config as cfg
from map_loader import (
    load_act_splits,
    load_act_split_kspace_filtered,
    load_npipe_split,
    project_npipe_to_car,
)
from tools.data_loading import load_act_beam, load_npipe_beam, act_map_path


def compute_mcm(window1, window2, binning_file, lmax=None, niter=0):
    """Compute mode coupling matrix and binning operator.

    Args:
        window1: Tuple (T_window, P_window) for field 1.
        window2: Tuple (T_window, P_window) for field 2.
        binning_file: Path to pspy binning file.
        lmax: Maximum multipole (default: LMAX from config).
        niter: Number of iterations for MCM inversion (default: 0 per PSpipe).

    Returns:
        Tuple (mbb_inv, bbl):
          - mbb_inv: dict with inverse mode coupling matrices per spectrum type.
          - bbl: dict with binning operators per spectrum type.
    """
    if lmax is None:
        lmax = cfg.ACT_X_ACT_COMPUTE_LMAX

    # Compute mode coupling matrix
    # Use binned_mcm=False to invert the full ell-by-ell MCM before binning.
    mbb_inv, bbl = so_mcm.mcm_and_bbl_spin0and2(
        win1=window1,
        binning_file=binning_file,
        lmax=lmax,
        niter=niter,
        type="Dl",  # PSpipe uses Dl for internal MCM/binning
        win2=window2,
        bl1=None,  # No beams in MCM (spectra are always beam-convolved)
        bl2=None,
        binned_mcm=False,
    )

    return mbb_inv, bbl


def _decouple_pseudocl_unbinned(cl_dict, mbb_inv, lmin, lmax):
    """Decouple pseudo-C_ℓ and return per-ell spectra (Dl).

    Applies the per-ell MCM inverse directly, then converts Cl -> Dl.

    Args:
        cl_dict: Dict with keys "TT", "TE", etc. and pseudo-C_ℓ arrays.
        mbb_inv: Inverse mode coupling matrix from compute_mcm.
        lmin: Minimum multipole for output.
        lmax: Maximum multipole.

    Returns:
        Dict with per-ell decoupled spectra (Dl) for each spectrum type.
    """
    spec_name = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    # MCM is computed for l >= 2; slice pseudo-Cl accordingly
    l = np.arange(2, lmax)
    cl_sliced = {f: cl_dict[f][l] for f in spec_name}

    _, decoupled = so_spectra.deconvolve_mode_coupling_matrix(
        l, cl_sliced, mbb_inv, spectra=spec_name)

    # Convert Cl -> Dl
    fac = l * (l + 1) / (2 * np.pi)
    Db_dict = {f: decoupled[f] * fac for f in spec_name}
    Db_dict["ell"] = l.astype(float)
    return Db_dict



def compute_cross_spectrum(alm1, alm2, mbb_inv, binning_file, lmax, lmin=2):
    """Compute one cross-spectrum from precomputed alms and MCM.

    Core function used by all pspy-based spectrum computations.

    Args:
        alm1, alm2: Precomputed windowed alms.
        mbb_inv: Inverse mode coupling matrix from compute_mcm.
        binning_file: Path to pspy binning file.
        lmax: Maximum multipole.
        lmin: Minimum multipole for unbinned output (default: 2).

    Returns:
        Dict with unbinned decoupled Dl spectra.
    """
    _ell, pseudo_cl = so_spectra.get_spectra_pixell(
        alm1, alm2,
        spectra=["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    )
    unbinned = _decouple_pseudocl_unbinned(pseudo_cl, mbb_inv, lmin, lmax)
    return unbinned


def compute_all_cross_spectra(alms1, alms2, labels1, labels2,
                              mbb_inv, binning_file, lmax, lmin=2):
    """Compute cross-spectra for all (i,j) pairs from two sets of alms.

    Args:
        alms1: List of alms for field 1 (e.g. ACT splits or NPIPE splits).
        alms2: List of alms for field 2.
        labels1: List of labels for alms1 (e.g. [0,1,2,3] or ["A","B"]).
        labels2: List of labels for alms2.
        mbb_inv: Single inverse MCM (one MCM for all pairs, using coadd beams).
        binning_file: Path to pspy binning file.
        lmax: Maximum multipole.
        lmin: Minimum multipole for unbinned output (default: 2).

    Returns:
        Dict with keys:
          - "cross_spectra": list of (label1, label2, unbinned_dict) tuples
          - "ell": ell array from first spectrum
    """
    cross_spectra = []
    for i, l1 in enumerate(labels1):
        for j, l2 in enumerate(labels2):
            unbinned = compute_cross_spectrum(
                alms1[i], alms2[j], mbb_inv, binning_file, lmax, lmin)
            cross_spectra.append((l1, l2, unbinned))

    print(f"  ✓ Computed {len(cross_spectra)} cross-spectra")

    return {
        "cross_spectra": cross_spectra,
        "ell": cross_spectra[0][2]["ell"],
    }


def compute_act_x_npipe_spectrum(act_array_band, npipe_freq, npipe_maps,
                                  act_window, npipe_window,
                                  binning_file, lmax=None, niter=None,
                                  act_alms=None, npipe_alms=None):
    """Compute ACT x NPIPE cross-spectra for all split pairs (4 ACT x 2 NPIPE = 8).

    Beam handling follows PSpipe (get_mcm_and_bbl.py): one MCM per array pair.
    No beams in MCM (spectra are always beam-convolved).

    Args:
        act_array_band: e.g. "pa5_f090"
        npipe_freq: NPIPE frequency in GHz (100, 143, 217)
        npipe_maps: Dict mapping split label ("A", "B") to projected so_map.
        act_window: Tuple (T_window, P_window) for ACT field.
        npipe_window: Tuple (T_window, P_window) for NPIPE field.
        binning_file: Path to pspy binning file.
        lmax: Maximum multipole (default: ACT_X_NPIPE_COMPUTE_LMAX).
        niter: Number of iterations for map2alm (default: ACT_X_NPIPE_NITER).
        act_alms: Precomputed ACT alms (list of 4). If None, computed from maps.
        npipe_alms: Precomputed NPIPE alms (dict by split). If None, computed from maps.

    Returns:
        Dict with "cross_spectra", "ell".
    """
    if lmax is None:
        lmax = cfg.ACT_X_NPIPE_COMPUTE_LMAX
    if niter is None:
        niter = cfg.ACT_X_NPIPE_NITER

    # One MCM for all split pairs (no beams)
    print(f"  Computing MCM for {act_array_band} x npipe_{npipe_freq}...")
    mbb_inv, _ = compute_mcm(act_window, npipe_window, binning_file,
                              lmax=lmax, niter=niter)

    # Compute alms if not precomputed
    if act_alms is None:
        if cfg.APPLY_KSPACE_FILTER:
            print(f"  Loading ACT {act_array_band} splits with k-space filter...")
            act_splits = [load_act_split_kspace_filtered(act_array_band, i)
                          for i in range(cfg.ACT_N_SPLITS)]
        else:
            print(f"  Loading ACT {act_array_band} splits...")
            act_splits = load_act_splits(act_array_band)
        act_alms = [sph_tools.get_alms(m, act_window, niter=niter, lmax=lmax)
                     for m in act_splits]

    if npipe_alms is None:
        print(f"  Computing NPIPE alms (projected to CAR)...")
        npipe_alms = {}
        for npipe_split in cfg.NPIPE_SPLITS:
            npipe_map = npipe_maps[npipe_split]
            npipe_alms[npipe_split] = sph_tools.get_alms(
                npipe_map, npipe_window, niter=niter, lmax=lmax)

    # Convert npipe_alms dict to list for compute_all_cross_spectra
    npipe_alms_list = [npipe_alms[s] for s in cfg.NPIPE_SPLITS]

    print(f"  Computing {cfg.ACT_N_SPLITS}x{len(cfg.NPIPE_SPLITS)} cross-spectra...")
    return compute_all_cross_spectra(
        act_alms, npipe_alms_list,
        list(range(cfg.ACT_N_SPLITS)), list(cfg.NPIPE_SPLITS),
        mbb_inv, binning_file, lmax, lmin=2,
    )


def compute_npipe_spectrum(freq1, split1, freq2, split2, mask, lmax=None):
    """Compute NPIPE cross-spectrum using NaMaster directly.

    Args:
        freq1: Frequency in GHz (100, 143, 217, or 353).
        split1: Split label ("A" or "B").
        freq2: Second frequency.
        split2: Second split.
        mask: HEALPix mask (temperature mask; same used for polarization).
        lmax: Maximum multipole (default: NPIPE_COMPUTE_LMAX from config).

    Returns:
        Dict with unbinned C_ℓ for all 9 spectra (TT, TE, TB, ..., BB).
    """
    if lmax is None:
        lmax = cfg.NPIPE_COMPUTE_LMAX

    # Load maps
    map1_somap = load_npipe_split(freq1, split1)
    map2_somap = load_npipe_split(freq2, split2)
    map1 = map1_somap.data.copy()  # (3, npix): I, Q, U
    map2 = map2_somap.data.copy()

    # Match all maps and mask to the same nside (use the highest resolution)
    nside1 = hp.npix2nside(map1.shape[1])
    nside2 = hp.npix2nside(map2.shape[1])
    mask_nside = hp.npix2nside(mask.shape[0])
    target_nside = max(nside1, nside2, mask_nside)

    if nside1 != target_nside:
        map1 = np.array([hp.ud_grade(map1[i], target_nside) for i in range(3)])
    if nside2 != target_nside:
        map2 = np.array([hp.ud_grade(map2[i], target_nside) for i in range(3)])
    if mask_nside != target_nside:
        mask = hp.ud_grade(mask, target_nside)
        mask = np.clip(mask, 0.0, 1.0)

    # Subtract monopole and dipole from temperature maps (matching PolSpice subav+subdipole)
    # NPIPE maps have residual monopole/dipole that leak to low-ell through mask coupling
    for m in [map1, map2]:
        m[0] = hp.remove_dipole(m[0], bad=0.0)

    # No beams in NaMaster fields (spectra are always beam-convolved)
    # Create NaMaster fields (spin-0 for T, spin-2 for E/B)
    # Pass raw (unmasked) maps — NaMaster applies the mask internally.
    f1_0 = nmt.NmtField(mask, [map1[0]], lmax=lmax, lmax_mask=lmax)  # T
    f1_2 = nmt.NmtField(mask, [map1[1], map1[2]], lmax=lmax,
                         lmax_mask=lmax)  # Q, U

    f2_0 = nmt.NmtField(mask, [map2[0]], lmax=lmax, lmax_mask=lmax)
    f2_2 = nmt.NmtField(mask, [map2[1], map2[2]], lmax=lmax,
                         lmax_mask=lmax)  # Q, U

    # Compute workspace for mode coupling (unbinned, so use flat binning)
    b = nmt.NmtBin.from_lmax_linear(lmax, 1, is_Dell=False)

    # TT
    w00 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(f1_0, f2_0, b)
    cl_coupled_TT = nmt.compute_coupled_cell(f1_0, f2_0)
    cl_TT = w00.decouple_cell(cl_coupled_TT)[0]

    # TE, TB, ET, BT (spin-0 x spin-2)
    w02 = nmt.NmtWorkspace()
    w02.compute_coupling_matrix(f1_0, f2_2, b)
    cl_coupled_T_pol = nmt.compute_coupled_cell(f1_0, f2_2)
    cl_T_pol = w02.decouple_cell(cl_coupled_T_pol)  # [TE, TB]
    cl_TE = cl_T_pol[0]
    cl_TB = cl_T_pol[1]

    w20 = nmt.NmtWorkspace()
    w20.compute_coupling_matrix(f1_2, f2_0, b)
    cl_coupled_pol_T = nmt.compute_coupled_cell(f1_2, f2_0)
    cl_pol_T = w20.decouple_cell(cl_coupled_pol_T)  # [ET, BT]
    cl_ET = cl_pol_T[0]
    cl_BT = cl_pol_T[1]

    # EE, EB, BE, BB (spin-2 x spin-2)
    w22 = nmt.NmtWorkspace()
    w22.compute_coupling_matrix(f1_2, f2_2, b)
    cl_coupled_pol = nmt.compute_coupled_cell(f1_2, f2_2)
    cl_pol = w22.decouple_cell(cl_coupled_pol)

    # Construct ell array from spectrum length
    ells_out = np.arange(len(cl_TT))

    # Return unbinned C_ℓ in μK²
    unbinned = {
        "ell": ells_out,
        "TT": cl_TT,
        "TE": cl_TE,
        "ET": cl_ET,
        "TB": cl_TB,
        "BT": cl_BT,
        "EE": cl_pol[0],
        "EB": cl_pol[1],
        "BE": cl_pol[2],
        "BB": cl_pol[3],
    }

    print(f"  ✓ NPIPE unbinned spectrum computed (NaMaster) for {freq1}{split1} x {freq2}{split2}")
    return unbinned


def compute_act_cross_band_spectrum(band1, band2, window1, window2,
                                     binning_file, lmax=None, niter=None,
                                     alms1=None, alms2=None):
    """Compute ACT cross-band spectra for all 4x4 split pairs.

    Args:
        band1, band2: e.g. "pa5_f090", "pa5_f150"
        window1, window2: Tuple (T_window, P_window) for each band.
        binning_file: Path to pspy binning file.
        lmax: Maximum multipole.
        niter: Number of iterations for map2alm.
        alms1, alms2: Precomputed alms (list of 4 each). If None, computed from maps.

    Returns:
        Dict with "cross_spectra", "ell".
    """
    if lmax is None:
        lmax = cfg.ACT_X_NPIPE_COMPUTE_LMAX
    if niter is None:
        niter = cfg.ACT_X_NPIPE_NITER

    n_splits = cfg.ACT_N_SPLITS

    # One MCM for all split pairs (no beams)
    print(f"  Computing MCM for {band1} x {band2}...")
    mbb_inv, _ = compute_mcm(window1, window2, binning_file, lmax=lmax, niter=niter)

    # Compute alms if not precomputed
    if alms1 is None:
        if cfg.APPLY_KSPACE_FILTER:
            print(f"  Loading ACT {band1} splits with k-space filter...")
            splits1 = [load_act_split_kspace_filtered(band1, i) for i in range(n_splits)]
        else:
            print(f"  Loading ACT {band1} splits...")
            splits1 = load_act_splits(band1)
        alms1 = [sph_tools.get_alms(m, window1, niter=niter, lmax=lmax) for m in splits1]

    if alms2 is None:
        if cfg.APPLY_KSPACE_FILTER:
            print(f"  Loading ACT {band2} splits with k-space filter...")
            splits2 = [load_act_split_kspace_filtered(band2, i) for i in range(n_splits)]
        else:
            print(f"  Loading ACT {band2} splits...")
            splits2 = load_act_splits(band2)
        alms2 = [sph_tools.get_alms(m, window2, niter=niter, lmax=lmax) for m in splits2]

    labels = list(range(n_splits))
    print(f"  Computing {n_splits}x{n_splits} cross-spectra...")
    return compute_all_cross_spectra(
        alms1, alms2, labels, labels,
        mbb_inv, binning_file, lmax, lmin=2,
    )


def compute_npipe_cross_on_act_footprint(freq1, split1, freq2, split2,
                                          npipe_window, binning_file,
                                          lmax=None, niter=None,
                                          alm1=None, alm2=None,
                                          mbb_inv=None):
    """Compute single NPIPE cross-spectrum on the ACT footprint using pspy.

    Uses the ACT window so NPIPE×NPIPE spectra are on the same footprint
    as ACT×NPIPE for consistent joint analysis.

    Args:
        freq1, freq2: NPIPE frequency in GHz.
        split1, split2: Split label ("A" or "B").
        npipe_window: Tuple (T_window, P_window).
        binning_file: Path to pspy binning file.
        lmax: Maximum multipole.
        niter: Number of iterations for map2alm.
        alm1, alm2: Pre-computed windowed alms. If None, computed from maps.
        mbb_inv: Pre-computed inverse MCM. If None, computed here.

    Returns:
        Dict with unbinned Dl spectra.
    """
    if lmax is None:
        lmax = cfg.ACT_X_NPIPE_COMPUTE_LMAX
    if niter is None:
        niter = cfg.ACT_X_NPIPE_NITER

    # Load and project NPIPE maps if alms not precomputed
    if alm1 is None or alm2 is None:
        # CAR geometry template for reprojecting NPIPE HEALPix maps onto the ACT grid
        act_template = so_map.read_map(act_map_path(cfg.ACT_ARRAY_BANDS[0], 0))

    if alm1 is None:
        npipe_map1 = project_npipe_to_car(freq1, split1, act_template)
        alm1 = sph_tools.get_alms(npipe_map1, npipe_window, niter=niter, lmax=lmax)
        del npipe_map1

    if alm2 is None:
        npipe_map2 = project_npipe_to_car(freq2, split2, act_template)
        alm2 = sph_tools.get_alms(npipe_map2, npipe_window, niter=niter, lmax=lmax)
        del npipe_map2

    if mbb_inv is None:
        mbb_inv, _ = compute_mcm(npipe_window, npipe_window, binning_file, lmax=lmax, niter=niter)

    unbinned = compute_cross_spectrum(
        alm1, alm2, mbb_inv, binning_file, lmax, lmin=2)

    print(f"  ✓ NPIPE×NPIPE on ACT footprint: {freq1}{split1} × {freq2}{split2}")
    return unbinned
