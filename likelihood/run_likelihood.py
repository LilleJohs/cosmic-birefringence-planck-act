"""Entry point for likelihood data generation.

Reads per-split spectra from disk, computes the Gaussian covariance,
bins spectra, and saves everything needed for MCMC to a single .npz
file per analysis type.

Usage: edit likelihood_config.py, then run `python run_likelihood.py`.
"""
import os
import sys
from pathlib import Path

import numpy as np
import sacc
from pspy import so_spectra, so_map

sys.path.insert(0, str(Path(__file__).parent.parent))

import likelihood_config as lcfg
from tools.data_loading import (
    NPIPE_TIME_SPLIT_FREQS, NPIPE_DUST_FREQS,
    load_act_beam, KSPACE_TF_DIR, KSPACE_TF_DIR_XPLANCK, npipe_mask_path,
)
from tools.naming import build_npz_filename
from tools.binning import load_binning_file, bin_array
from power_spectrum_corrections import (
    apply_kspace_transfer_correction,
    apply_kspace_transfer_correction_xplanck,
    compute_leakage_residual,
    apply_leakage_correction,
)
from covariance import (
    build_detector_spectrum_arrays, compute_ek_covariance,
    load_coupling_data, bin_detector_spectra,
)
from covariance_corrections import apply_kspace_to_covariance, compute_leakage_covariance
from compute_psi_ell import compute_psi_ell
from spectrum_loader import COL_EE, COL_EB, COL_BE, COL_BB
from tools.spectra import dell_to_cell, camb_theory, bin_theory_with_window
from tools.masks import get_act_analysis_window, compute_fsky
from validate import run_null_tests


# ============================================================
# Active bins helper
# ============================================================

def _build_active_bins(cross_spec_list, bin_centers, det_to_band,
                       ell_max_aa, ell_max_ap, ell_max_pp,
                       ell_min_per_band, ell_min_npipe,
                       act_band_set=None, n_aa_ap=None,
                       ell_min_pp=None):
    """Build per-cross boolean active_bins mask.

    Args:
        cross_spec_list: List of (det1, det2) tuples.
        bin_centers: [n_bins] array.
        det_to_band: Dict mapping detector labels to band names.
        ell_max_aa, ell_max_ap, ell_max_pp: Per-type ell_max.
        ell_min_per_band: Dict mapping band name -> minimum ell.
        ell_min_npipe: Default ell_min for NPIPE detectors on ACT mask.
        act_band_set: Set of ACT band names. If None, derived from
            det_to_band values. Needed for averaged mode where det_to_band
            maps bands to themselves, or unified mode.
        n_aa_ap: If not None, cross-spectra at index >= n_aa_ap are PP
            (block 3) and use ell_min_pp as their ell_min.
        ell_min_pp: ell_min for PP crosses (unified mode).

    Returns:
        active_bins: [n_cross, n_bins] bool array.
    """
    if act_band_set is None:
        act_band_set = set(det_to_band.values())

    n_cross = len(cross_spec_list)
    n_bins = len(bin_centers)
    active_bins = np.zeros((n_cross, n_bins), dtype=bool)

    for k, (det1, det2) in enumerate(cross_spec_list):
        is_act_1 = det1 in det_to_band or det1 in act_band_set
        is_act_2 = det2 in det_to_band or det2 in act_band_set

        # ell_min
        if n_aa_ap is not None and k >= n_aa_ap:
            ell_min = ell_min_pp
        else:
            band1 = det_to_band.get(det1, det1 if det1 in act_band_set else "")
            ell_min_1 = ell_min_per_band.get(
                band1, ell_min_npipe if not band1 else 0)
            band2 = det_to_band.get(det2, det2 if det2 in act_band_set else "")
            ell_min_2 = ell_min_per_band.get(
                band2, ell_min_npipe if not band2 else 0)
            ell_min = max(ell_min_1, ell_min_2)

        # ell_max
        if n_aa_ap is not None and k >= n_aa_ap:
            cross_ell_max = ell_max_pp
        elif is_act_1 and is_act_2:
            cross_ell_max = ell_max_aa
        elif is_act_1 or is_act_2:
            cross_ell_max = ell_max_ap
        else:
            cross_ell_max = ell_max_pp

        active_bins[k] = (bin_centers >= ell_min) & (bin_centers <= cross_ell_max)

    return active_bins


# ============================================================
# Covariance stitching helper
# ============================================================

def _stitch_block_covariance(n_bins, block_sizes, diag_blocks, cross_blocks):
    """Stitch sub-block covariances into a full symmetric matrix.

    Args:
        n_bins: Number of output bins (first axis).
        block_sizes: List of block sizes [n_b1, n_b2, ...].
        diag_blocks: List of (cov_ee, cov_bb, cov_eb) tuples for each diagonal
                     block, or None to skip. Shape [n_bins_block, n_bi, n_bi].
        cross_blocks: Dict mapping (i, j) -> (cov_ee, cov_bb, cov_eb) for
                      off-diagonal blocks (i < j). Shape [n_bins_block, n_bi, n_bj].

    Returns:
        (cov_ee, cov_bb, cov_eb): Each [n_bins, n_cross, n_cross].
    """
    n_cross = sum(block_sizes)
    cov_ee = np.zeros((n_bins, n_cross, n_cross))
    cov_bb = np.zeros((n_bins, n_cross, n_cross))
    cov_eb = np.zeros((n_bins, n_cross, n_cross))
    out = [cov_ee, cov_bb, cov_eb]

    # Compute block boundary slices
    starts = []
    s = 0
    for sz in block_sizes:
        starts.append(s)
        s += sz

    # Diagonal blocks
    for i, blk in enumerate(diag_blocks):
        if blk is None or block_sizes[i] == 0:
            continue
        si = starts[i]
        ei = si + block_sizes[i]
        nb = min(blk[0].shape[0], n_bins)
        for c in range(3):
            out[c][:nb, si:ei, si:ei] = blk[c][:nb]

    # Off-diagonal blocks (symmetric)
    for (i, j), blk in cross_blocks.items():
        if block_sizes[i] == 0 or block_sizes[j] == 0:
            continue
        si, ei = starts[i], starts[i] + block_sizes[i]
        sj, ej = starts[j], starts[j] + block_sizes[j]
        nb = min(blk[0].shape[0], n_bins)
        for c in range(3):
            out[c][:nb, si:ei, sj:ej] = blk[c][:nb]
            out[c][:nb, sj:ej, si:ei] = np.transpose(blk[c][:nb], (0, 2, 1))

    return cov_ee, cov_bb, cov_eb


# ============================================================
# Label and index helpers (shared across likelihood builders)
# ============================================================

def _build_act_detector_labels(act_bands, n_splits):
    """Build ACT detector labels: ['band_set0', 'band_set1', ...]."""
    return [f"{band}_set{i}" for band in act_bands for i in range(n_splits)]


def _build_npipe_detector_labels(npipe_freqs, npipe_splits):
    """Build NPIPE detector labels: ['100A', '100B', ...]."""
    return [f"{freq}{split}" for freq in npipe_freqs for split in npipe_splits]


def _build_npipe_alpha_labels(npipe_freqs, npipe_splits):
    """Build NPIPE alpha labels, sharing one alpha for time-split freqs."""
    alpha_labels = []
    for freq in npipe_freqs:
        if freq in NPIPE_TIME_SPLIT_FREQS:
            alpha_labels.append(str(freq))
        else:
            for split in npipe_splits:
                alpha_labels.append(f"{freq}{split}")
    return alpha_labels


def _build_cross_idx_map(cross_spec_list, detector_map):
    """Build [n_cross, 2] index array from cross-spectrum pair list."""
    n_cross = len(cross_spec_list)
    cross_idx_map = np.zeros((n_cross, 2), dtype=np.int32)
    for k, (det1, det2) in enumerate(cross_spec_list):
        cross_idx_map[k, 0] = detector_map[det1]
        cross_idx_map[k, 1] = detector_map[det2]
    return cross_idx_map


def _build_partitioned_cross_list(detector_labels, det_to_band):
    """Build cross-spectrum pairs partitioned into AA, AP, PP blocks.

    Returns:
        (aa_cross, ap_cross, pp_cross): Lists of (det_i, det_j) tuples
            for ACT×ACT, ACT×NPIPE, and NPIPE×NPIPE pairs respectively.
    """
    aa_cross = []
    ap_cross = []
    pp_cross = []
    for det_i in detector_labels:
        for det_j in detector_labels:
            if det_i == det_j:
                continue
            is_act_i = det_i in det_to_band
            is_act_j = det_j in det_to_band
            if is_act_i and is_act_j:
                aa_cross.append((det_i, det_j))
            elif is_act_i or is_act_j:
                ap_cross.append((det_i, det_j))
            else:
                pp_cross.append((det_i, det_j))
    return aa_cross, ap_cross, pp_cross


def _load_unbinned_and_rebin_data_vectors(data_dir, cross_spec_list,
                                          find_file_fn, bin_edges):
    """Load unbinned spectra and bin onto a target binning grid.

    Reads _obs_unbinned.dat and _theory_unbinned.dat files (per-ell C_ell)
    and averages them into bins defined by bin_edges.

    Args:
        data_dir: Directory containing spectrum files.
        cross_spec_list: List of cross-spectrum keys.
        find_file_fn: Callable(det1, det2, suffix) -> (path, swapped) or None.
        bin_edges: 1D array of n_bins+1 bin edges from _parse_binning_file.

    Returns:
        (C_obs_3d, C_theory_3d, bin_centers):
            C_obs_3d: [n_bins, n_cross, 3] (EE, BB, EB)
            C_theory_3d: [n_bins, n_cross, 2] (EE, BB)
            bin_centers: [n_bins] midpoints of each bin
    """
    n_bins = len(bin_edges) - 1
    n_cross = len(cross_spec_list)
    C_obs_3d = np.zeros((n_bins, n_cross, 3))
    C_theory_3d = np.zeros((n_bins, n_cross, 2))
    # Centers must match the binning file: (bin_lo + bin_hi_inclusive) / 2
    # bin_edges are [lo_0, lo_1, ..., hi_n_exclusive], so hi_inclusive = edge - 1
    bin_centers = 0.5 * (bin_edges[:-1] + (bin_edges[1:] - 1))

    file_cache = {}
    _bin_lo = bin_edges[:-1]
    _bin_hi = bin_edges[1:] - 1  # convert exclusive edges to inclusive upper bounds

    for k, (det1, det2) in enumerate(cross_spec_list):
        # Observed spectrum (unbinned)
        obs_path, swapped = find_file_fn(det1, det2, "obs_unbinned.dat")
        if obs_path is None:
            raise FileNotFoundError(
                f"Observed spectrum file not found for ({det1}, {det2})")
        if obs_path not in file_cache:
            file_cache[obs_path] = np.loadtxt(obs_path, comments='#')
        obs = file_cache[obs_path]
        ell_obs = obs[:, 0]
        col_eb = COL_BE if swapped else COL_EB
        C_obs_3d[:, k, 0] = bin_array(ell_obs, obs[:, COL_EE], _bin_lo, _bin_hi)
        C_obs_3d[:, k, 1] = bin_array(ell_obs, obs[:, COL_BB], _bin_lo, _bin_hi)
        C_obs_3d[:, k, 2] = bin_array(ell_obs, obs[:, col_eb], _bin_lo, _bin_hi)

        # Theory spectrum (unbinned)
        theory_path, _ = find_file_fn(det1, det2, "theory_unbinned.dat")
        if theory_path is None:
            raise FileNotFoundError(
                f"Theory spectrum file not found for ({det1}, {det2})")
        if theory_path not in file_cache:
            file_cache[theory_path] = np.loadtxt(theory_path, comments='#')
        theory = file_cache[theory_path]
        ell_th = theory[:, 0]
        C_theory_3d[:, k, 0] = bin_array(ell_th, theory[:, COL_EE], _bin_lo, _bin_hi)
        C_theory_3d[:, k, 1] = bin_array(ell_th, theory[:, COL_BB], _bin_lo, _bin_hi)

    return C_obs_3d, C_theory_3d, bin_centers




def _load_unbinned_auto_cross(data_dir, detector_labels, act_bands, n_act_splits,
                               npipe_freqs, npipe_splits, ell_max):
    """Load per-ell (unbinned) auto and cross spectra for covariance computation.

    Reads _obs_unbinned.dat files (Cl format, per-ell).

    Returns:
        (auto_spectra, cross_spectra, ell_arr)
    """
    # Get ell grid from first available auto-spectrum
    if act_bands:
        sample_path = os.path.join(data_dir, f"act_{act_bands[0]}_set0xact_{act_bands[0]}_set0_obs_unbinned.dat")
    else:
        freq0, split0 = npipe_freqs[0], npipe_splits[0]
        sample_path = os.path.join(data_dir, f"npipe_{freq0}{split0}xnpipe_{freq0}{split0}_obs_unbinned.dat")
    sample = np.loadtxt(sample_path, comments='#')
    all_ell = sample[:, 0]
    ell_mask = all_ell <= ell_max
    ell_arr = all_ell[ell_mask]
    n_ell = len(ell_arr)

    auto_spectra = {}
    cross_spectra = {}

    def _load(path):
        data = np.loadtxt(path, comments='#')
        mask = data[:, 0] <= ell_max
        ee_raw = data[mask, COL_EE]
        bb_raw = data[mask, COL_BB]
        if len(ee_raw) >= n_ell:
            return ee_raw[:n_ell], bb_raw[:n_ell]
        # File has fewer ells than the reference grid (e.g. NPIPE files
        # when ell_max exceeds their range). Pad with zeros — these ells
        # will be excluded via active_bins in the likelihood.
        ee = np.zeros(n_ell)
        bb = np.zeros(n_ell)
        ee[:len(ee_raw)] = ee_raw
        bb[:len(bb_raw)] = bb_raw
        return ee, bb

    # ACT auto-spectra
    for band in act_bands:
        for i in range(n_act_splits):
            path = os.path.join(data_dir, f"act_{band}_set{i}xact_{band}_set{i}_obs_unbinned.dat")
            ee, bb = _load(path)
            auto_spectra[f"{band}_set{i}"] = {'EE': ee, 'BB': bb}

    # NPIPE auto-spectra
    if npipe_freqs is not None:
        for freq in npipe_freqs:
            for split in npipe_splits:
                path = os.path.join(data_dir, f"npipe_{freq}{split}xnpipe_{freq}{split}_obs_unbinned.dat")
                ee, bb = _load(path)
                auto_spectra[f"{freq}{split}"] = {'EE': ee, 'BB': bb}

    # ACT cross-split spectra
    for band in act_bands:
        for i in range(n_act_splits):
            for j in range(i + 1, n_act_splits):
                path = os.path.join(data_dir, f"act_{band}_set{i}xact_{band}_set{j}_obs_unbinned.dat")
                ee, bb = _load(path)
                cross_spectra[(f"{band}_set{i}", f"{band}_set{j}")] = {'EE': ee, 'BB': bb}

    # ACT cross-band spectra
    for ib, band1 in enumerate(act_bands):
        for band2 in act_bands[ib + 1:]:
            for i in range(n_act_splits):
                for j in range(n_act_splits):
                    path = os.path.join(data_dir, f"act_{band1}_set{i}xact_{band2}_set{j}_obs_unbinned.dat")
                    ee, bb = _load(path)
                    cross_spectra[(f"{band1}_set{i}", f"{band2}_set{j}")] = {'EE': ee, 'BB': bb}

    # ACT x NPIPE cross-spectra
    if npipe_freqs is not None:
        for band in act_bands:
            for freq in npipe_freqs:
                for split in npipe_splits:
                    for i_act in range(n_act_splits):
                        path = os.path.join(data_dir, f"act_{band}_set{i_act}xnpipe_{freq}{split}_obs_unbinned.dat")
                        ee, bb = _load(path)
                        cross_spectra[(f"{band}_set{i_act}", f"{freq}{split}")] = {'EE': ee, 'BB': bb}

    # NPIPE cross-spectra (all distinct split map pairs: cross-split and cross-frequency)
    if npipe_freqs is not None:
        npipe_dets = [(freq, split) for freq in npipe_freqs for split in npipe_splits]
        for i, (freq1, s1) in enumerate(npipe_dets):
            for freq2, s2 in npipe_dets[i + 1:]:
                path = os.path.join(data_dir, f"npipe_{freq1}{s1}xnpipe_{freq2}{s2}_obs_unbinned.dat")
                if not os.path.exists(path):
                    path = os.path.join(data_dir, f"npipe_{freq2}{s2}xnpipe_{freq1}{s1}_obs_unbinned.dat")
                ee, bb = _load(path)
                cross_spectra[(f"{freq1}{s1}", f"{freq2}{s2}")] = {'EE': ee, 'BB': bb}

    print(f"  Loaded {len(auto_spectra)} auto + {len(cross_spectra)} cross unbinned spectra")
    return auto_spectra, cross_spectra, ell_arr


def _load_kspace_F_inv(tf_dir, spec_name, n_bins, bin_centers):
    """Load k-space transfer matrix and compute F^-1 EE/EB/BB sub-block.

    The pspipe kspace matrix is [9*n_bins_file, 9*n_bins_file], block-diagonal
    in ell. Spectra ordering: TT=0, TE=1, TB=2, ET=3, BT=4, EE=5, EB=6, BE=7, BB=8.
    Index for spectrum s, bin b: s * n_bins_file + b.

    We extract the 3x3 (EE, EB, BB) sub-block per bin and invert.
    Bins are aligned by matching ell values from the TE correction file,
    since the kspace matrix and the spectrum may use different binnings.

    Args:
        tf_dir: Directory containing kspace_matrix_*.npy files.
        spec_name: e.g. "dr6_pa5_f090xdr6_pa5_f150"
        n_bins: Expected number of output bins.
        bin_centers: [n_bins] array of output bin center multipoles.

    Returns:
        [n_bins, 3, 3] array of F^-1 sub-blocks (EE, EB, BB ordering).
    """
    matrix_path = os.path.join(tf_dir, f"kspace_matrix_{spec_name}.npy")
    te_corr_path = os.path.join(tf_dir, f"TE_correction_{spec_name}.dat")
    if not os.path.exists(matrix_path):
        parts = spec_name.split('x', 1)
        if len(parts) == 2:
            spec_name_rev = f"{parts[1]}x{parts[0]}"
            matrix_path = os.path.join(tf_dir, f"kspace_matrix_{spec_name_rev}.npy")
            te_corr_path = os.path.join(tf_dir, f"TE_correction_{spec_name_rev}.dat")
    F_flat = np.load(matrix_path)  # [9*n_bins_file, 9*n_bins_file]

    n_bins_file = F_flat.shape[0] // 9

    # Read kspace bin centers from TE correction file
    spectra_9 = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    lb_tf, _ = so_spectra.read_ps(te_corr_path, spectra=spectra_9)

    # Match output bins to kspace bins by ell value
    # Start with identity (no correction) for all bins
    F_inv_sub = np.tile(np.eye(3), (n_bins, 1, 1))

    sub_spec = [5, 6, 8]  # EE=5, EB=6, BB=8

    for j, ell_k in enumerate(lb_tf):
        match = np.where(np.abs(bin_centers - ell_k) < 0.5)[0]
        if len(match) == 1:
            idx = [s * n_bins_file + j for s in sub_spec]
            F_sub = F_flat[np.ix_(idx, idx)]
            F_inv_sub[match[0]] = np.linalg.inv(F_sub)

    return F_inv_sub


def _parse_binning_file(binning_file, ell_max):
    """Parse BIN_ACTPOL_50 file into bin edges and centers arrays.

    ell_max can be a bin center (.5 value) or a bin high edge — any bin
    whose low edge <= ell_max is included, and bin edges are never clipped.

    Returns:
        bin_edges: 1D array of n_bins+1 edges.
        bin_centers: 1D array of n_bins bin centers (from file column 3).
    """
    bin_lo, bin_hi, bin_ctr = load_binning_file(binning_file, lmax=ell_max)
    bin_lo = bin_lo.astype(int)
    bin_hi = bin_hi.astype(int)
    # Build edges array: [lo_0, lo_1, ..., lo_n, hi_n]
    bin_edges = np.append(bin_lo, bin_hi[-1])
    return bin_edges, bin_ctr


def _det_file_prefix(det_label):
    """Return file-name prefix for a detector label.

    ACT detector 'pa5_f090_set0' -> 'act_pa5_f090_set0'
    NPIPE detector '100A' -> 'npipe_100A'
    """
    if '_set' in det_label:
        return f"act_{det_label}"
    return f"npipe_{det_label}"


def _find_cross_file(data_dir, det1, det2, suffix):
    """Find spectrum file for a cross-spectrum pair, trying both orderings.

    Returns:
        (path, swapped): path to file and whether det ordering was swapped.
        (None, False) if no file found.
    """
    p1 = _det_file_prefix(det1)
    p2 = _det_file_prefix(det2)
    path = os.path.join(data_dir, f"{p1}x{p2}_{suffix}")
    if os.path.exists(path):
        return path, False
    path_swap = os.path.join(data_dir, f"{p2}x{p1}_{suffix}")
    if os.path.exists(path_swap):
        return path_swap, True
    return None, False


def _build_per_cross_F_inv_joint(cross_spec_list, det_to_band, n_bins,
                                 bin_centers):
    """Build [n_cross, n_bins, 3, 3] k-space F_inv for unified cross-spectrum list.

    ACT x ACT: load from KSPACE_TF_DIR (band pair)
    ACT x NPIPE: load from KSPACE_TF_DIR_XPLANCK (ACT band x Planck freq)
    NPIPE x NPIPE: identity (no k-space filter)
    """
    n_cross = len(cross_spec_list)
    F_inv_all = np.tile(np.eye(3), (n_cross, n_bins, 1, 1))

    tf_cache = {}
    for k, (det1, det2) in enumerate(cross_spec_list):
        is_act1 = det1 in det_to_band
        is_act2 = det2 in det_to_band

        if not is_act1 and not is_act2:
            continue  # NPIPE x NPIPE: keep identity

        if is_act1 and is_act2:
            band1 = det_to_band[det1]
            band2 = det_to_band[det2]
            spec_name = f"dr6_{band1}xdr6_{band2}"
            tf_dir = KSPACE_TF_DIR
        else:
            act_det = det1 if is_act1 else det2
            npipe_det = det2 if is_act1 else det1
            band = det_to_band[act_det]
            freq_str = ''.join(c for c in npipe_det if c.isdigit())
            # Only f100, f143, f217 TF files exist; others default to f100
            # (kspace matrices are identical across Planck bands)
            tf_freq = int(freq_str) if int(freq_str) not in [30, 44, 70, 353] else 100
            spec_name = f"dr6_{band}xPlanck_f{tf_freq}"
            tf_dir = KSPACE_TF_DIR_XPLANCK

        cache_key = (tf_dir, spec_name)
        if cache_key not in tf_cache:
            tf_cache[cache_key] = _load_kspace_F_inv(tf_dir, spec_name, n_bins,
                                                     bin_centers)
        F_inv_all[k] = tf_cache[cache_key]

    return F_inv_all


def _save_npz(out_path, cov_ee, cov_bb, cov_eb, C_obs_3d, C_theory_3d,
              bin_centers, cross_idx_map, detector_labels, alpha_labels,
              analysis_type, cross_spec_list,
              band_labels=None, det_to_band=None, act_bands=None,
              active_bins=None, npipe_freqs=None, npipe_splits=None,
              psi_ell_binned=None,
              act_alpha_labels=None, npipe_alpha_labels=None,
              dust_cross_mask=None):
    """Save likelihood data to .npz file."""
    n_bins = len(bin_centers)
    n_cross = cross_idx_map.shape[0]

    save_dict = {
        'cov_ee': cov_ee,
        'cov_bb': cov_bb,
        'cov_eb': cov_eb,
        'C_obs_3d': C_obs_3d,
        'C_theory_3d': C_theory_3d,
        'bin_centers': bin_centers,
        'cross_idx_map': cross_idx_map,
        'detector_labels': np.array(detector_labels, dtype=str),
        'alpha_labels': np.array(alpha_labels, dtype=str),
        'n_bins': n_bins,
        'n_cross': n_cross,
        'analysis_type': analysis_type,
    }

    # Serialize cross_spec_list as component arrays
    if analysis_type == "npipe":
        # Tuples: (freq1, split1, freq2, split2)
        freqs1 = np.array([s[0] for s in cross_spec_list], dtype=np.int32)
        splits1 = np.array([s[1] for s in cross_spec_list], dtype=str)
        freqs2 = np.array([s[2] for s in cross_spec_list], dtype=np.int32)
        splits2 = np.array([s[3] for s in cross_spec_list], dtype=str)
        save_dict['cross_freqs1'] = freqs1
        save_dict['cross_splits1'] = splits1
        save_dict['cross_freqs2'] = freqs2
        save_dict['cross_splits2'] = splits2

    elif analysis_type in ("joint_act_mask", "unified", "joint"):
        # Tuples: (det1_label, det2_label)
        cross_det1 = np.array([s[0] for s in cross_spec_list], dtype=str)
        cross_det2 = np.array([s[1] for s in cross_spec_list], dtype=str)
        save_dict['cross_det1'] = cross_det1
        save_dict['cross_det2'] = cross_det2

    # Optional fields
    if band_labels is not None:
        save_dict['band_labels'] = np.array(band_labels, dtype=str)
    if det_to_band is not None:
        save_dict['det_to_band_keys'] = np.array(list(det_to_band.keys()), dtype=str)
        save_dict['det_to_band_vals'] = np.array(list(det_to_band.values()), dtype=str)
    if act_bands is not None:
        save_dict['act_bands'] = np.array(act_bands, dtype=str)
    if active_bins is not None:
        save_dict['active_bins'] = active_bins
    if npipe_freqs is not None:
        save_dict['npipe_freqs'] = np.array(npipe_freqs, dtype=np.int32)
    if npipe_splits is not None:
        save_dict['npipe_splits'] = np.array(npipe_splits, dtype=str)
    if psi_ell_binned is not None:
        save_dict['psi_ell_binned'] = psi_ell_binned
    if dust_cross_mask is not None:
        save_dict['dust_cross_mask'] = dust_cross_mask
    if act_alpha_labels is not None:
        save_dict['act_alpha_labels'] = np.array(act_alpha_labels, dtype=str)
    if npipe_alpha_labels is not None:
        save_dict['npipe_alpha_labels'] = np.array(npipe_alpha_labels, dtype=str)

    np.savez(out_path, **save_dict)


# ============================================================
# Averaged spectra helpers (Eq. C9, Louis et al. 2025)
# ============================================================


def _load_sacc_aa_block(bands, bin_centers_target):
    """Load ACT×ACT spectra and covariance from official SACC file.

    Reads all 25 ordered cross-pairs from the SACC file in alphabetical
    (band_e, band_b) order — matching the ordering produced by
    _build_averaged_cross_spec_lists. For each pair (bi, bj):
      - bi <= bj: extract cl_eb(bi, bj) directly
      - bi > bj:  extract cl_be(bj, bi), which gives EB(bi, bj)

    Converts D_ℓ → C_ℓ, beam-convolves (SACC is beam-deconvolved, but our
    pipeline uses beam-convolved spectra), and maps onto the target binning grid.

    Args:
        bands: List of ACT band names, e.g. ["pa5_f090", ...].
        bin_centers_target: 1D array of target bin centers (amended binning).

    Returns:
        Dict with keys:
            cross_pairs: list of (band_e, band_b) tuples (25 pairs)
            C_obs_3d: [n_target_bins, 25, 3] (EE, BB, EB) — zero-padded
            cov_ee, cov_bb, cov_eb: [n_target_bins, 25, 25] — zero-padded
            active_bins: [25, n_target_bins] bool mask
            sacc_bin_mask: [n_target_bins] bool — which target bins have SACC data
    """
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data")
    sacc_path = os.path.join(data_dir, "act_dr6", "v1.0", "dr6_data.fits")
    s = sacc.Sacc.load_fits(sacc_path)
    full_cov = s.covariance.covmat

    bands_sorted = sorted(set(bands))

    # Per-band minimum multipole (matches Diego-Palazuelos & Komatsu 2025)
    band_ell_min = {
        "pa4_f220": 1000.5, "pa5_f090": 1000.5, "pa5_f150": 800.5,
        "pa6_f090": 1000.5, "pa6_f150": 600.5,
    }

    # SACC tracer name for each band
    def _tracer(band):
        return f"dr6_{band}_s2"

    cross_pairs = []
    ee_indices, bb_indices, eb_indices = [], [], []
    ee_data, bb_data, eb_data = [], [], []
    ell_sacc = None

    # Loop in alphabetical (bi, bj) order — matches _build_averaged_cross_spec_lists
    for bi in bands_sorted:
        for bj in bands_sorted:
            if bi <= bj:
                # Upper triangle + diagonal: stored as cl_eb(bi, bj)
                t_i, t_j = _tracer(bi), _tracer(bj)
                ell, cl_ee, ind_ee = s.get_ell_cl(
                    'cl_ee', t_i, t_j, return_ind=True)
                _, cl_bb, ind_bb = s.get_ell_cl(
                    'cl_bb', t_i, t_j, return_ind=True)
                _, cl_eb, ind_eb = s.get_ell_cl(
                    'cl_eb', t_i, t_j, return_ind=True)
            else:
                # Lower triangle: stored as cl_be(bj, bi)
                # BE(bj, bi) = <B_bj E_bi> = EB(bi, bj)
                t_i, t_j = _tracer(bj), _tracer(bi)
                ell, cl_ee, ind_ee = s.get_ell_cl(
                    'cl_ee', t_i, t_j, return_ind=True)
                _, cl_bb, ind_bb = s.get_ell_cl(
                    'cl_bb', t_i, t_j, return_ind=True)
                _, cl_eb, ind_eb = s.get_ell_cl(
                    'cl_be', t_i, t_j, return_ind=True)

            if ell_sacc is None:
                ell_sacc = ell
            cross_pairs.append((bi, bj))
            ee_indices.append(ind_ee)
            bb_indices.append(ind_bb)
            eb_indices.append(ind_eb)
            ee_data.append(dell_to_cell(ell, cl_ee))
            bb_data.append(dell_to_cell(ell, cl_bb))
            eb_data.append(dell_to_cell(ell, cl_eb))

    n_sacc_cross = len(cross_pairs)
    n_sacc_bins = len(ell_sacc)
    n_target_bins = len(bin_centers_target)

    # Map SACC bins onto target bins (find matching bin centers)
    # Some SACC bins may not match (e.g. low-ℓ bins with different binning);
    # these are skipped (masked out by active_bins anyway).
    sacc_to_target = np.full(n_sacc_bins, -1, dtype=int)
    n_matched = 0
    for i_s, ell_s in enumerate(ell_sacc):
        diffs = np.abs(bin_centers_target - ell_s)
        best = np.argmin(diffs)
        if diffs[best] < 1.0:  # must match within 1 ell
            sacc_to_target[i_s] = best
            n_matched += 1

    # Extract per-bin covariance from SACC (in D_ell space), convert to C_ell
    dell_to_cell_factor = 2.0 * np.pi / (ell_sacc * (ell_sacc + 1.0))

    # Build output arrays on target grid (zero-padded)
    C_obs_3d = np.zeros((n_target_bins, n_sacc_cross, 3))
    cov_ee = np.zeros((n_target_bins, n_sacc_cross, n_sacc_cross))
    cov_bb = np.zeros((n_target_bins, n_sacc_cross, n_sacc_cross))
    cov_eb = np.zeros((n_target_bins, n_sacc_cross, n_sacc_cross))
    sacc_bin_mask = np.zeros(n_target_bins, dtype=bool)

    for i_s in range(n_sacc_bins):
        i_t = sacc_to_target[i_s]
        if i_t < 0:
            continue  # no match in target binning
        sacc_bin_mask[i_t] = True
        f = dell_to_cell_factor[i_s]
        f2 = f * f

        for k in range(n_sacc_cross):
            C_obs_3d[i_t, k, 0] = ee_data[k][i_s]
            C_obs_3d[i_t, k, 1] = bb_data[k][i_s]
            C_obs_3d[i_t, k, 2] = eb_data[k][i_s]

        for m in range(n_sacc_cross):
            for n in range(n_sacc_cross):
                cov_ee[i_t, m, n] = full_cov[
                    ee_indices[m][i_s], ee_indices[n][i_s]] * f2
                cov_bb[i_t, m, n] = full_cov[
                    bb_indices[m][i_s], bb_indices[n][i_s]] * f2
                cov_eb[i_t, m, n] = full_cov[
                    eb_indices[m][i_s], eb_indices[n][i_s]] * f2

    # Beam-convolve SACC spectra and covariance.
    # SACC spectra are beam-deconvolved (PSpipe convention), but our pipeline
    # works with beam-convolved spectra. Multiply by bl_e * bl_b per cross-pair.
    beam_lmax = int(np.max(ell_sacc)) + 2
    beam_prods = np.zeros((n_sacc_cross, n_sacc_bins))
    for k, (band_e, band_b) in enumerate(cross_pairs):
        bl_e = load_act_beam(band_e, lmax=beam_lmax)
        bl_b = load_act_beam(band_b, lmax=beam_lmax)
        for i_s in range(n_sacc_bins):
            ell_int = int(round(ell_sacc[i_s]))
            beam_prods[k, i_s] = bl_e[ell_int] * bl_b[ell_int]

    for i_s in range(n_sacc_bins):
        i_t = sacc_to_target[i_s]
        if i_t < 0:
            continue
        for k in range(n_sacc_cross):
            C_obs_3d[i_t, k, :] *= beam_prods[k, i_s]
        for m in range(n_sacc_cross):
            for n in range(n_sacc_cross):
                bp = beam_prods[m, i_s] * beam_prods[n, i_s]
                cov_ee[i_t, m, n] *= bp
                cov_bb[i_t, m, n] *= bp
                cov_eb[i_t, m, n] *= bp

    # Active bins: per-pair ell_min + SACC coverage
    active_bins = np.zeros((n_sacc_cross, n_target_bins), dtype=bool)
    for k, (band_e, band_b) in enumerate(cross_pairs):
        pair_ell_min = max(band_ell_min[band_e], band_ell_min[band_b])
        active_bins[k] = sacc_bin_mask & (bin_centers_target >= pair_ell_min)

    print(f"  SACC AA block: {n_sacc_cross} cross-pairs, "
          f"{n_matched}/{n_sacc_bins} bins matched to target grid (beam-convolved)")

    return {
        'cross_pairs': cross_pairs,
        'C_obs_3d': C_obs_3d,
        'cov_ee': cov_ee,
        'cov_bb': cov_bb,
        'cov_eb': cov_eb,
        'active_bins': active_bins,
        'sacc_bin_mask': sacc_bin_mask,
    }


def _build_averaged_cross_spec_lists(act_bands, npipe_det_labels):
    """Build cross-spectrum lists for averaged mode.

    Args:
        act_bands: List of ACT band names, e.g. ["pa5_f090", ...].
        npipe_det_labels: List of NPIPE detector labels, e.g. ["100A", "100B", ...].

    Returns:
        (b1_avg, b2_avg, b3_avg): Lists of (det_e, det_b) tuples.
            b1_avg: 25 ordered ACT band pairs (same structure as SACC).
            b2_avg: ACT band × NPIPE det (both orderings).
            b3_avg: NPIPE × NPIPE (unchanged from per-split).
    """
    bands_sorted = sorted(set(act_bands))

    # B1: All ordered band pairs (i, j) for i != j, plus same-band (i, i).
    # For cross-band: both orderings (i,j) and (j,i) are distinct in the
    # birefringence likelihood because alpha_i and alpha_j enter differently
    # in the E vs B positions.
    # For same-band: only one entry (i,i) since EB(i,i) = EB(i,i).
    b1_avg = []
    for bi in bands_sorted:
        for bj in bands_sorted:
            b1_avg.append((bi, bj))  # all 25 ordered pairs

    # B2: ACT band × NPIPE det (both orderings)
    b2_avg = []
    for band in act_bands:
        for npipe_det in npipe_det_labels:
            b2_avg.append((band, npipe_det))
            b2_avg.append((npipe_det, band))

    # B3: NPIPE × NPIPE (same as per-split)
    b3_avg = []
    for det_i in npipe_det_labels:
        for det_j in npipe_det_labels:
            if det_i != det_j:
                b3_avg.append((det_i, det_j))

    return b1_avg, b2_avg, b3_avg


def _build_averaging_weights(cross_spec_list_splits, cross_spec_list_avg,
                             det_to_band, n_act_splits):
    """Build averaging weight matrix W mapping per-split to band-averaged spectra.

    For each averaged cross-pair, identifies contributing split pairs and
    assigns weight 1/n_cross per Eq. C9 (Louis et al. 2025).

    Args:
        cross_spec_list_splits: Per-split cross-spectrum list [(det1, det2), ...].
        cross_spec_list_avg: Averaged cross-spectrum list [(band1/det, band2/det), ...].
        det_to_band: Dict mapping split labels to band names (ACT only).
        n_act_splits: Number of ACT splits (4).

    Returns:
        W: [n_avg, n_splits] averaging weight matrix.
    """
    n_avg = len(cross_spec_list_avg)
    n_splits = len(cross_spec_list_splits)
    W = np.zeros((n_avg, n_splits))

    # Build index for fast lookup
    split_index = {pair: k for k, pair in enumerate(cross_spec_list_splits)}

    for i_avg, (avg_e, avg_b) in enumerate(cross_spec_list_avg):
        is_act_e = avg_e in det_to_band or avg_e in set(det_to_band.values())
        is_act_b = avg_b in det_to_band or avg_b in set(det_to_band.values())

        # Determine which split pairs contribute
        contributing = []

        if is_act_e and is_act_b:
            # ACT × ACT: band_e × band_b
            band_e, band_b = avg_e, avg_b
            for si in range(n_act_splits):
                for sj in range(n_act_splits):
                    if si == sj:
                        continue  # exclude same-index (Eq. C9: 1-δ_{ij})
                    det_e = f"{band_e}_set{si}"
                    det_b = f"{band_b}_set{sj}"
                    # For EB, both orderings are distinct (X≠Y in Eq. C9),
                    # so n_cross = n_d*(n_d-1) = 12 for all band pairs.
                    pair = (det_e, det_b)
                    if pair in split_index:
                        contributing.append(split_index[pair])

        elif is_act_e and not is_act_b:
            # ACT band × NPIPE det: average over ACT splits
            band_e = avg_e
            npipe_det = avg_b
            for si in range(n_act_splits):
                det_e = f"{band_e}_set{si}"
                pair = (det_e, npipe_det)
                if pair in split_index:
                    contributing.append(split_index[pair])

        elif not is_act_e and is_act_b:
            # NPIPE det × ACT band: average over ACT splits
            npipe_det = avg_e
            band_b = avg_b
            for sj in range(n_act_splits):
                det_b = f"{band_b}_set{sj}"
                pair = (npipe_det, det_b)
                if pair in split_index:
                    contributing.append(split_index[pair])

        else:
            # NPIPE × NPIPE: no averaging, 1:1 mapping
            pair = (avg_e, avg_b)
            if pair in split_index:
                contributing.append(split_index[pair])

        if not contributing:
            raise ValueError(
                f"No split pairs found for averaged pair ({avg_e}, {avg_b})")

        weight = 1.0 / len(contributing)
        for idx in contributing:
            W[i_avg, idx] = weight

    return W


def _contract_covariance(cov_splits, W_A, W_B):
    """Contract per-split covariance with averaging weight matrices.

    cov_avg[b] = W_A @ cov_splits[b] @ W_B.T

    Args:
        cov_splits: [n_bins, n_splits_A, n_splits_B] per-split covariance.
        W_A: [n_avg_A, n_splits_A] weight matrix for group A.
        W_B: [n_avg_B, n_splits_B] weight matrix for group B.

    Returns:
        [n_bins, n_avg_A, n_avg_B] averaged covariance.
    """
    # Two batched matmuls use BLAS (multithreaded) instead of single-threaded einsum
    return np.matmul(np.matmul(W_A, cov_splits), W_B.T)


def _apply_split_averaging(cov_ee, cov_bb, cov_eb, C_obs_3d, C_theory_3d,
                            b1_splits, b2_splits, b3_splits,
                            act_bands, npipe_det_labels,
                            det_to_band, n_act_splits,
                            sacc_aa=None, bin_centers=None):
    """Average per-split spectra and covariance into band-level pairs.

    Contracts AA and AP blocks using Eq. C9 averaging weights;
    PP block is unchanged (NPIPE splits kept separate).

    Args:
        cov_ee, cov_bb, cov_eb: [n_bins, n_cross, n_cross] full covariance.
        C_obs_3d: [n_bins, n_cross, 3] observed spectra.
        C_theory_3d: [n_bins, n_cross, 2] theory spectra.
        b1_splits, b2_splits, b3_splits: Per-split cross lists for AA, AP, PP.
        act_bands: List of ACT band names.
        npipe_det_labels: List of NPIPE detector labels.
        det_to_band: Dict mapping split labels to band names.
        n_act_splits: Number of ACT splits.
        sacc_aa: If not None, dict with 'cov_ee', 'cov_bb', 'cov_eb',
            'C_obs_3d', 'active_bins' from SACC for AA block replacement.
        bin_centers: [n_bins] array, required when sacc_aa is not None.

    Returns:
        Dict with keys: cov_ee, cov_bb, cov_eb, C_obs_3d, C_theory_3d,
            cross_spec_list, det_labels, det_to_band, cross_idx_map,
            act_alpha_labels, n_b1, n_b2, n_b3.
    """
    n_bins = cov_ee.shape[0]
    n_b1 = len(b1_splits)
    n_b2 = len(b2_splits)
    n_b3 = len(b3_splits)
    s1, e1 = 0, n_b1
    s2, e2 = n_b1, n_b1 + n_b2
    s3, e3 = n_b1 + n_b2, n_b1 + n_b2 + n_b3

    # Build averaged cross-spec lists
    b1_avg, b2_avg, b3_avg = _build_averaged_cross_spec_lists(
        act_bands, npipe_det_labels)
    n_b1_avg = len(b1_avg)
    n_b2_avg = len(b2_avg)
    n_b3_avg = len(b3_avg)
    cross_spec_list_avg = b1_avg + b2_avg + b3_avg
    n_cross_avg = len(cross_spec_list_avg)

    print(f"    Averaged cross-spectra: B1={n_b1_avg}, B2={n_b2_avg}, "
          f"B3={n_b3_avg}, total={n_cross_avg}")

    # ── AA block ──
    print("    Contracting AA block covariance...")
    W_AA = None
    if sacc_aa is not None:
        cov_ee_b1_avg = sacc_aa['cov_ee']
        cov_bb_b1_avg = sacc_aa['cov_bb']
        cov_eb_b1_avg = sacc_aa['cov_eb']
        C_obs_b1_avg = sacc_aa['C_obs_3d']

        # Theory for AA: beam-convolved CAMB per cross-pair
        ell_max_th = int(np.max(bin_centers)) + 10
        ells_th, EE_th, BB_th = camb_theory(ell_max=ell_max_th)
        C_theory_b1_avg = np.zeros((n_bins, n_b1_avg, 2))
        for k, (band_e, band_b) in enumerate(sacc_aa['cross_pairs']):
            bl_e = load_act_beam(band_e, lmax=ell_max_th)
            bl_b = load_act_beam(band_b, lmax=ell_max_th)
            n_bl = min(len(bl_e), len(bl_b), len(EE_th))
            beam_prod = bl_e[:n_bl] * bl_b[:n_bl]
            C_theory_b1_avg[:, k, 0] = bin_theory_with_window(
                ells_th[:n_bl], EE_th[:n_bl] * beam_prod, bin_centers.astype(int))
            C_theory_b1_avg[:, k, 1] = bin_theory_with_window(
                ells_th[:n_bl], BB_th[:n_bl] * beam_prod, bin_centers.astype(int))
    else:
        W_AA = _build_averaging_weights(
            b1_splits, b1_avg, det_to_band, n_act_splits)
        cov_ee_b1_avg = _contract_covariance(
            cov_ee[:, s1:e1, s1:e1], W_AA, W_AA)
        cov_bb_b1_avg = _contract_covariance(
            cov_bb[:, s1:e1, s1:e1], W_AA, W_AA)
        cov_eb_b1_avg = _contract_covariance(
            cov_eb[:, s1:e1, s1:e1], W_AA, W_AA)

        C_obs_b1_avg = np.zeros((n_bins, n_b1_avg, 3))
        for s in range(3):
            C_obs_b1_avg[:, :, s] = np.einsum(
                'ij,bj->bi', W_AA, C_obs_3d[:, s1:e1, s])
        C_theory_b1_avg = np.zeros((n_bins, n_b1_avg, 2))
        for s in range(2):
            C_theory_b1_avg[:, :, s] = np.einsum(
                'ij,bj->bi', W_AA, C_theory_3d[:, s1:e1, s])

    # ── AP block ──
    print("    Contracting AP block covariance...")
    if n_b2 > 0:
        W_AP = _build_averaging_weights(
            b2_splits, b2_avg, det_to_band, n_act_splits)
        cov_ee_b2_avg = _contract_covariance(
            cov_ee[:, s2:e2, s2:e2], W_AP, W_AP)
        cov_bb_b2_avg = _contract_covariance(
            cov_bb[:, s2:e2, s2:e2], W_AP, W_AP)
        cov_eb_b2_avg = _contract_covariance(
            cov_eb[:, s2:e2, s2:e2], W_AP, W_AP)

        C_obs_b2_avg = np.zeros((n_bins, n_b2_avg, 3))
        for s in range(3):
            C_obs_b2_avg[:, :, s] = np.einsum(
                'ij,bj->bi', W_AP, C_obs_3d[:, s2:e2, s])
        C_theory_b2_avg = np.zeros((n_bins, n_b2_avg, 2))
        for s in range(2):
            C_theory_b2_avg[:, :, s] = np.einsum(
                'ij,bj->bi', W_AP, C_theory_3d[:, s2:e2, s])
    else:
        W_AP = np.zeros((0, 0))
        C_obs_b2_avg = np.zeros((n_bins, 0, 3))
        C_theory_b2_avg = np.zeros((n_bins, 0, 2))

    # ── PP block (unchanged) ──
    W_PP = np.eye(n_b3) if n_b3 > 0 else np.zeros((0, 0))
    C_obs_b3_avg = C_obs_3d[:, s3:e3, :]
    C_theory_b3_avg = C_theory_3d[:, s3:e3, :]
    cov_ee_b3_avg = cov_ee[:, s3:e3, s3:e3]
    cov_bb_b3_avg = cov_bb[:, s3:e3, s3:e3]
    cov_eb_b3_avg = cov_eb[:, s3:e3, s3:e3]

    # ── Cross-block covariance ──
    # Need W_AA for cross-blocks even when SACC replaced AA diagonal
    if W_AA is None and n_b1 > 0 and (n_b2 > 0 or n_b3 > 0):
        W_AA = _build_averaging_weights(
            b1_splits, b1_avg, det_to_band, n_act_splits)

    print("    Contracting cross-block covariance...")
    if n_b1 > 0 and n_b2 > 0:
        cov_ee_b12_avg = _contract_covariance(
            cov_ee[:, s1:e1, s2:e2], W_AA, W_AP)
        cov_bb_b12_avg = _contract_covariance(
            cov_bb[:, s1:e1, s2:e2], W_AA, W_AP)
        cov_eb_b12_avg = _contract_covariance(
            cov_eb[:, s1:e1, s2:e2], W_AA, W_AP)

    if n_b1 > 0 and n_b3 > 0:
        cov_ee_b13_avg = _contract_covariance(
            cov_ee[:, s1:e1, s3:e3], W_AA, W_PP)
        cov_bb_b13_avg = _contract_covariance(
            cov_bb[:, s1:e1, s3:e3], W_AA, W_PP)
        cov_eb_b13_avg = _contract_covariance(
            cov_eb[:, s1:e1, s3:e3], W_AA, W_PP)

    if n_b2 > 0 and n_b3 > 0:
        cov_ee_b23_avg = _contract_covariance(
            cov_ee[:, s2:e2, s3:e3], W_AP, W_PP)
        cov_bb_b23_avg = _contract_covariance(
            cov_bb[:, s2:e2, s3:e3], W_AP, W_PP)
        cov_eb_b23_avg = _contract_covariance(
            cov_eb[:, s2:e2, s3:e3], W_AP, W_PP)

    # ── Concatenate data vectors ──
    C_obs_avg = np.concatenate(
        [C_obs_b1_avg, C_obs_b2_avg, C_obs_b3_avg], axis=1)
    C_theory_avg = np.concatenate(
        [C_theory_b1_avg, C_theory_b2_avg, C_theory_b3_avg], axis=1)

    # ── Reassemble covariance ──
    print("    Stitching block covariance...")
    diag = [
        (cov_ee_b1_avg, cov_bb_b1_avg, cov_eb_b1_avg),
        (cov_ee_b2_avg, cov_bb_b2_avg, cov_eb_b2_avg) if n_b2_avg > 0 else None,
        (cov_ee_b3_avg, cov_bb_b3_avg, cov_eb_b3_avg) if n_b3_avg > 0 else None,
    ]
    cross = {}
    if n_b1_avg > 0 and n_b2_avg > 0:
        cross[(0, 1)] = (cov_ee_b12_avg, cov_bb_b12_avg, cov_eb_b12_avg)
    if n_b1_avg > 0 and n_b3_avg > 0:
        cross[(0, 2)] = (cov_ee_b13_avg, cov_bb_b13_avg, cov_eb_b13_avg)
    if n_b2_avg > 0 and n_b3_avg > 0:
        cross[(1, 2)] = (cov_ee_b23_avg, cov_bb_b23_avg, cov_eb_b23_avg)
    cov_ee_out, cov_bb_out, cov_eb_out = _stitch_block_covariance(
        n_bins, [n_b1_avg, n_b2_avg, n_b3_avg], diag, cross)

    # Build averaged metadata
    det_labels_avg = list(act_bands) + list(npipe_det_labels)
    det_map_avg = {label: idx for idx, label in enumerate(det_labels_avg)}
    cross_idx_map_avg = _build_cross_idx_map(cross_spec_list_avg, det_map_avg)
    det_to_band_avg = {band: band for band in act_bands}

    print(f"    Averaged: {len(cross_spec_list_avg)} cross-spectra, "
          f"{len(det_labels_avg)} split maps")

    return {
        'cov_ee': cov_ee_out, 'cov_bb': cov_bb_out, 'cov_eb': cov_eb_out,
        'C_obs_3d': C_obs_avg, 'C_theory_3d': C_theory_avg,
        'cross_spec_list': cross_spec_list_avg,
        'det_labels': det_labels_avg,
        'det_to_band': det_to_band_avg,
        'cross_idx_map': cross_idx_map_avg,
        'act_alpha_labels': list(act_bands),
        'n_b1': n_b1_avg, 'n_b2': n_b2_avg, 'n_b3': n_b3_avg,
        'sacc_active_bins': sacc_aa['active_bins'] if sacc_aa else None,
    }


def _get_band_pair(det1, det2, det_to_band):
    """Extract band pair from detector labels.

    ACT splits (e.g. "pa5_f090_set0") → band from det_to_band.
    NPIPE detectors (e.g. "100A") → "npipe_{freq}".

    Returns:
        (band1, band2, is_act1, is_act2)
    """
    is_act1 = det1 in det_to_band
    is_act2 = det2 in det_to_band
    if is_act1:
        band1 = det_to_band[det1]
    else:
        freq = ''.join(c for c in det1 if c.isdigit())
        band1 = f"npipe_{freq}"
    if is_act2:
        band2 = det_to_band[det2]
    else:
        freq = ''.join(c for c in det2 if c.isdigit())
        band2 = f"npipe_{freq}"
    return band1, band2, is_act1, is_act2


def _apply_spectrum_corrections(C_obs_3d, bin_centers, cross_spec_list,
                                det_to_band, binning_file, ell_max):
    """Apply kspace and leakage corrections to data vectors.

    Correction flow: kspace F_b⁻¹ → leakage

    Corrections are per-band-pair (same for all splits of a given band pair).
    Spectra are converted to D_ℓ, corrected using the tested functions from
    power_spectrum_corrections.py, then converted back to C_ℓ.

    Only ACT-involved cross-spectra are corrected:
      - ACT×ACT: kspace + leakage
      - ACT×NPIPE: kspace + leakage (zeros for NPIPE side)
      - NPIPE×NPIPE: no corrections

    Args:
        C_obs_3d: [n_bins, n_cross, 3] beam-convolved C_ℓ (EE=0, BB=1, EB=2).
        bin_centers: [n_bins] bin center multipoles.
        cross_spec_list: list of (det1, det2) tuples.
        det_to_band: dict mapping ACT detector labels to band names.
        binning_file: path to binning file.
        ell_max: maximum multipole.

    Returns:
        Corrected C_obs_3d (copy).
    """
    spectra_9 = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    n_bins, n_cross, _ = C_obs_3d.shape
    C_corr = C_obs_3d.copy()

    # C_ℓ ↔ D_ℓ conversion at bin centers
    cl_to_dl = bin_centers * (bin_centers + 1.0) / (2.0 * np.pi)
    cl_to_dl[bin_centers == 0] = 0.0
    dl_to_cl = np.where(cl_to_dl > 0, 1.0 / cl_to_dl, 0.0)

    n_corrected_kspace = 0
    n_corrected_leakage = 0

    # Cache per band pair: store corrected D_ℓ deltas so each unique
    # band pair is computed only once
    correction_cache = {}

    for k, (det1, det2) in enumerate(cross_spec_list):
        band1, band2, is_act1, is_act2 = _get_band_pair(det1, det2, det_to_band)

        # Skip NPIPE×NPIPE — no corrections needed
        if not is_act1 and not is_act2:
            continue

        # Pack this cross-spectrum into a 9-spectrum D_ℓ dict
        # (T-spectra set to zero — kspace EE/EB/BB sub-block is independent)
        ps_dl = {s: np.zeros(n_bins) for s in spectra_9}
        ps_dl["EE"] = C_corr[:, k, 0] * cl_to_dl
        ps_dl["BB"] = C_corr[:, k, 1] * cl_to_dl
        ps_dl["EB"] = C_corr[:, k, 2] * cl_to_dl

        # ── 1. Kspace F_b⁻¹ (multiplicative) ──
        if lcfg.APPLY_KSPACE_TRANSFER_CORRECTION:
            if is_act1 and is_act2:
                _, ps_dl = apply_kspace_transfer_correction(
                    bin_centers, ps_dl, band1, band2)
            else:
                act_band = band1 if is_act1 else band2
                npipe_band = band2 if is_act1 else band1
                npipe_freq = int(npipe_band.replace("npipe_", ""))
                _, ps_dl = apply_kspace_transfer_correction_xplanck(
                    bin_centers, ps_dl, act_band, npipe_freq)
            n_corrected_kspace += 1

        # ── 2. Leakage subtraction ──
        if lcfg.APPLY_LEAKAGE_CORRECTION:
            cache_key = ("leakage", band1, band2)
            if cache_key not in correction_cache:
                correction_cache[cache_key] = compute_leakage_residual(
                    band1, band2, ell_max, binning_file)
            lb_leak, residual = correction_cache[cache_key]
            ps_dl = apply_leakage_correction(
                bin_centers, ps_dl, lb_leak, residual)
            n_corrected_leakage += 1

        # Convert corrected D_ℓ back to C_ℓ
        C_corr[:, k, 0] = ps_dl["EE"] * dl_to_cl
        C_corr[:, k, 1] = ps_dl["BB"] * dl_to_cl
        C_corr[:, k, 2] = ps_dl["EB"] * dl_to_cl

    print(f"  Spectrum corrections applied: "
          f"kspace={n_corrected_kspace}, "
          f"leakage={n_corrected_leakage}")

    return C_corr


def save_likelihood_data():
    """Compute and save unified-covariance likelihood data.

    Places all cross-spectra into one data vector with a single covariance:
      - Block 1 (AA): ACT×ACT crosses (ACT mask)
      - Block 2 (AP): ACT×NPIPE crosses (ACT mask)
      - Block 3 (PP): NPIPE×NPIPE crosses (PP_MASK controls source)

    PP_MASK controls where PP spectra and covariance come from:
      "act_window"    — PP on ACT footprint (same mask as AA/AP)
      "npipe_mask"     — PP on NPIPE mask (cross-mask covariance)
    """
    pp_mask = lcfg.PP_MASK
    act_mask_dir = os.path.join(lcfg.OUTPUT_DIR, lcfg.ACT_MASK_SUBDIR)
    npipe_full_dir = os.path.join(lcfg.OUTPUT_DIR, lcfg.NPIPE_MASK_SUBDIR)
    pp_dir = act_mask_dir if pp_mask == "act_window" else npipe_full_dir
    act_bands = lcfg.ACT_BANDS
    npipe_freqs = lcfg.NPIPE_FREQS
    npipe_splits = lcfg.NPIPE_SPLITS
    n_act_splits = lcfg.ACT_N_SPLITS

    ell_max_act_x_act = lcfg.ACT_X_ACT_ELL_MAX
    ell_max_act_x_npipe = lcfg.ACT_X_NPIPE_ELL_MAX
    ell_max_npipe_x_npipe = lcfg.NPIPE_ELL_MAX

    print("=" * 60)
    print(f"Computing JOINT likelihood data (PP_MASK={pp_mask})")
    print("=" * 60)
    print(f"  ACT bands: {act_bands}")
    print(f"  NPIPE freqs: {npipe_freqs}")

    # ── Build split map pool (shared across both groups) ──
    act_det_labels = _build_act_detector_labels(act_bands, n_act_splits)
    npipe_det_labels = _build_npipe_detector_labels(npipe_freqs, npipe_splits)
    all_det_labels = act_det_labels + npipe_det_labels
    n_act_det = len(act_det_labels)
    n_npipe_det = len(npipe_det_labels)
    detector_map_all = {label: idx for idx, label in enumerate(all_det_labels)}

    # Alpha labels
    act_alpha_labels = list(act_bands)
    npipe_alpha_labels = _build_npipe_alpha_labels(npipe_freqs, npipe_splits)
    alpha_labels = act_alpha_labels + npipe_alpha_labels

    det_to_band = {f"{band}_set{i}": band
                   for band in act_bands for i in range(n_act_splits)}

    # ── Partition cross-spectra into AA (block 1), AP (block 2), PP (block 3) ──
    aa_cross, ap_cross, pp_cross = _build_partitioned_cross_list(
        all_det_labels, det_to_band)
    n_b1 = len(aa_cross)
    n_b2 = len(ap_cross)
    n_b3 = len(pp_cross)
    n_aa_ap = n_b1 + n_b2  # blocks 1+2 share ACT mask

    cross_spec_list = aa_cross + ap_cross + pp_cross
    n_cross = len(cross_spec_list)

    print(f"  Split maps: {len(all_det_labels)} ({n_act_det} ACT + {n_npipe_det} NPIPE)")
    print(f"  Block 1 (AA): {n_b1} crosses")
    print(f"  Block 2 (AP): {n_b2} crosses")
    print(f"  Block 3 (PP): {n_b3} crosses")
    print(f"  Total cross-spectra: {n_cross}")

    # ── Get binning ──
    npipe_only = len(lcfg.ACT_BANDS) == 0
    if npipe_only and lcfg.NPIPE_ONLY_BINNING == "uniform":
        bin_edges = np.arange(lcfg.NPIPE_ELL_MIN,
                              lcfg.NPIPE_ELL_MAX + lcfg.NPIPE_DELTA_ELL + 1,
                              lcfg.NPIPE_DELTA_ELL)
        bin_centers = 0.5 * (bin_edges[:-1] + (bin_edges[1:] - 1))
        binning_file = None  # no corrections need it for NPIPE-only
        print(f"  NPIPE-only uniform binning: ell=[{bin_edges[0]}..{bin_edges[-1]}], "
              f"delta_ell={lcfg.NPIPE_DELTA_ELL}, {len(bin_centers)} bins")
    else:
        binning_file = lcfg.get_unified_binning_file()
        ell_max_all = max(ell_max_act_x_act, ell_max_act_x_npipe, ell_max_npipe_x_npipe)
        bin_edges, bin_centers = _parse_binning_file(binning_file, ell_max_all)
    ell_max_data = int(bin_edges[-1])  # last bin high edge for data loading
    n_bins = len(bin_centers)

    # ── Build cross_idx_maps ──
    # Unified cross_idx_map (for data vector, all split maps in one pool)
    cross_idx_map = _build_cross_idx_map(cross_spec_list, detector_map_all)

    # Blocks 1+2 (AA+AP): indices in full split map pool
    cross_idx_map_aa_ap = cross_idx_map[:n_aa_ap]

    # Block 3 (PP): indices in NPIPE-only split map pool (for Cov(3,3))
    npipe_det_map = {label: idx for idx, label in enumerate(npipe_det_labels)}
    cross_idx_map_pp_npipe = _build_cross_idx_map(pp_cross, npipe_det_map)

    # Block 3 (PP): indices in full split map pool (for Cov(1,3) and Cov(2,3))
    cross_idx_map_pp_full = cross_idx_map[n_aa_ap:]

    # ── Load data vectors ──

    def _find_act_mask_file(det1, det2, suffix):
        return _find_cross_file(act_mask_dir, det1, det2, suffix)

    def _find_pp_file(det1, det2, suffix):
        return _find_cross_file(pp_dir, det1, det2, suffix)

    # Load all data vectors and apply spectrum corrections
    # (corrections skip NPIPE×NPIPE internally, so pass full list)
    print(f"  Loading AA+AP data vectors ({n_aa_ap} crosses)...")
    aa_ap_cross = aa_cross + ap_cross
    C_obs_aa_ap, C_theory_aa_ap, _ = _load_unbinned_and_rebin_data_vectors(
        act_mask_dir, aa_ap_cross, _find_act_mask_file, bin_edges)

    print(f"  Loading PP data vectors ({n_b3} crosses, from {pp_dir})...")
    C_obs_pp, C_theory_pp, _ = _load_unbinned_and_rebin_data_vectors(
        pp_dir, pp_cross, _find_pp_file, bin_edges)

    # Stitch data vectors: [AA | AP | PP]
    C_obs_3d = np.zeros((n_bins, n_cross, 3))
    C_theory_3d = np.zeros((n_bins, n_cross, 2))
    C_obs_3d[:, :n_aa_ap, :] = C_obs_aa_ap
    C_theory_3d[:, :n_aa_ap, :] = C_theory_aa_ap
    C_obs_3d[:, n_aa_ap:, :] = C_obs_pp
    C_theory_3d[:, n_aa_ap:, :] = C_theory_pp

    # Apply spectrum corrections (skips NPIPE×NPIPE internally)
    print("  Applying spectrum corrections...")
    C_obs_3d = _apply_spectrum_corrections(
        C_obs_3d, bin_centers, cross_spec_list, det_to_band,
        binning_file, ell_max_data)

    print(f"  Loaded and binned data vectors: {n_bins} bins")

    # ── Load unbinned spectra for covariance ──
    print("  Loading unbinned spectra for covariance (ACT mask)...")
    # ACT-mask cl arrays (all split maps, for Cov(1,1), Cov(2,2), Cov(1,2))
    auto_act, cross_act, ell_act = _load_unbinned_auto_cross(
        act_mask_dir, all_det_labels, act_bands, n_act_splits,
        npipe_freqs, npipe_splits, ell_max_data)
    cl_ee_act, cl_bb_act = build_detector_spectrum_arrays(
        all_det_labels, auto_act, cross_act, len(ell_act))

    if npipe_det_labels:
        print(f"  Loading unbinned spectra for Cov(3,3) PP (from {pp_dir})...")
        ell_max_pp = ell_max_npipe_x_npipe
        auto_pp, cross_pp, ell_pp = _load_unbinned_auto_cross(
            pp_dir, npipe_det_labels, [], 0,
            npipe_freqs, npipe_splits, ell_max_pp)
        cl_ee_pp, cl_bb_pp = build_detector_spectrum_arrays(
            npipe_det_labels, auto_pp, cross_pp, len(ell_pp))
    else:
        cl_ee_pp = cl_bb_pp = np.empty((0, 0, 0))
        ell_pp = np.array([])

    # ── Compute f_sky for PP covariance ──
    if not npipe_freqs:
        f_pp = 0.0
    elif pp_mask == "act_window":
        _, f_pp = get_act_analysis_window()
        print(f"  f_sky_pp (act window): {f_pp:.4f}")
    else:  # "npipe_mask"
        npipe_mask = so_map.read_map(npipe_mask_path(2048, lcfg.NPIPE_MASK_PERCENT))
        f_pp = compute_fsky(npipe_mask)
        print(f"  f_sky_pp (npipe mask): {f_pp:.4f}")

    # ── Build k-space F_inv for AA+AP only ──
    kspace_F_inv_aa_ap = None
    if lcfg.APPLY_KSPACE_TRANSFER_CORRECTION:
        print("  Loading k-space transfer matrices for AA+AP...")
        n_bins_tf = len(bin_edges) - 1
        kspace_F_inv_aa_ap = _build_per_cross_F_inv_joint(
            aa_ap_cross, det_to_band, n_bins_tf, bin_centers)
        n_with_tf = sum(1 for d1, d2 in aa_ap_cross
                        if d1 in det_to_band or d2 in det_to_band)
        print(f"  K-space TF: {n_with_tf}/{n_aa_ap} AA+AP crosses")

    # ── Compute binned spectra (needed by all pspy blocks) ──
    cl_ee_act_binned = bin_detector_spectra(cl_ee_act, ell_act, bin_edges)
    cl_bb_act_binned = bin_detector_spectra(cl_bb_act, ell_act, bin_edges)

    # Sub-block cross_idx_maps
    cross_idx_map_b1 = cross_idx_map[:n_b1]
    cross_idx_map_b2 = cross_idx_map[n_b1:n_aa_ap]

    # ── Cov(1,1), Cov(2,2), Cov(1,2): AA and AP blocks (ACT mask) ──
    b1_method = lcfg.COV_METHOD_ACT
    b2_method = lcfg.COV_METHOD_ACT
    need_act_window = (n_b1 > 0 and b1_method != "mk") or \
                      (n_b2 > 0 and b2_method != "mk")
    print(f"  Covariance methods: AA={b1_method}, AP={b2_method}")

    # Load ACT coupling data if needed
    coupling_act = None
    if need_act_window:
        coupling_act = load_coupling_data(lcfg.ACT_COUPLING_PATH)

    # Cov(1,1): AA×AA
    if n_b1 > 0:
        print(f"  Computing Cov(1,1) AA×AA ({n_b1} crosses, method={b1_method})...")
        cov_ee_b1, cov_bb_b1, cov_eb_b1 = compute_ek_covariance(
            cl_ee_act, cl_bb_act, cross_idx_map_b1,
            coupling_data=coupling_act)

    # Cov(2,2) AP×AP and Cov(1,2) AA×AP
    if n_b2 > 0:
        print(f"  Computing Cov(2,2) AP×AP ({n_b2} crosses, method={b2_method})...")
        cov_ee_b2, cov_bb_b2, cov_eb_b2 = compute_ek_covariance(
            cl_ee_act, cl_bb_act, cross_idx_map_b2,
            coupling_data=coupling_act)
        if n_b1 > 0:
            print(f"  Computing Cov(1,2) AA×AP ({n_b1}x{n_b2} crosses, method={b2_method})...")
            cov_ee_b12, cov_bb_b12, cov_eb_b12 = compute_ek_covariance(
                cl_ee_act, cl_bb_act, cross_idx_map_b1,
                cross_idx_map_B=cross_idx_map_b2,
                coupling_data=coupling_act)

    # Stitch Cov(1,1), Cov(2,2), Cov(1,2) into AA+AP block
    n_bins_act = cl_ee_act_binned.shape[0]
    diag_11 = [
        (cov_ee_b1, cov_bb_b1, cov_eb_b1) if n_b1 > 0 else None,
        (cov_ee_b2, cov_bb_b2, cov_eb_b2) if n_b2 > 0 else None,
    ]
    cross_11 = {}
    if n_b1 > 0 and n_b2 > 0:
        cross_11[(0, 1)] = (cov_ee_b12, cov_bb_b12, cov_eb_b12)
    cov_ee_11, cov_bb_11, cov_eb_11 = _stitch_block_covariance(
        n_bins_act, [n_b1, n_b2], diag_11, cross_11)

    # Apply kspace F_inv to AA+AP block
    if kspace_F_inv_aa_ap is not None:
        cov_ee_11, cov_bb_11, cov_eb_11 = apply_kspace_to_covariance(
            cov_ee_11, cov_bb_11, cov_eb_11, kspace_F_inv_aa_ap,
            n_bins_act, n_aa_ap)

    # Cov(3,3): PP×PP
    pp_method = lcfg.COV_METHOD_PP
    print(f"  Computing Cov(3,3) PP×PP ({n_b3}x{n_b3}, method={pp_method}, "
          f"pp_mask={pp_mask})...")
    if pp_method == "mk":
        cov_ee_33, cov_bb_33, cov_eb_33 = compute_ek_covariance(
            cl_ee_pp, cl_bb_pp, cross_idx_map_pp_npipe, ell_pp,
            f_pp, bin_edges)
    else:
        coupling_pp = coupling_act if pp_mask == "act_window" else None
        cov_ee_33, cov_bb_33, cov_eb_33 = compute_ek_covariance(
            cl_ee_pp, cl_bb_pp, cross_idx_map_pp_npipe,
            coupling_data=coupling_pp)

    # Cov(1,3) AA×PP and Cov(2,3) AP×PP cross-covariance
    if pp_mask == "act_window":
        # Same mask: use ACT coupling kernel, single f_sky
        cross_mask_method = lcfg.COV_METHOD_ACT
        coupling_13 = coupling_act
        coupling_23 = coupling_act
    else:
        # Cross-mask: separate coupling kernels for each cross-block
        cross_mask_method = lcfg.COV_METHOD_ACT
        coupling_13 = (load_coupling_data(lcfg.ACT_X_NPIPE_COUPLING_PATH)
                       if cross_mask_method != "mk" else None)
        coupling_23 = (load_coupling_data(lcfg.ACTPS_X_NPIPE_COUPLING_PATH)
                       if cross_mask_method != "mk" else None)
    print(f"  Computing Cov(1,3)+Cov(2,3) cross-covariance ({n_aa_ap}x{n_b3}, "
          f"method={cross_mask_method})...")
    cross_parts = []
    if n_b1 > 0:
        cov_ee_13, cov_bb_13, cov_eb_13 = compute_ek_covariance(
            cl_ee_act, cl_bb_act, cross_idx_map_b1,
            cross_idx_map_B=cross_idx_map_pp_full,
            coupling_data=coupling_13)
        cross_parts.append((cov_ee_13, cov_bb_13, cov_eb_13))
    if n_b2 > 0:
        cov_ee_23, cov_bb_23, cov_eb_23 = compute_ek_covariance(
            cl_ee_act, cl_bb_act, cross_idx_map_b2,
            cross_idx_map_B=cross_idx_map_pp_full,
            coupling_data=coupling_23)
        cross_parts.append((cov_ee_23, cov_bb_23, cov_eb_23))
    # Concatenate Cov(1,3) and Cov(2,3) along cross-spectrum axis
    cross_blocks = {}
    if cross_parts:
        cov_ee_cross = np.concatenate([p[0] for p in cross_parts], axis=1)
        cov_bb_cross = np.concatenate([p[1] for p in cross_parts], axis=1)
        cov_eb_cross = np.concatenate([p[2] for p in cross_parts], axis=1)

        # Apply kspace F_inv to cross-blocks (AA+AP side filtered, PP side identity)
        if kspace_F_inv_aa_ap is not None:
            cov_ee_cross, cov_bb_cross, cov_eb_cross = apply_kspace_to_covariance(
                cov_ee_cross, cov_bb_cross, cov_eb_cross,
                kspace_F_inv_aa_ap, n_bins_act, n_aa_ap)

        cross_blocks[(0, 1)] = (cov_ee_cross, cov_bb_cross, cov_eb_cross)

    # ── Stitch into unified covariance: [AA+AP | PP] ──
    cov_ee, cov_bb, cov_eb = _stitch_block_covariance(
        n_bins, [n_aa_ap, n_b3],
        [(cov_ee_11, cov_bb_11, cov_eb_11),
         (cov_ee_33, cov_bb_33, cov_eb_33)],
        cross_blocks)

    # Add T→P leakage covariance (additive term, ACT crosses only)
    if lcfg.APPLY_LEAKAGE_CORRECTION:
        leak_cov_ee, leak_cov_bb, leak_cov_eb = compute_leakage_covariance(
            ell_act, bin_edges, cross_idx_map, all_det_labels,
            det_to_band, ell_max_data)
        cov_ee = cov_ee + leak_cov_ee
        cov_bb = cov_bb + leak_cov_bb
        cov_eb = cov_eb + leak_cov_eb

    print(f"  Bins: {n_bins}, ell range: [{bin_centers[0]:.0f}, {bin_centers[-1]:.0f}]")

    # ── Average ACT splits (Eq. C9) ──
    print("  ── Averaging ACT splits (Eq. C9) ──")

    sacc_aa = None
    if lcfg.USE_SACC_AA and act_bands:
        print("  Loading AA block from SACC...")
        sacc_aa = _load_sacc_aa_block(act_bands, bin_centers)

    avg = _apply_split_averaging(
        cov_ee, cov_bb, cov_eb, C_obs_3d, C_theory_3d,
        aa_cross, ap_cross, pp_cross,
        act_bands, npipe_det_labels, det_to_band, n_act_splits,
        sacc_aa=sacc_aa, bin_centers=bin_centers)

    cov_ee = avg['cov_ee']
    cov_bb = avg['cov_bb']
    cov_eb = avg['cov_eb']
    C_obs_3d = avg['C_obs_3d']
    C_theory_3d = avg['C_theory_3d']
    cross_spec_list = avg['cross_spec_list']
    all_det_labels = avg['det_labels']
    cross_idx_map = avg['cross_idx_map']
    det_to_band = avg['det_to_band']
    act_alpha_labels = avg['act_alpha_labels']
    alpha_labels = act_alpha_labels + npipe_alpha_labels
    n_b1 = avg['n_b1']
    n_b2 = avg['n_b2']
    n_aa_ap = n_b1 + n_b2
    n_b3 = avg['n_b3']
    n_cross = len(cross_spec_list)

    # ── Build active_bins ──
    active_bins = _build_active_bins(
        cross_spec_list, bin_centers, det_to_band,
        ell_max_act_x_act, ell_max_act_x_npipe, ell_max_npipe_x_npipe,
        lcfg.ACT_ELL_MIN_PER_BAND, lcfg.NPIPE_ACT_MASK_ELL_MIN,
        act_band_set=set(act_bands), n_aa_ap=n_aa_ap,
        ell_min_pp=lcfg.NPIPE_ELL_MIN)

    n_active_bins = np.sum(active_bins.any(axis=0))
    print(f"  Active bins: {n_active_bins}/{n_bins}")
    if n_b1 > 0 and n_b3 > 0:
        for label, slc in [("ACT×ACT", slice(0, n_b1)),
                           ("ACT×Planck", slice(n_b1, n_aa_ap)),
                           ("Planck×Planck", slice(n_aa_ap, None))]:
            block_active = active_bins[slc].any(axis=0)
            if block_active.any():
                bc_block = bin_centers[block_active]
                print(f"    {label}: ell = [{bc_block[0]:.1f}, {bc_block[-1]:.1f}] "
                      f"({int(block_active.sum())} bins)")

    # ── Dust cross mask (only block 3 PP NPIPE HFI crosses) ──
    npipe_dust_det_set = {f"{freq}{split}" for freq in npipe_freqs
                          for split in npipe_splits if freq in NPIPE_DUST_FREQS}
    dust_cross_mask = np.array([
        (k >= n_aa_ap) and det1 in npipe_dust_det_set and det2 in npipe_dust_det_set
        for k, (det1, det2) in enumerate(cross_spec_list)
    ])
    print(f"  Dust crosses: {np.sum(dust_cross_mask)}/{n_cross}")

    # ── Compute dust psi_ell from 353 GHz spectra ──
    out_dir = os.path.join(lcfg.OUTPUT_DIR, lcfg.JOINT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    psi_ell_binned = None
    if lcfg.COMPUTE_PSI_ELL:
        psi_raw, _ = compute_psi_ell(pp_dir, binning="joint",
                                     output_dir=out_dir)
        if psi_raw is not None:
            psi_ell_binned = np.zeros(n_bins)
            n_copy = min(len(psi_raw), n_bins)
            psi_ell_binned[:n_copy] = psi_raw[:n_copy]

    # ── Save ──
    npipe_only = not act_bands
    out_filename = build_npz_filename(
        act_bands, npipe_freqs,
        use_sacc_aa=lcfg.USE_SACC_AA and not npipe_only,
        npipe_only_binning=lcfg.NPIPE_ONLY_BINNING if npipe_only else None,
        npipe_ell_min=lcfg.NPIPE_ELL_MIN if npipe_only else None,
        npipe_ell_max=lcfg.NPIPE_ELL_MAX if npipe_only else None,
        npipe_delta_ell=lcfg.NPIPE_DELTA_ELL if npipe_only else None,
        pp_mask_percent=lcfg.NPIPE_MASK_PERCENT)
    out_path = os.path.join(out_dir, out_filename)

    _save_npz(out_path, cov_ee, cov_bb, cov_eb, C_obs_3d, C_theory_3d,
              bin_centers, cross_idx_map, all_det_labels, alpha_labels,
              "joint", cross_spec_list,
              band_labels=act_bands,
              det_to_band=det_to_band,
              act_bands=act_bands,
              active_bins=active_bins,
              npipe_freqs=npipe_freqs,
              npipe_splits=npipe_splits,
              act_alpha_labels=act_alpha_labels,
              npipe_alpha_labels=npipe_alpha_labels,
              dust_cross_mask=dust_cross_mask,
              psi_ell_binned=psi_ell_binned)

    # Add unified-specific metadata in a single re-save
    data = dict(np.load(out_path))
    data.update({
        'n_b1': n_b1,
        'n_b2': n_b2,
    })
    np.savez(out_path, **data)

    print(f"  Saved: {out_path}")
    return out_path


def run():
    """Run the full likelihood processing pipeline."""
    print("=" * 60)
    print("Likelihood Processing Pipeline")
    print("=" * 60)
    print(f"  ACT bands:   {lcfg.ACT_BANDS or '(none)'}")
    print(f"  NPIPE freqs:  {lcfg.NPIPE_FREQS or '(none)'}")
    print(f"  USE_SACC_AA:  {lcfg.USE_SACC_AA}")
    print(f"  Mask:         {lcfg.MASK}")
    print(f"  PP mask:      {lcfg.PP_MASK}")
    if lcfg.NPIPE_FREQS:
        npipe_dir = os.path.join(lcfg.OUTPUT_DIR, lcfg.NPIPE_MASK_SUBDIR)
        print(f"  NPIPE dir:    {npipe_dir}")
    print(f"  K-space TF:   {lcfg.APPLY_KSPACE_TRANSFER_CORRECTION}")
    print(f"  Leakage:      {lcfg.APPLY_LEAKAGE_CORRECTION}")

    print("=" * 60)

    npz_path = save_likelihood_data()

    if lcfg.MAKE_PLOTS:
        run_null_tests(npz_path)

    print("\n" + "=" * 60)
    print("Likelihood processing complete!")
    print("=" * 60)
    return npz_path


if __name__ == "__main__":
    run()
