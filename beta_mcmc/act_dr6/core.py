"""ACT DR6 birefringence analysis: SACC data loading."""
import os
import sys
from pathlib import Path

import numpy as np
import sacc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_DIR, ACT_DR6_ELL_MAX, ACT_DR6_OFFDIAG_COV
from tools.spectra import dell_to_cell, camb_theory, bin_theory_with_window
from birefringence_likelihood import precompute_bin_groups


def load_act_sacc_data(bands):
    """Load ACT DR6 data from SACC file into E&K-compatible data_dict.

    Builds all 25 ordered cross-pairs from 5 bands (15 from cl_eb + 10 from
    cl_be treated as reversed EB), applies per-pair b_min cuts matching
    Diego-Palazuelos & Komatsu (2025) Table I, and extracts the full
    cross-pair covariance.

    Args:
        bands: List of ACT band names, e.g. ["pa5_f090", "pa5_f150", ...].

    Returns:
        data_dict with 'likelihood_cache' pre-populated.
    """
    sacc_path = os.path.join(DATA_DIR, "act_dr6", "v1.0", "dr6_data.fits")
    s = sacc.Sacc.load_fits(sacc_path)
    full_cov = s.covariance.covmat

    # Per-band minimum multipole
    band_ell_min = {
        "pa4_f220": 1000.5,
        "pa5_f090": 1000.5,
        "pa5_f150": 800.5,
        "pa6_f090": 1000.5,
        "pa6_f150": 600.5,
    }

    bands_sorted = sorted(set(bands))

    # Build all 25 ordered cross-pairs:
    #   5 auto-spectra from cl_eb (i,i)
    #   10 upper-triangle from cl_eb (i,j) with i<j
    #   10 lower-triangle from cl_be (i,j) → treated as EB(j,i)
    # cross_pairs stores (band_e, band_b) where band_e contributes E, band_b contributes B
    cross_pairs = []        # (band_e, band_b) for alpha mapping
    ee_indices = []
    bb_indices = []
    eb_indices = []
    ee_data = []
    bb_data = []
    eb_data = []

    ell_ref = None

    # 1) All cl_eb pairs: upper triangle + diagonal (15 pairs)
    eb_sacc_pairs = []
    for d in s.data:
        if d.data_type == 'cl_eb':
            pair = (d.tracers[0], d.tracers[1])
            if pair not in eb_sacc_pairs:
                eb_sacc_pairs.append(pair)

    for t_i, t_j in eb_sacc_pairs:
        band_i = t_i.replace("dr6_", "").replace("_s2", "")
        band_j = t_j.replace("dr6_", "").replace("_s2", "")
        if band_i not in bands_sorted or band_j not in bands_sorted:
            continue

        ell, cl_ee, ind_ee = s.get_ell_cl('cl_ee', t_i, t_j, return_ind=True)
        _, cl_bb, ind_bb = s.get_ell_cl('cl_bb', t_i, t_j, return_ind=True)
        _, cl_eb, ind_eb = s.get_ell_cl('cl_eb', t_i, t_j, return_ind=True)

        if ell_ref is None:
            ell_ref = ell
        cross_pairs.append((band_i, band_j))  # E from i, B from j
        ee_indices.append(ind_ee)
        bb_indices.append(ind_bb)
        eb_indices.append(ind_eb)
        ee_data.append(dell_to_cell(ell, cl_ee))
        bb_data.append(dell_to_cell(ell, cl_bb))
        eb_data.append(dell_to_cell(ell, cl_eb))

    # 2) All cl_be pairs: treat BE(i,j) as EB(j,i) (10 off-diagonal pairs)
    be_sacc_pairs = []
    for d in s.data:
        if d.data_type == 'cl_be':
            pair = (d.tracers[0], d.tracers[1])
            if pair not in be_sacc_pairs:
                be_sacc_pairs.append(pair)

    for t_i, t_j in be_sacc_pairs:
        band_i = t_i.replace("dr6_", "").replace("_s2", "")
        band_j = t_j.replace("dr6_", "").replace("_s2", "")
        if band_i not in bands_sorted or band_j not in bands_sorted:
            continue

        # BE(i,j) = <B_i E_j> = EB(j,i): E from j, B from i
        # EE/BB are symmetric: EE(i,j) = EE(j,i)
        _, cl_ee_ij, ind_ee = s.get_ell_cl('cl_ee', t_i, t_j, return_ind=True)
        _, cl_bb_ij, ind_bb = s.get_ell_cl('cl_bb', t_i, t_j, return_ind=True)
        _, cl_be, ind_be = s.get_ell_cl('cl_be', t_i, t_j, return_ind=True)

        cross_pairs.append((band_j, band_i))  # E from j, B from i (reversed)
        ee_indices.append(ind_ee)  # EE is symmetric
        bb_indices.append(ind_bb)  # BB is symmetric
        eb_indices.append(ind_be)  # BE indices in SACC covariance
        ee_data.append(dell_to_cell(ell_ref, cl_ee_ij))
        bb_data.append(dell_to_cell(ell_ref, cl_bb_ij))
        eb_data.append(dell_to_cell(ell_ref, cl_be))

    n_cross = len(cross_pairs)
    n_bins = len(ell_ref)

    # Stack into arrays
    ee_arr = np.column_stack(ee_data)
    bb_arr = np.column_stack(bb_data)
    eb_arr = np.column_stack(eb_data)
    C_obs_3d = np.stack([ee_arr, bb_arr, eb_arr], axis=-1)

    # CMB theory
    ell_max = int(np.max(ell_ref)) + 10
    ells_th, EE_th, BB_th = camb_theory(ell_max=ell_max)
    theory_ee = bin_theory_with_window(ells_th, EE_th, ell_ref.astype(int))
    theory_bb = bin_theory_with_window(ells_th, BB_th, ell_ref.astype(int))

    C_theory_3d = np.zeros((n_bins, n_cross, 2))
    for k in range(n_cross):
        C_theory_3d[:, k, 0] = theory_ee
        C_theory_3d[:, k, 1] = theory_bb

    # Covariance extraction (D_ell cov -> C_ell cov)
    dell_to_cell_factor = 2.0 * np.pi / (ell_ref * (ell_ref + 1.0))

    if ACT_DR6_OFFDIAG_COV:
        # Full bin-to-bin covariance (Diego-Palazuelos & Komatsu 2025)
        ee_idx_arr = np.column_stack(ee_indices)  # (n_bins, n_cross)
        bb_idx_arr = np.column_stack(bb_indices)
        eb_idx_arr = np.column_stack(eb_indices)
        f_flat = np.repeat(dell_to_cell_factor, n_cross)
        f2 = np.outer(f_flat, f_flat)
        cov_ee = (full_cov[np.ix_(ee_idx_arr.ravel(), ee_idx_arr.ravel())] * f2
                  ).reshape(n_bins, n_cross, n_bins, n_cross)
        cov_bb = (full_cov[np.ix_(bb_idx_arr.ravel(), bb_idx_arr.ravel())] * f2
                  ).reshape(n_bins, n_cross, n_bins, n_cross)
        cov_eb = (full_cov[np.ix_(eb_idx_arr.ravel(), eb_idx_arr.ravel())] * f2
                  ).reshape(n_bins, n_cross, n_bins, n_cross)
    else:
        # Per-bin covariance blocks (diagonal in ell)
        cov_ee = np.zeros((n_bins, n_cross, n_cross))
        cov_bb = np.zeros((n_bins, n_cross, n_cross))
        cov_eb = np.zeros((n_bins, n_cross, n_cross))
        for b in range(n_bins):
            f = dell_to_cell_factor[b]
            f2 = f * f
            for m in range(n_cross):
                for n in range(n_cross):
                    cov_ee[b, m, n] = full_cov[ee_indices[m][b], ee_indices[n][b]] * f2
                    cov_bb[b, m, n] = full_cov[bb_indices[m][b], bb_indices[n][b]] * f2
                    cov_eb[b, m, n] = full_cov[eb_indices[m][b], eb_indices[n][b]] * f2

    # Apply ell_max cut
    if ACT_DR6_ELL_MAX is not None:
        keep = ell_ref <= ACT_DR6_ELL_MAX
        ell_ref = ell_ref[keep]
        n_bins = len(ell_ref)
        ee_arr = ee_arr[keep]
        bb_arr = bb_arr[keep]
        eb_arr = eb_arr[keep]
        C_obs_3d = C_obs_3d[keep]
        C_theory_3d = C_theory_3d[keep]
        if ACT_DR6_OFFDIAG_COV:
            cov_ee = cov_ee[keep][:, :, keep, :]
            cov_bb = cov_bb[keep][:, :, keep, :]
            cov_eb = cov_eb[keep][:, :, keep, :]
        else:
            cov_ee = cov_ee[keep]
            cov_bb = cov_bb[keep]
            cov_eb = cov_eb[keep]

    # Per-pair b_min cuts via active_bins (n_cross, n_bins)
    active_bins = np.zeros((n_cross, n_bins), dtype=bool)
    for k, (band_e, band_b) in enumerate(cross_pairs):
        pair_ell_min = max(band_ell_min[band_e], band_ell_min[band_b])
        active_bins[k, :] = ell_ref >= pair_ell_min

    n_active = int(np.sum(active_bins))

    # Alpha mapping
    alpha_labels = bands_sorted
    label_to_idx = {band: k for k, band in enumerate(alpha_labels)}
    n_alpha = len(alpha_labels)

    alpha_idx_i = np.zeros(n_cross, dtype=int)
    alpha_idx_j = np.zeros(n_cross, dtype=int)
    for k, (band_e, band_b) in enumerate(cross_pairs):
        alpha_idx_i[k] = label_to_idx[band_e]
        alpha_idx_j[k] = label_to_idx[band_b]

    detector_labels = list(bands_sorted)
    cross_idx_map = np.zeros((n_cross, 2), dtype=int)
    for k, (band_e, band_b) in enumerate(cross_pairs):
        cross_idx_map[k, 0] = detector_labels.index(band_e)
        cross_idx_map[k, 1] = detector_labels.index(band_b)

    data_dict = {
        'cross_spec_list': cross_pairs,
        'detector_labels': detector_labels,
        'alpha_labels': alpha_labels,
        'act_alpha_labels': alpha_labels,
        'npipe_alpha_labels': [],
        'ell': ell_ref,
        'n_bins': n_bins,
        'active_bins': active_bins,
        'likelihood_cache': {
            **({'cov_ee_full': cov_ee, 'cov_bb_full': cov_bb,
                'cov_eb_full': cov_eb}
               if ACT_DR6_OFFDIAG_COV else
               {'cov_ee': cov_ee, 'cov_bb': cov_bb, 'cov_eb': cov_eb}),
            'C_obs_3d': C_obs_3d,
            'C_theory_3d': C_theory_3d,
        },
        'cov_arrays': {
            'cross_idx_map': cross_idx_map,
            'detector_map': {label: idx for idx, label in enumerate(detector_labels)},
        },
        'alpha_maps': {
            'alpha_idx_i': alpha_idx_i,
            'alpha_idx_j': alpha_idx_j,
            'n_alpha': n_alpha,
        },
    }

    # Eagerly precompute bin_groups so multiprocessing workers inherit them
    precompute_bin_groups(data_dict)

    print(f"Loaded SACC data from {sacc_path}")
    print(f"  Bands: {bands_sorted}")
    print(f"  Cross-pairs: {n_cross} (ordered), ell bins: {n_bins}")
    print(f"  ell range: [{ell_ref[0]:.0f}, {ell_ref[-1]:.0f}]")
    print(f"  Active data points: {n_active} (after per-pair b_min cuts)")
    print(f"  Alpha params: {n_alpha} ({alpha_labels})")

    return data_dict
