"""Load precomputed likelihood data from .npz files."""
import numpy as np

from config import DUST_ELL_BINS, AP_ONLY


def load_precomputed(npz_path):
    """Load .npz and return data_dict with likelihood_cache pre-populated.

    Args:
        npz_path: Path to likelihood_data.npz.

    Returns:
        data_dict with all fields needed for MCMC.
    """
    d = np.load(npz_path, allow_pickle=False)

    analysis_type = str(d['analysis_type'])
    n_bins = int(d['n_bins'])
    n_cross = int(d['n_cross'])
    bin_centers = d['bin_centers']
    cross_idx_map = d['cross_idx_map']
    detector_labels = list(d['detector_labels'])
    alpha_labels = list(d['alpha_labels'])

    # Reconstruct cross_spec_list
    cross_spec_list = _reconstruct_cross_spec_list(d, analysis_type)

    # Build likelihood_cache
    likelihood_cache = {
        'cov_ee': d['cov_ee'],
        'cov_bb': d['cov_bb'],
        'cov_eb': d['cov_eb'],
        'C_obs_3d': d['C_obs_3d'],
        'C_theory_3d': d['C_theory_3d'],
    }

    # Build cov_arrays (needed by likelihood functions for cross_idx_map)
    cov_arrays = {
        'cross_idx_map': cross_idx_map,
        'detector_map': {label: idx for idx, label in enumerate(detector_labels)},
    }

    # Assemble data_dict
    data_dict = {
        'cross_spec_list': cross_spec_list,
        'detector_labels': detector_labels,
        'alpha_labels': alpha_labels,
        'ell': bin_centers,
        'n_bins': n_bins,
        'likelihood_cache': likelihood_cache,
        'cov_arrays': cov_arrays,
        'cl_arrays': {'spec_idx_map': cross_idx_map,
                      'detector_map': cov_arrays['detector_map']},
    }

    # Analysis-type-specific fields
    if 'band_labels' in d:
        data_dict['band_labels'] = list(d['band_labels'])
    if 'det_to_band_keys' in d:
        data_dict['det_to_band'] = dict(zip(d['det_to_band_keys'], d['det_to_band_vals']))
    if 'act_bands' in d:
        data_dict['act_bands'] = list(d['act_bands'])
    if 'active_bins' in d:
        data_dict['active_bins'] = d['active_bins']
    if 'npipe_freqs' in d:
        data_dict['npipe_freqs'] = list(d['npipe_freqs'])
    if 'npipe_splits' in d:
        data_dict['npipe_splits'] = list(d['npipe_splits'])
    if 'act_alpha_labels' in d:
        data_dict['act_alpha_labels'] = list(d['act_alpha_labels'])
    if 'npipe_alpha_labels' in d:
        data_dict['npipe_alpha_labels'] = list(d['npipe_alpha_labels'])
    if 'psi_ell_binned' in d:
        dust_bin_idx = np.full(n_bins, -1, dtype=np.int32)
        for d_idx, (lo, hi) in enumerate(DUST_ELL_BINS):
            mask = (bin_centers >= lo) & (bin_centers <= hi)
            dust_bin_idx[mask] = d_idx
        dust_info = {
            'psi_ell_binned': d['psi_ell_binned'],
            'dust_bin_idx': dust_bin_idx,
        }
        if 'dust_cross_mask' in d:
            dust_info['dust_cross_mask'] = d['dust_cross_mask']
        data_dict['dust'] = dust_info

    print(f"Loaded precomputed likelihood data from {npz_path}")
    print(f"  Analysis: {analysis_type}, bins: {n_bins}, cross-spectra: {n_cross}")
    print(f"  ell stored: [{bin_centers[0]:.0f}, {bin_centers[-1]:.0f}]")
    if 'active_bins' in d:
        active = d['active_bins']
        any_active = active.any(axis=0)
        active_centers = bin_centers[any_active]
        print(f"  ell used (active_bins): [{active_centers[0]:.0f}, {active_centers[-1]:.0f}]")

    if AP_ONLY:
        _filter_ap_only(data_dict, d)

    return data_dict


def _filter_ap_only(data_dict, d):
    """Filter data_dict in-place to keep only ACT×NPIPE cross-spectra.

    Uses n_b1 (AA count) and n_b2 (AP count) from the .npz metadata.
    Cross-spectra are ordered [AA, AP, PP] in the unified .npz.
    """
    n_b1 = int(d['n_b1'])
    n_b2 = int(d['n_b2'])
    ap_idx = np.arange(n_b1, n_b1 + n_b2)
    n_bins = data_dict['n_bins']

    # Subset likelihood_cache
    cache = data_dict['likelihood_cache']
    cache['cov_ee'] = cache['cov_ee'][:, ap_idx, :][:, :, ap_idx]
    cache['cov_bb'] = cache['cov_bb'][:, ap_idx, :][:, :, ap_idx]
    cache['cov_eb'] = cache['cov_eb'][:, ap_idx, :][:, :, ap_idx]
    cache['C_obs_3d'] = cache['C_obs_3d'][:, ap_idx, :]
    cache['C_theory_3d'] = cache['C_theory_3d'][:, ap_idx, :]

    # Subset cross_idx_map
    data_dict['cov_arrays']['cross_idx_map'] = \
        data_dict['cov_arrays']['cross_idx_map'][ap_idx, :]
    data_dict['cl_arrays']['spec_idx_map'] = \
        data_dict['cov_arrays']['cross_idx_map']

    # Subset cross_spec_list
    data_dict['cross_spec_list'] = [
        data_dict['cross_spec_list'][i] for i in ap_idx]

    # Subset active_bins
    if 'active_bins' in data_dict:
        data_dict['active_bins'] = data_dict['active_bins'][ap_idx, :]

    # Subset dust cross mask if present
    if 'dust' in data_dict and 'dust_cross_mask' in data_dict['dust']:
        data_dict['dust']['dust_cross_mask'] = \
            data_dict['dust']['dust_cross_mask'][ap_idx]

    n_pp = int(d['n_cross']) - n_b1 - n_b2
    print(f"  AP_ONLY: filtered to {n_b2} ACT×NPIPE cross-spectra "
          f"(removed {n_b1} AA + {n_pp} PP)")


def _reconstruct_cross_spec_list(d, analysis_type):
    """Reconstruct cross_spec_list from component arrays in .npz."""
    if analysis_type == "npipe":
        freqs1 = d['cross_freqs1']
        splits1 = d['cross_splits1']
        freqs2 = d['cross_freqs2']
        splits2 = d['cross_splits2']
        return [(int(f1), str(s1), int(f2), str(s2))
                for f1, s1, f2, s2 in zip(freqs1, splits1, freqs2, splits2)]

    elif analysis_type in ("joint_act_mask", "unified", "joint"):
        det1 = d['cross_det1']
        det2 = d['cross_det2']
        return [(str(d1), str(d2)) for d1, d2 in zip(det1, det2)]

    raise ValueError(f"Unknown analysis_type: {analysis_type}")
