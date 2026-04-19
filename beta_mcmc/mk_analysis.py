"""Unified M&K likelihood, prior, and posterior for all analysis types.

Handles npipe, act_x_npipe, act_x_act, and joint_act_mask analysis types
in a single module. Each type stores alpha labels differently in data_dict;
this module dispatches on data_dict structure (duck typing) to handle all cases.
"""

import numpy as np

from birefringence_likelihood import (
    precompute_bin_groups, mk_log_likelihood,
    log_beta_prior, log_act_alpha_prior, log_npipe_alpha_prior,
)
from rotation import compute_f_ell


def get_alpha_labels(data_dict):
    """Get the ordered list of alpha parameter labels from data_dict.

    Dispatches on data_dict structure:
    - All .npz-loaded types: alpha_labels (always present, set correctly
      per analysis type by run_likelihood.py)
    """
    if 'act_alpha_labels' in data_dict:
        return data_dict['act_alpha_labels'] + data_dict['npipe_alpha_labels']
    return data_dict['alpha_labels']


def _precompute_alpha_maps(data_dict):
    """One-time precomputation of detector -> alpha index mapping.

    Handles three alpha-mapping strategies:
    - act_x_npipe: ACT bands via startswith prefix, NPIPE by direct label
    - act_x_act: all detectors via det_to_band dict
    - joint_act_mask: ACT via det_to_band, NPIPE by direct label
    """
    cross_idx_map = data_dict['cov_arrays']['cross_idx_map']
    n_cross = len(data_dict['cross_spec_list'])
    detector_labels = data_dict['detector_labels']

    all_labels = get_alpha_labels(data_dict)
    n_alpha = len(all_labels)
    label_to_idx = {label: k for k, label in enumerate(all_labels)}

    det_to_band = data_dict.get('det_to_band')
    act_bands = data_dict.get('act_bands')

    def _det_alpha_idx(det_label):
        if det_to_band and det_label in det_to_band:
            return label_to_idx[det_to_band[det_label]]
        if act_bands:
            for band in act_bands:
                if det_label.startswith(band + '_set'):
                    return label_to_idx[band]
        if det_label in label_to_idx:
            return label_to_idx[det_label]
        # Time-split fallback: "30A" -> "30" (shared alpha for time-splits)
        freq_only = ''.join(c for c in det_label if c.isdigit())
        if freq_only in label_to_idx:
            return label_to_idx[freq_only]
        raise KeyError(f"No alpha label found for detector '{det_label}'")

    alpha_idx_i = np.zeros(n_cross, dtype=int)
    alpha_idx_j = np.zeros(n_cross, dtype=int)
    for k in range(n_cross):
        alpha_idx_i[k] = _det_alpha_idx(detector_labels[cross_idx_map[k, 0]])
        alpha_idx_j[k] = _det_alpha_idx(detector_labels[cross_idx_map[k, 1]])

    data_dict['alpha_maps'] = {
        'alpha_idx_i': alpha_idx_i,
        'alpha_idx_j': alpha_idx_j,
        'n_alpha': n_alpha,
    }


def mk_likelihood(theta, data_dict, sample_alphas=False,
                   n_dust_params=0, sample_beta=True):
    """Unified M&K log-likelihood for all analysis types.

    Uses precomputed alpha_maps to map detector pairs to alpha parameter
    indices. Works for all types: NPIPE (1:1 detector-to-alpha), ACT×ACT
    (many-to-one via det_to_band), ACT×NPIPE (mixed), joint_act_mask.

    Args:
        theta: Parameter vector. Layout depends on sample_beta/sample_alphas:
            sample_beta=True:  [beta, alphas..., dust_params...]
            sample_beta=False: [alphas..., dust_params...]
        data_dict: Precomputed data dict from load_precomputed.
        sample_alphas: Whether to sample miscalibration angles.
        n_dust_params: Number of dust A_l parameters at end of theta.
        sample_beta: Whether beta is included in theta. If False, beta=0.

    Returns:
        Log-likelihood value.
    """
    assert 'likelihood_cache' in data_dict, "likelihood_cache missing from data_dict"
    if 'active_bins' in data_dict and 'bin_groups' not in data_dict:
        precompute_bin_groups(data_dict)
    if 'alpha_maps' not in data_dict:
        _precompute_alpha_maps(data_dict)

    f_ell = None
    if n_dust_params > 0 and 'dust' in data_dict:
        n_cross = len(data_dict['cross_spec_list'])
        f_ell = compute_f_ell(theta[-n_dust_params:], data_dict['dust'], n_cross)

    alpha_offset = 1 if sample_beta else 0
    beta = theta[0] if sample_beta else 0.0
    maps = data_dict['alpha_maps']
    alpha_by_label = np.zeros(maps['n_alpha'])
    if sample_alphas:
        alpha_by_label[:] = theta[alpha_offset:alpha_offset + maps['n_alpha']]
    alpha_i = alpha_by_label[maps['alpha_idx_i']]
    alpha_j = alpha_by_label[maps['alpha_idx_j']]

    return mk_log_likelihood(alpha_i, alpha_j, beta, data_dict, f_ell=f_ell)


def log_prior_mk(theta, data_dict, sample_alphas=False, n_dust_params=0,
                 sample_beta=True):
    """Unified prior for all M&K analysis types.

    Dispatches alpha prior on data_dict structure:
    - Has act_alpha_labels: correlated ACT prior + independent NPIPE prior
    - Otherwise: independent Gaussian prior on all alphas

    Args:
        theta: Parameter vector. Layout depends on sample_beta:
            sample_beta=True:  [beta, alphas..., dust_params...]
            sample_beta=False: [alphas..., dust_params...]
        data_dict: Precomputed data dict.
        sample_alphas: Whether alphas are being sampled.
        n_dust_params: Number of dust A_l parameters at end of theta.
        sample_beta: Whether beta is included in theta.

    Returns:
        Log-prior value.
    """
    lp = 0.0
    alpha_offset = 0

    if sample_beta:
        lp = log_beta_prior(theta[0])
        if not np.isfinite(lp):
            return -np.inf
        alpha_offset = 1

    if n_dust_params > 0:
        if np.any(theta[-n_dust_params:] < 0):
            return -np.inf

    if not sample_alphas:
        return lp

    if 'act_alpha_labels' in data_dict:
        act_labels = data_dict['act_alpha_labels']
        npipe_labels = data_dict['npipe_alpha_labels']
        n_act = len(act_labels)
        n_npipe = len(npipe_labels)
        if n_act > 0:
            lp += log_act_alpha_prior(
                theta[alpha_offset:alpha_offset + n_act], act_labels)
        if n_npipe > 0:
            lp += log_npipe_alpha_prior(
                theta[alpha_offset + n_act:alpha_offset + n_act + n_npipe],
                npipe_labels)
    else:
        alpha_labels = get_alpha_labels(data_dict)
        n_alphas = len(alpha_labels)
        lp += log_npipe_alpha_prior(
            theta[alpha_offset:alpha_offset + n_alphas], alpha_labels)

    return lp


def log_posterior(log_prior_fn, log_likelihood_fn):
    """Combine a prior and likelihood into a log-posterior.

    Returns -inf if either component is non-finite.
    """
    lp = log_prior_fn()
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_fn()
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def log_posterior_mk(theta, data_dict, sample_alphas=False,
                     n_dust_params=0, sample_beta=True):
    """Unified log-posterior for all analysis types."""
    return log_posterior(
        lambda: log_prior_mk(theta, data_dict, sample_alphas, n_dust_params,
                             sample_beta),
        lambda: mk_likelihood(theta, data_dict, sample_alphas,
                              n_dust_params, sample_beta),
    )
