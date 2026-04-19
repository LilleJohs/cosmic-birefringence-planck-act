"""Shared M&K likelihood core for all analysis types.

Implements the common chi2 + logdet computation from Minami & Komatsu (2020).
Each analysis module provides its own alpha-mapping logic and calls
the shared functions here.
"""

import numpy as np
import scipy.linalg
from rotation import compute_A_B_vectorized
from config import (
    BETA_PRIOR_MAX_DEG,
    ACT_ALPHA_PRIOR_STD,
    ACT_ALPHA_CORRELATED, ACT_ALPHA_CORRELATION_COEFF,
    NPIPE_HFI_ALPHA_PRIOR_STD, NPIPE_HFI_ALPHA_CORRELATION,
    NPIPE_LFI_ALPHA_PRIOR_STD,
)
from tools.data_loading import NPIPE_TIME_SPLIT_FREQS, NPIPE_DUST_FREQS



def precompute_bin_groups(data_dict):
    """Group bins by their active cross-spectrum set for batched linalg.

    Bins with identical active sets get batched into one np.linalg.solve call,
    which LAPACK parallelizes across cores internally.

    Stores result in data_dict['bin_groups']: list of (active_idx, bin_indices).
    """
    active_bins = data_dict['active_bins']
    n_bins = data_dict['n_bins']

    groups = {}
    for b in range(n_bins):
        active = active_bins[:, b]
        if not np.any(active):
            continue
        key = tuple(np.where(active)[0])
        if key not in groups:
            groups[key] = []
        groups[key].append(b)

    data_dict['bin_groups'] = [
        (np.array(active_key), np.array(bin_list))
        for active_key, bin_list in groups.items()
    ]


def _compute_residuals_and_covariance(alpha_i, alpha_j, beta, cache, f_ell):
    """Compute residual vector and rotated covariance matrix.

    Shared core for both log-likelihood (needs logdet) and chi2 (does not).

    The residual is v = A . C_obs - B . C_theory, and the rotated covariance
    is M = A . Cov . A^T, expanded using the block-diagonal structure:
        M[vi,hi] = a0[vi]*a0[hi]*cov_ee + a1[vi]*a1[hi]*cov_bb + a2[vi]*a2[hi]*cov_eb

    Two code paths:
    - f_ell=None: A/B are (n_cross, 3/2), ell-independent (fast path for ACT)
    - f_ell provided: A/B are (n_bins, n_cross, 3/2), per-bin (NPIPE dust model)

    Returns:
        (v_all, M_full): residuals [n_bins, n_cross] and covariance [n_bins, n_cross, n_cross]
    """
    A_vectors, B_vectors = compute_A_B_vectorized(alpha_i, alpha_j, beta, f_ell=f_ell)

    if f_ell is None:
        # Fast path: A/B are (n_cross, 3/2), ell-independent
        v_all = (np.einsum('bik,ik->bi', cache['C_obs_3d'], A_vectors) -
                 np.einsum('bik,ik->bi', cache['C_theory_3d'], B_vectors))

        a0, a1, a2 = A_vectors[:, 0], A_vectors[:, 1], A_vectors[:, 2]
        M_full = (np.outer(a0, a0)[np.newaxis] * cache['cov_ee'] +
                  np.outer(a1, a1)[np.newaxis] * cache['cov_bb'] +
                  np.outer(a2, a2)[np.newaxis] * cache['cov_eb'])
    else:
        # Per-bin path: A/B are (n_bins, n_cross, 3/2)
        v_all = (np.einsum('bik,bik->bi', cache['C_obs_3d'], A_vectors) -
                 np.einsum('bik,bik->bi', cache['C_theory_3d'], B_vectors))

        a0 = A_vectors[:, :, 0]  # (n_bins, n_cross)
        a1 = A_vectors[:, :, 1]
        a2 = A_vectors[:, :, 2]
        M_full = (a0[:, :, None] * a0[:, None, :] * cache['cov_ee'] +
                  a1[:, :, None] * a1[:, None, :] * cache['cov_bb'] +
                  a2[:, :, None] * a2[:, None, :] * cache['cov_eb'])

    return v_all, M_full


def _build_full_cov_system(alpha_i, alpha_j, beta, data_dict, f_ell=None):
    """Build flattened residual and full rotated covariance (off-diagonal bins).

    Used when the covariance includes bin-to-bin correlations (4D structure).
    Only supports ell-independent A vectors (f_ell=None).

    Returns:
        (v_flat, M_2d): masked 1D residual and 2D covariance.
    """
    cache = data_dict['likelihood_cache']
    A_vectors, B_vectors = compute_A_B_vectorized(
        alpha_i, alpha_j, beta, f_ell=f_ell)

    v_all = (np.einsum('bik,ik->bi', cache['C_obs_3d'], A_vectors) -
             np.einsum('bik,ik->bi', cache['C_theory_3d'], B_vectors))

    a0, a1, a2 = A_vectors[:, 0], A_vectors[:, 1], A_vectors[:, 2]
    n_bins, n_cross = v_all.shape

    # M[(b,k),(b',k')] = a0[k]*a0[k']*cov_ee[b,k,b',k'] + ...
    a0a0 = np.outer(a0, a0)[np.newaxis, :, np.newaxis, :]
    a1a1 = np.outer(a1, a1)[np.newaxis, :, np.newaxis, :]
    a2a2 = np.outer(a2, a2)[np.newaxis, :, np.newaxis, :]
    M_4d = (a0a0 * cache['cov_ee_full'] +
            a1a1 * cache['cov_bb_full'] +
            a2a2 * cache['cov_eb_full'])

    M_2d = M_4d.reshape(n_bins * n_cross, n_bins * n_cross)
    v_flat = v_all.ravel()

    if 'active_bins' in data_dict:
        # active_bins is (n_cross, n_bins); transpose to (n_bins, n_cross) for ravel
        active = data_dict['active_bins'].T.ravel()
        v_flat = v_flat[active]
        M_2d = M_2d[np.ix_(active, active)]

    return v_flat, M_2d


def mk_log_likelihood(alpha_i, alpha_j, beta, data_dict, f_ell=None):
    """Compute the M&K log-likelihood.

    This is the shared core used by all analysis types. The caller is
    responsible for mapping theta -> (alpha_i, alpha_j, beta).

    Automatically dispatches to full bin-to-bin covariance path when
    'cov_ee_full' is present in the likelihood cache (ACT DR6 with
    ACT_DR6_OFFDIAG_COV=True).

    Args:
        alpha_i: (n_cross,) alpha angles for detector i of each cross-pair
        alpha_j: (n_cross,) alpha angles for detector j of each cross-pair
        beta: scalar birefringence angle (radians)
        data_dict: Must contain 'likelihood_cache' and either 'bin_groups'
                   (for active-bin masking) or not (all bins fully active).
        f_ell: None or (n_bins,) dust f values per analysis bin

    Returns:
        Log-likelihood value (float), or -inf on failure.
    """
    # Full bin-to-bin covariance path
    if 'cov_ee_full' in data_dict.get('likelihood_cache', {}):
        v_flat, M_2d = _build_full_cov_system(
            alpha_i, alpha_j, beta, data_dict, f_ell)
        try:
            L, info = scipy.linalg.lapack.dpotrf(M_2d, lower=True)
            if info != 0:
                return -np.inf
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            y = scipy.linalg.solve_triangular(L, v_flat, lower=True)
            chi2 = np.dot(y, y)
            return -0.5 * (chi2 + logdet)
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf

    # Diagonal-in-bin path
    v_all, M_full = _compute_residuals_and_covariance(
        alpha_i, alpha_j, beta, data_dict['likelihood_cache'], f_ell)

    if 'bin_groups' in data_dict:
        return _solve_with_active_bins(v_all, M_full, data_dict['bin_groups'])
    else:
        return _solve_all_bins(v_all, M_full)


def mk_chi2(alpha_i, alpha_j, beta, data_dict, f_ell=None):
    """Compute chi2 and degrees of freedom (without logdet term).

    Used for goodness-of-fit reporting, not for sampling.

    Returns:
        (chi2_total, n_data)
    """
    # Full bin-to-bin covariance path
    if 'cov_ee_full' in data_dict.get('likelihood_cache', {}):
        v_flat, M_2d = _build_full_cov_system(
            alpha_i, alpha_j, beta, data_dict, f_ell)
        M_inv_v = np.linalg.solve(M_2d, v_flat)
        chi2_total = float(np.dot(v_flat, M_inv_v))
        return chi2_total, len(v_flat)

    # Diagonal-in-bin path
    v_all, M_full = _compute_residuals_and_covariance(
        alpha_i, alpha_j, beta, data_dict['likelihood_cache'], f_ell)

    if 'bin_groups' in data_dict:
        chi2_total = 0.0
        n_data = 0
        for active_idx, bin_idx in data_dict['bin_groups']:
            v_batch = v_all[np.ix_(bin_idx, active_idx)]
            M_batch = M_full[np.ix_(bin_idx, active_idx, active_idx)]
            M_inv_v = np.linalg.solve(M_batch, v_batch)
            chi2_total += float(np.einsum('bi,bi->', v_batch, M_inv_v))
            n_data += len(bin_idx) * len(active_idx)
        return chi2_total, n_data
    else:
        M_inv_v = np.linalg.solve(M_full, v_all)
        chi2_total = float(np.einsum('bi,bi->', v_all, M_inv_v))
        n_data = v_all.shape[0] * v_all.shape[1]
        return chi2_total, n_data


def _cholesky_chi2_logdet(M_batch, v_batch):
    """Compute chi2 and logdet from a single Cholesky factorization.

    One Cholesky replaces separate solve (LU) + slogdet (LU), halving
    the O(n^3) work per bin group.

    Args:
        M_batch: (n_batch, n, n) positive-definite covariance matrices
        v_batch: (n_batch, n) residual vectors

    Returns:
        (chi2, logdet) arrays of shape (n_batch,), or None on failure.
    """
    n_batch, n, _ = M_batch.shape
    chi2 = np.empty(n_batch)
    logdet = np.empty(n_batch)
    for b in range(n_batch):
        L, info = scipy.linalg.lapack.dpotrf(M_batch[b], lower=True)
        if info != 0:
            return None
        logdet[b] = 2.0 * np.sum(np.log(np.diag(L)))
        y = scipy.linalg.solve_triangular(L, v_batch[b], lower=True)
        chi2[b] = np.dot(y, y)
    return chi2, logdet


def _solve_all_bins(v_all, M_all):
    """Batched solve when all cross-spectra are active in every bin (e.g. NPIPE)."""
    try:
        result = _cholesky_chi2_logdet(M_all, v_all)
        if result is None:
            return -np.inf
        chi2_all, logdet = result
        return -0.5 * np.sum(chi2_all + logdet)
    except np.linalg.LinAlgError:
        return -np.inf


def _solve_with_active_bins(v_all, M_full, bin_groups):
    """Batched solve with active-bin masking.

    Groups bins by their active set and batches each group into a single
    Cholesky factorization, enabling LAPACK multi-threading.
    """
    total = 0.0
    try:
        for active_idx, bin_idx in bin_groups:
            v_batch = v_all[np.ix_(bin_idx, active_idx)]
            M_batch = M_full[np.ix_(bin_idx, active_idx, active_idx)]

            result = _cholesky_chi2_logdet(M_batch, v_batch)
            if result is None:
                return -np.inf
            chi2, logdet = result

            total += np.sum(chi2 + logdet)

        return -0.5 * total
    except np.linalg.LinAlgError:
        return -np.inf


def log_beta_prior(beta):
    """Flat prior on beta in [-BETA_PRIOR_MAX_DEG, +BETA_PRIOR_MAX_DEG] (radians)."""
    beta_max = np.deg2rad(BETA_PRIOR_MAX_DEG)
    if beta < -beta_max or beta > beta_max:
        return -np.inf
    return 0.0


def _is_hfi_label(label):
    """Check if an NPIPE alpha label belongs to HFI (not ACT, not LFI)."""
    freq_str = ''.join(c for c in label if c.isdigit())
    if not freq_str:
        return False
    return int(freq_str) in NPIPE_DUST_FREQS


def build_npipe_alpha_prior_cov(npipe_alpha_labels):
    """Build NPIPE alpha prior covariance matrix.

    HFI splits get correlated priors (shared focal plane absolute angle
    uncertainty from Rosset et al. 2010).  LFI splits get independent priors.

    Returns:
        (n, n) covariance matrix in radians^2.
    """
    n = len(npipe_alpha_labels)
    cov = np.zeros((n, n))
    for i, label_i in enumerate(npipe_alpha_labels):
        hfi_i = _is_hfi_label(label_i)
        std_i = np.deg2rad(
            NPIPE_HFI_ALPHA_PRIOR_STD if hfi_i else NPIPE_LFI_ALPHA_PRIOR_STD)
        cov[i, i] = std_i**2
        if hfi_i:
            for j in range(i + 1, n):
                if _is_hfi_label(npipe_alpha_labels[j]):
                    std_j = np.deg2rad(NPIPE_HFI_ALPHA_PRIOR_STD)
                    cov[i, j] = NPIPE_HFI_ALPHA_CORRELATION * std_i * std_j
                    cov[j, i] = cov[i, j]
    return cov


def log_npipe_alpha_prior(npipe_alphas, npipe_alpha_labels):
    """Multivariate Gaussian prior on NPIPE alpha angles.

    HFI splits are correlated (shared focal plane reference);
    LFI splits are independent.

    Args:
        npipe_alphas: array of NPIPE alpha values (radians)
        npipe_alpha_labels: list of NPIPE label strings

    Returns:
        Log-prior value
    """
    n = len(npipe_alpha_labels)
    cov = build_npipe_alpha_prior_cov(npipe_alpha_labels)
    L = np.linalg.cholesky(cov)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    x = np.linalg.solve(L, npipe_alphas)
    return -0.5 * np.dot(x, x) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)


def build_act_alpha_prior_cov(act_alpha_labels):
    """Build ACT alpha prior covariance matrix with optional within-tube correlations.

    If ACT_ALPHA_CORRELATED is True, alphas within the same optical tube
    (pa5-pa5 or pa6-pa6) are correlated with ACT_ALPHA_CORRELATION_COEFF.
    """
    n = len(act_alpha_labels)
    cov = np.zeros((n, n))
    for i, band_i in enumerate(act_alpha_labels):
        std_i = np.deg2rad(ACT_ALPHA_PRIOR_STD[band_i])
        cov[i, i] = std_i**2
        if ACT_ALPHA_CORRELATED:
            tube_i = band_i.split('_')[0]
            for j, band_j in enumerate(act_alpha_labels):
                if j <= i:
                    continue
                tube_j = band_j.split('_')[0]
                if tube_i == tube_j:
                    std_j = np.deg2rad(ACT_ALPHA_PRIOR_STD[band_j])
                    cov[i, j] = ACT_ALPHA_CORRELATION_COEFF * std_i * std_j
                    cov[j, i] = cov[i, j]
    return cov


def log_act_alpha_prior(act_alphas, act_alpha_labels):
    """Multivariate Gaussian prior on ACT alpha angles (with optional tube correlation).

    Args:
        act_alphas: array of ACT alpha values (radians)
        act_alpha_labels: list of ACT band label strings

    Returns:
        Log-prior value
    """
    n = len(act_alpha_labels)
    cov = build_act_alpha_prior_cov(act_alpha_labels)
    L = np.linalg.cholesky(cov)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    x = np.linalg.solve(L, act_alphas)
    return -0.5 * np.dot(x, x) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
