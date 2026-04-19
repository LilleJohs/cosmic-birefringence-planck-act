"""Eskilt & Komatsu (2022) rotation matrices and A/B formalism.

Implements Eqs. 6-18 from arXiv:2205.13962 for the EB cross-spectrum
rotation under polarization angle miscalibration (alpha) and cosmic
birefringence (beta), with optional filamentary dust EB (F matrix).
"""

import numpy as np


def compute_f_ell(A_ell_params, dust_info, n_cross):
    """Map dust amplitude parameters to per-bin, per-cross f_ell values.

    f_ell[b, k] = A_l[dust_bin_idx[b]] * sin(4 * psi_l[b]) for active bins
    and dust-applicable crosses; 0 otherwise.

    Args:
        A_ell_params: (N_DUST_BINS,) amplitude parameters A_l >= 0
        dust_info: dict with 'psi_ell_binned' (n_bins,), 'dust_bin_idx' (n_bins,),
                   and optional 'dust_cross_mask' (n_cross,) bool array
        n_cross: number of cross-spectra

    Returns:
        f_ell: (n_bins, n_cross)
    """
    psi = dust_info['psi_ell_binned']
    idx = dust_info['dust_bin_idx']
    n_bins = len(psi)

    # 1D f_ell per bin
    f_ell_1d = np.zeros(n_bins)
    active = idx >= 0
    f_ell_1d[active] = A_ell_params[idx[active]] * np.sin(4.0 * psi[active])

    # Broadcast to (n_bins, n_cross) and zero out non-dust crosses
    f_ell_2d = np.broadcast_to(f_ell_1d[:, np.newaxis], (n_bins, n_cross)).copy()
    dust_cross_mask = dust_info.get('dust_cross_mask')
    if dust_cross_mask is not None:
        f_ell_2d[:, ~dust_cross_mask] = 0.0
    return f_ell_2d


def compute_A_B_vectorized(alpha_i, alpha_j, beta, f_ell=None):
    """Compute A and B vectors for all cross-spectra at once.

    When f_ell is None (F=0), returns ell-independent shapes:
        A_vectors: (n_cross, 3), B_vectors: (n_cross, 2)

    When f_ell is provided, returns per-bin shapes:
        A_vectors: (n_bins, n_cross, 3), B_vectors: (n_bins, n_cross, 2)

    The dust model adds intrinsic C_EB = f_l * C_EE to the foreground sky
    before rotation by alpha. This modifies the Lambda matrices (Eqs. 14-18):
        Lambda_EE_col0 -= f * sin(2(alpha_i + alpha_j))
        Lambda_BB_col0 += f * sin(2(alpha_i + alpha_j))
        Lambda_EB_col0 += f * cos(2(alpha_i + alpha_j))
    The CMB part (theta = alpha + beta) uses plain R(theta) with no dust
    correction, since the CMB has no intrinsic EB (Eq. 13).

    Args:
        alpha_i: (n_cross,) alpha angles for detector i (radians)
        alpha_j: (n_cross,) alpha angles for detector j (radians)
        beta: scalar birefringence angle (radians)
        f_ell: None or (n_bins, n_cross) dust f values per bin and cross

    Returns:
        A_vectors, B_vectors
    """
    n_cross = len(alpha_i)

    c2ai = np.cos(2.0 * alpha_i)
    s2ai = np.sin(2.0 * alpha_i)
    c2aj = np.cos(2.0 * alpha_j)
    s2aj = np.sin(2.0 * alpha_j)

    # R_vec(alpha_i, alpha_j) — EB row of rotation matrix
    rv_a0 = c2ai * s2aj
    rv_a1 = -s2ai * c2aj

    # R(alpha) 2x2 block (EE/BB mixing)
    diag_a = c2ai * c2aj
    off_a = s2ai * s2aj

    # R^{-1}(alpha) amplitude factor
    amp_a = 2.0 / (np.cos(4.0 * alpha_i) + np.cos(4.0 * alpha_j))

    # theta = alpha + beta
    theta_i = alpha_i + beta
    theta_j = alpha_j + beta

    c2ti = np.cos(2.0 * theta_i)
    s2ti = np.sin(2.0 * theta_i)
    c2tj = np.cos(2.0 * theta_j)
    s2tj = np.sin(2.0 * theta_j)

    rv_t0 = c2ti * s2tj
    rv_t1 = -s2ti * c2tj

    diag_t = c2ti * c2tj
    off_t = s2ti * s2tj

    if f_ell is None:
        # F=0: ell-independent A/B vectors
        A_vectors = np.empty((n_cross, 3))
        A_vectors[:, 0] = -amp_a * (rv_a0 * diag_a - rv_a1 * off_a)
        A_vectors[:, 1] = -amp_a * (-rv_a0 * off_a + rv_a1 * diag_a)
        A_vectors[:, 2] = 1.0

        # R_product = R_inv(alpha) @ R(theta)
        rp_00 = amp_a * (diag_a * diag_t - off_a * off_t)
        rp_01 = amp_a * (diag_a * off_t - off_a * diag_t)
        rp_10 = amp_a * (-off_a * diag_t + diag_a * off_t)
        rp_11 = amp_a * (-off_a * off_t + diag_a * diag_t)

        B_vectors = np.empty((n_cross, 2))
        B_vectors[:, 0] = rv_t0 - (rv_a0 * rp_00 + rv_a1 * rp_10)
        B_vectors[:, 1] = rv_t1 - (rv_a0 * rp_01 + rv_a1 * rp_11)

        return A_vectors, B_vectors

    # f_ell provided: per-bin Lambda matrices including dust F correction
    n_bins = f_ell.shape[0]
    f = f_ell  # (n_bins, n_cross)

    # Dust correction trig quantities (foreground only, not CMB)
    s_a = np.sin(2.0 * (alpha_i + alpha_j))  # (n_cross,)
    c_a = np.cos(2.0 * (alpha_i + alpha_j))

    # Lambda(alpha) 2x2 per bin: Eq. 14, Λ = R + D·F
    # D·F adds -f·sin(2(αi+αj)) to (0,0) and +f·sin(2(αi+αj)) to (1,0)
    L_00 = diag_a[np.newaxis, :] - f * s_a[np.newaxis, :]
    L_01 = np.broadcast_to(off_a[np.newaxis, :], (n_bins, n_cross))
    L_10 = off_a[np.newaxis, :] + f * s_a[np.newaxis, :]
    L_11 = np.broadcast_to(diag_a[np.newaxis, :], (n_bins, n_cross))

    # Lambda_vec(alpha) per bin: (n_bins, n_cross)
    Lv_0 = rv_a0[np.newaxis, :] + f * c_a[np.newaxis, :]
    Lv_1 = np.broadcast_to(rv_a1[np.newaxis, :], (n_bins, n_cross))

    # R(theta) 2x2 per bin — CMB has no intrinsic EB, so no dust (Eq. 13)
    LT_00 = np.broadcast_to(diag_t[np.newaxis, :], (n_bins, n_cross))
    LT_01 = np.broadcast_to(off_t[np.newaxis, :], (n_bins, n_cross))
    LT_10 = np.broadcast_to(off_t[np.newaxis, :], (n_bins, n_cross))
    LT_11 = np.broadcast_to(diag_t[np.newaxis, :], (n_bins, n_cross))

    # R_vec(theta) per bin — plain rotation, no dust (Eq. 13)
    LTv_0 = np.broadcast_to(rv_t0[np.newaxis, :], (n_bins, n_cross))
    LTv_1 = np.broadcast_to(rv_t1[np.newaxis, :], (n_bins, n_cross))

    # Lambda^{-1} via 2x2 determinant formula
    det_L = L_00 * L_11 - L_01 * L_10
    Li_00 = L_11 / det_L
    Li_01 = -L_01 / det_L
    Li_10 = -L_10 / det_L
    Li_11 = L_00 / det_L

    # A = [-Lambda_vec @ Lambda^{-1}, 1]
    LvLi_0 = Lv_0 * Li_00 + Lv_1 * Li_10
    LvLi_1 = Lv_0 * Li_01 + Lv_1 * Li_11

    A_vectors = np.empty((n_bins, n_cross, 3))
    A_vectors[:, :, 0] = -LvLi_0
    A_vectors[:, :, 1] = -LvLi_1
    A_vectors[:, :, 2] = 1.0

    # B = Lambda_T_vec - Lambda_vec @ Lambda^{-1} @ Lambda_T
    LiLT_00 = Li_00 * LT_00 + Li_01 * LT_10
    LiLT_01 = Li_00 * LT_01 + Li_01 * LT_11
    LiLT_10 = Li_10 * LT_00 + Li_11 * LT_10
    LiLT_11 = Li_10 * LT_01 + Li_11 * LT_11

    LvLiLT_0 = Lv_0 * LiLT_00 + Lv_1 * LiLT_10
    LvLiLT_1 = Lv_0 * LiLT_01 + Lv_1 * LiLT_11

    B_vectors = np.empty((n_bins, n_cross, 2))
    B_vectors[:, :, 0] = LTv_0 - LvLiLT_0
    B_vectors[:, :, 1] = LTv_1 - LvLiLT_1

    return A_vectors, B_vectors
