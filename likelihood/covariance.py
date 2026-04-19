"""Analytic Gaussian covariance estimation for power spectra.

Uses the standard Gaussian approximation:
    Cov(C_l^XY, C_l^X'Y') = [C_l^XX' * C_l^YY' + C_l^XY' * C_l^YX'] / ((2l+1) * Delta_l * f_sky)

where C_l includes both signal and noise.

Also provides MASTER coupling kernel covariance using precomputed coupling
data from pspy (Xi kernels, MCM inverse, binning). Uses the same homogeneous
MASTER formalism as ACT DR6 (Louis et al. 2025, Sec 3.4; Atkins et al. 2024),
with an efficient 4-term Wick contraction that avoids forming full [n_ell, n_ell]
matrices.
"""
import numpy as np
from tqdm import tqdm

from covariance_corrections import apply_kspace_to_covariance


def _parse_bin_def(bin_def):
    """Parse bin definition into (bin_lo, bin_hi) arrays.

    Args:
        bin_def: Either a 2D array of (bin_lo, bin_hi) pairs, or a 1D array
                 of bin edges (n_bins+1 values).

    Returns:
        (bin_lo, bin_hi): 1D arrays of bin boundaries.
    """
    if bin_def.ndim == 2:
        return bin_def[:, 0], bin_def[:, 1]
    return bin_def[:-1], bin_def[1:]


def build_detector_spectrum_arrays(detector_labels, auto_spectra, cross_spectra,
                                   n_ell):
    """Build [n_ell, n_det, n_det] EE and BB arrays from auto and cross spectra.

    Args:
        detector_labels: Ordered list of detector labels.
        auto_spectra: Dict keyed by detector label -> {'EE': array, 'BB': array}.
        cross_spectra: Dict keyed by (det_a, det_b) -> {'EE': array, 'BB': array}.
        n_ell: Number of ell values (length of spectrum arrays).

    Returns:
        (cl_ee, cl_bb): Arrays of shape [n_ell, n_det, n_det].
    """
    n_det = len(detector_labels)
    detector_map = {label: idx for idx, label in enumerate(detector_labels)}

    cl_ee = np.zeros((n_ell, n_det, n_det))
    cl_bb = np.zeros((n_ell, n_det, n_det))

    # Diagonal: auto-spectra (signal + noise)
    for det_label in detector_labels:
        idx = detector_map[det_label]
        auto = auto_spectra[det_label]
        n = min(n_ell, len(auto['EE']))
        cl_ee[:n, idx, idx] = auto['EE'][:n]
        cl_bb[:n, idx, idx] = auto['BB'][:n]

    # Off-diagonal: cross-spectra (signal-only)
    for (det_a, det_b), spectra in cross_spectra.items():
        if det_a in detector_map and det_b in detector_map:
            ia, ib = detector_map[det_a], detector_map[det_b]
            n = min(n_ell, len(spectra['EE']))
            cl_ee[:n, ia, ib] = spectra['EE'][:n]
            cl_ee[:n, ib, ia] = spectra['EE'][:n]
            cl_bb[:n, ia, ib] = spectra['BB'][:n]
            cl_bb[:n, ib, ia] = spectra['BB'][:n]

    # Verify all off-diagonal entries filled
    for i in range(n_det):
        for j in range(i + 1, n_det):
            if np.all(cl_ee[:, i, j] == 0):
                raise ValueError(
                    f"Missing EE cross-spectrum for {detector_labels[i]} x "
                    f"{detector_labels[j]}.")

    return cl_ee, cl_bb



def bin_detector_spectra(cl_full, ell_values, bin_def):
    """Bin per-ell detector spectra [n_ell, n_det, n_det] → [n_bins, n_det, n_det].

    Uses [lo, hi) convention matching _parse_bin_def / compute_ek_covariance.

    Args:
        cl_full: [n_ell, n_det, n_det] spectra.
        ell_values: 1D array of ell values (length n_ell).
        bin_def: 2D array of (bin_lo, bin_hi) or 1D bin edges.

    Returns:
        [n_bins, n_det, n_det] binned spectra.
    """
    bin_lo, bin_hi = _parse_bin_def(bin_def)
    n_bins = len(bin_lo)
    n_det = cl_full.shape[1]
    binned = np.zeros((n_bins, n_det, n_det))

    for b in range(n_bins):
        mask = (ell_values >= bin_lo[b]) & (ell_values < bin_hi[b])
        if np.any(mask):
            binned[b] = np.mean(cl_full[mask], axis=0)

    return binned


def compute_ek_covariance(cl_ee, cl_bb, cross_idx_map_A, ell_values=None,
                          f_sky_A=None, bin_def=None, cross_idx_map_B=None,
                          f_sky_B=None, f_sky_overlap=None,
                          kspace_F_inv=None, coupling_data=None):
    """Compute Gaussian covariance (E&K 2022 Eq. 20-21), binned.

    Handles both the square case (single group of cross-spectra) and the
    rectangular cross-block case (two groups on different sky patches).

    Two mode-counting methods:

    1. Eq. 21 (coupling_data=None): per-ell factor 1/((2l+1)*f_sky), summed
       over ells in each bin and divided by n_ell². Requires per-ell spectra,
       ell_values, f_sky_A, and bin_def.

    2. Coupling kernel (coupling_data=dict): precomputed Xi coupling kernel,
       MCM inverse, and binning matrix. Uses actual per-ell spectra
       [n_ell, n_det, n_det] — no pre-binning needed. Properly accounts for
       mask geometry via the MASTER formalism.

    Gaussian approximation (E&K Eq. 20-21):
        Cov(C_EE^{ij}, C_EE^{pq}) = [C_EE^{ip} C_EE^{jq} + C_EE^{iq} C_EE^{jp}] * K(b)
        Cov(C_EB^{ij}, C_EB^{pq}) = C_EE^{ip} C_BB^{jq} * K(b)
    where (i,j) index group A and (p,q) index group B.

    Optional F_b^{-1} kspace correction (ACT DR6 Eq. C8) is applied from
    both sides after binning. Only supported for square mode.

    Args:
        cl_ee: EE spectra [n_ell, n_det, n_det] (per-ell for both modes).
        cl_bb: BB spectra. Same shape as cl_ee.
        cross_idx_map_A: [n_cross_A, 2] int array mapping cross-spectrum index
                         to (E-source det, B-source det) indices.
        ell_values: 1D array of ell values. Required for mk mode.
        f_sky_A: Sky fraction for group A. Required for mk mode.
        bin_def: Either a 2D array of (bin_lo, bin_hi) pairs, or an array of
                 bin edges (n_bins+1 values). Required for mk mode.
        cross_idx_map_B: [n_cross_B, 2] int array for group B. If None,
                         square mode: B = A.
        f_sky_B: Sky fraction for group B. Required if cross_idx_map_B given.
        f_sky_overlap: Sky fraction of A intersection B overlap. Required if
                       cross_idx_map_B given. For square mode, equals f_sky_A.
        kspace_F_inv: If not None, [n_cross_A, n_bins, 3, 3] array of F^{-1}
                      sub-blocks (EE/EB/BB ordering) per cross-spectrum pair.
                      Only valid for square mode.
        coupling_data: If not None, dict from load_coupling_data() containing
                       Xi coupling kernel(s), mbb_inv_spin2, binning matrix P,
                       and n_bins. When provided, uses the coupling kernel
                       approach with per-ell spectra.

    Returns:
        (cov_ee, cov_bb, cov_eb): Each [n_bins, n_cross_A, n_cross_B].
    """
    # Determine square vs rectangular mode
    if cross_idx_map_B is None:
        cross_idx_map_B = cross_idx_map_A
        f_sky_B = f_sky_A
        f_sky_overlap = f_sky_A
    else:
        if kspace_F_inv is not None:
            raise ValueError("kspace_F_inv only supported for square mode")

    n_cross_A = cross_idx_map_A.shape[0]
    n_cross_B = cross_idx_map_B.shape[0]

    # Detector indices: (i,j) for group A, (p,q) for group B
    i_A = cross_idx_map_A[:, 0]
    j_A = cross_idx_map_A[:, 1]
    p_B = cross_idx_map_B[:, 0]
    q_B = cross_idx_map_B[:, 1]

    if coupling_data is not None:
        # MASTER coupling kernel mode: precomputed Xi + MCM from pspy
        n_bins = coupling_data['n_bins']
        cov_ee, cov_bb, cov_eb = _wick_contraction_coupling_kernel(
            cl_ee, cl_bb, i_A, j_A, p_B, q_B,
            n_bins, n_cross_A, n_cross_B, coupling_data)
    else:
        # Eq. 21 mode: per-ell spectra [n_ell, n_det, n_det]
        bin_lo, bin_hi = _parse_bin_def(bin_def)
        n_bins = len(bin_lo)
        cov_ee, cov_bb, cov_eb = _wick_contraction_per_ell(
            cl_ee, cl_bb, ell_values, i_A, j_A, p_B, q_B,
            bin_lo, bin_hi, n_bins, n_cross_A, n_cross_B,
            f_sky_A, f_sky_B, f_sky_overlap)

    # F_b^{-1} kspace correction (ACT DR6 Eq. C8), applied from both sides
    if kspace_F_inv is not None:
        cov_ee, cov_bb, cov_eb = apply_kspace_to_covariance(
            cov_ee, cov_bb, cov_eb, kspace_F_inv, n_bins, n_cross_A)

    return cov_ee, cov_bb, cov_eb


def _extract_cross_indices(ee, bb, i_A, j_A, p_B, q_B):
    """Extract the 8 detector-index subsets needed for Wick contractions.

    Given [n_det, n_det] spectrum matrices ee and bb, returns the
    [n_cross_A, n_cross_B] slices for all index combinations (ip, jq, iq, jp).
    """
    return (ee[i_A][:, p_B], ee[j_A][:, q_B],
            ee[i_A][:, q_B], ee[j_A][:, p_B],
            bb[i_A][:, p_B], bb[j_A][:, q_B],
            bb[i_A][:, q_B], bb[j_A][:, p_B])


def _wick_contraction_per_ell(cl_ee, cl_bb, ell_values, i_A, j_A, p_B, q_B,
                              bin_lo, bin_hi, n_bins, n_cross_A, n_cross_B,
                              f_sky_A, f_sky_B, f_sky_overlap):
    """Wick contraction with per-ell 1/((2l+1)*f_sky) mode counting (Eq. 21)."""
    cov_ee = np.zeros((n_bins, n_cross_A, n_cross_B))
    cov_bb = np.zeros((n_bins, n_cross_A, n_cross_B))
    cov_eb = np.zeros((n_bins, n_cross_A, n_cross_B))

    for b in tqdm(range(n_bins), desc="  Covariance bins"):
        ell_mask = (ell_values >= bin_lo[b]) & (ell_values < bin_hi[b])
        ells_in_bin = ell_values[ell_mask]
        n_ell_bin = len(ells_in_bin)
        if n_ell_bin == 0:
            continue

        ee_bin = cl_ee[ell_mask]  # [n_ell_bin, n_det, n_det]
        bb_bin = cl_bb[ell_mask]

        for ell_idx in range(n_ell_bin):
            ee = ee_bin[ell_idx]  # [n_det, n_det]
            bb = bb_bin[ell_idx]
            # f_overlap / ((2l+1) * f_A * f_B)
            factor = f_sky_overlap / (
                (2.0 * ells_in_bin[ell_idx] + 1.0) * f_sky_A * f_sky_B)

            (ee_ip, ee_jq, ee_iq, ee_jp,
             bb_ip, bb_jq, bb_iq, bb_jp) = _extract_cross_indices(
                ee, bb, i_A, j_A, p_B, q_B)

            cov_ee[b] += (ee_ip * ee_jq + ee_iq * ee_jp) * factor
            cov_bb[b] += (bb_ip * bb_jq + bb_iq * bb_jp) * factor
            cov_eb[b] += ee_ip * bb_jq * factor

        # Binning: P_bl = 1/n_ell_bin applied from both sides
        cov_ee[b] /= n_ell_bin ** 2
        cov_bb[b] /= n_ell_bin ** 2
        cov_eb[b] /= n_ell_bin ** 2

    return cov_ee, cov_bb, cov_eb


def _sym_spectrum(cl, i, j):
    """Symmetrize spectrum: (C[i,j] + C[j,i]) / 2.

    For per-ell arrays [n_ell, n_det, n_det], returns [n_ell].
    """
    return 0.5 * (cl[:, i, j] + cl[:, j, i])


def _binned_sym_cov(E, B, P, Xi, Xi_right):
    """Compute P @ [sym(E) * sym(B) * Xi] @ P^T using the 4-term expansion.

    Symmetrized spectra (Atkins et al. 2024 Eq. 7):
        sym(E)[l,l'] = (E[l] + E[l']) / 2
        sym(B)[l,l'] = (B[l] + B[l']) / 2

    Product: sym(E)*sym(B) = (E_l*B_l + E_l*B_l' + E_l'*B_l + E_l'*B_l') / 4

    Each term is a diagonal-scaled Xi, binned from both sides:
        term1 = (P * E*B) @ Xi_right         (row-weighted Xi)
        term2 = (P * E) @ Xi @ (P * B)^T     (E rows, B cols)
        term3 = term2^T                       (Xi symmetric)
        term4 = P @ Xi_right * (E*B)          (col-weighted Xi) = term1^T

    So: result = (term1 + term2 + term2^T + term1^T) / 4
              = (sym(term1) + sym(term2)) / 2

    This avoids forming the full [n_ell, n_ell] symmetrized matrix; cost
    is O(n_bins * n_ell) per term instead of O(n_ell^2).

    Args:
        E: [n_ell] first spectrum.
        B: [n_ell] second spectrum.
        P: [n_bins, n_ell] binning matrix.
        Xi: [n_ell, n_ell] coupling kernel.
        Xi_right: [n_ell, n_bins] = Xi @ P^T (precomputed).

    Returns:
        [n_bins, n_bins] binned pseudo-covariance.
    """
    EB = E * B
    term1 = (P * EB[None, :]) @ Xi_right       # [n_bins, n_bins]
    term2 = (P * E[None, :]) @ Xi @ (P * B[None, :]).T  # [n_bins, n_bins]
    return (term1 + term1.T + term2 + term2.T) / 4


def _fast_binned_sym_cov(E, B, PE_Xi, P, Xi_right):
    """Like _binned_sym_cov but with precomputed PE_Xi = (P * E) @ Xi.

    Avoids the expensive O(n_bins * n_ell^2) matmul (P * E) @ Xi per call,
    reducing it to O(n_bins^2 * n_ell) for PE_Xi @ (P * B)^T.

    Args:
        E: [n_ell] first spectrum (for term1 only).
        B: [n_ell] second spectrum.
        PE_Xi: [n_bins, n_ell] = (P * E) @ Xi (precomputed).
        P: [n_bins, n_ell] binning matrix.
        Xi_right: [n_ell, n_bins] = Xi @ P^T (precomputed).

    Returns:
        [n_bins, n_bins] binned pseudo-covariance.
    """
    EB = E * B
    term1 = (P * EB[None, :]) @ Xi_right       # [n_bins, n_bins]
    term2 = PE_Xi @ (P * B[None, :]).T          # [n_bins, n_bins]
    return (term1 + term1.T + term2 + term2.T) / 4


def _wick_contraction_coupling_kernel(cl_ee, cl_bb, i_A, j_A, p_B, q_B,
                                       n_bins, n_cross_A, n_cross_B,
                                       coupling_data):
    """Wick contraction using precomputed coupling kernels and MCM inverse.

    Uses the same homogeneous MASTER formalism as ACT DR6 (Atkins et al. 2024),
    with an efficient 4-term expansion that avoids forming full [n_ell, n_ell]
    symmetrized matrices. The coupling kernel Xi and MCM inverse are precomputed
    by pspy in spectrum_pipeline/.

    For each cross-pair (vi, hi):
        pseudo_cov[b, b'] = P @ [sym(C^ip) * sym(C^jq) * Xi] @ P^T
        final = mbb_inv @ pseudo_cov @ mbb_inv^T

    Args:
        cl_ee: [n_ell, n_det, n_det] per-ell EE spectra.
        cl_bb: [n_ell, n_det, n_det] per-ell BB spectra.
        i_A, j_A: [n_cross_A] E-source and B-source detector indices for group A.
        p_B, q_B: [n_cross_B] E-source and B-source detector indices for group B.
        n_bins: Number of output bins.
        n_cross_A, n_cross_B: Number of cross-spectra in groups A and B.
        coupling_data: Dict with:
            'Xi_1': [n_ell, n_ell] primary coupling kernel (first Wick term).
            'Xi_2': [n_ell, n_ell] second coupling kernel (second Wick term).
            'mbb_inv_spin2': [4*n_bins, 4*n_bins] MCM inverse (spin2xspin2).
                             Or tuple (mbb_inv_ab, mbb_inv_cd) for cross-mask.
            'P': [n_bins, n_ell] binning matrix.
            'n_bins': int.

    Returns:
        (cov_ee, cov_bb, cov_eb): Each [n_bins, n_cross_A, n_cross_B].
    """
    Xi_1 = coupling_data['Xi_1']
    Xi_2 = coupling_data['Xi_2']
    mbb_inv_raw = coupling_data['mbb_inv_spin2']
    P = coupling_data['P']
    nb = n_bins

    if isinstance(mbb_inv_raw, tuple):
        mbb_inv_ab, mbb_inv_cd = mbb_inv_raw
    else:
        mbb_inv_ab = mbb_inv_raw
        mbb_inv_cd = mbb_inv_raw

    # Align ell ranges: spectra may be shorter than Xi
    n_ell_data = cl_ee.shape[0]
    n_ell_xi = Xi_1.shape[0]
    if n_ell_data < n_ell_xi:
        Xi_1 = Xi_1[:n_ell_data, :n_ell_data]
        Xi_2 = Xi_2[:n_ell_data, :n_ell_data]
        P = P[:, :n_ell_data]
    elif n_ell_data > n_ell_xi:
        cl_ee = cl_ee[:n_ell_xi]
        cl_bb = cl_bb[:n_ell_xi]

    Xi_right_1 = Xi_1 @ P.T
    Xi_right_2 = Xi_2 @ P.T

    # ── Step 1: Symmetrize spectra ──
    # _sym_spectrum(cl, i, j) = 0.5*(cl[:,i,j] + cl[:,j,i]).
    # Bulk-symmetrize once so the inner loop uses cheap array slicing.
    cl_ee_sym = 0.5 * (cl_ee + cl_ee.swapaxes(1, 2))  # [n_ell, n_det, n_det]
    cl_bb_sym = 0.5 * (cl_bb + cl_bb.swapaxes(1, 2))
    n_ell = cl_ee_sym.shape[0]
    n_det = cl_ee_sym.shape[1]

    # ── Step 2: Precompute (P * spec) @ Xi for all detector pairs ──
    # The bottleneck in _binned_sym_cov is (P * E) @ Xi: O(n_bins * n_ell^2).
    # With n_det detectors there are n_det^2 unique spectra, so we precompute
    # PsXi[a, b] = (P * cl_sym[:, a, b]) @ Xi for each (a, b) pair.
    # This moves the O(n_ell^2) cost out of the 144K-iteration inner loop.
    #
    # To maximise BLAS thread utilisation, we reshape all n_det^2 weighted
    # spectra into one large 2D matmul: [n_det^2 * nb, n_ell] @ [n_ell, n_ell].
    same_xi = Xi_1 is Xi_2
    n_pairs = n_det * n_det
    print(f"  Precomputing (P * spec) @ Xi for {n_det} detectors "
          f"({n_pairs} pairs, {'same Xi' if same_xi else 'two Xi kernels'})...")

    # cl_sym is [n_ell, n_det, n_det] → reshape to [n_det^2, n_ell]
    ee_flat = cl_ee_sym.reshape(n_ell, n_pairs).T   # [n_pairs, n_ell]
    bb_flat = cl_bb_sym.reshape(n_ell, n_pairs).T

    # Weighted P: [n_pairs, nb, n_ell] → reshape to [n_pairs*nb, n_ell]
    weighted_ee = (P[None, :, :] * ee_flat[:, None, :]).reshape(n_pairs * nb, n_ell)
    weighted_bb = (P[None, :, :] * bb_flat[:, None, :]).reshape(n_pairs * nb, n_ell)

    # Single large 2D matmul — fully threaded by BLAS
    PsXi1_ee = (weighted_ee @ Xi_1).reshape(n_det, n_det, nb, n_ell)
    PsXi1_bb = (weighted_bb @ Xi_1).reshape(n_det, n_det, nb, n_ell)
    if same_xi:
        PsXi2_ee = PsXi1_ee
        PsXi2_bb = PsXi1_bb
    else:
        PsXi2_ee = (weighted_ee @ Xi_2).reshape(n_det, n_det, nb, n_ell)
        PsXi2_bb = (weighted_bb @ Xi_2).reshape(n_det, n_det, nb, n_ell)
    del weighted_ee, weighted_bb, ee_flat, bb_flat

    # ── Step 3: Main Wick contraction loop ──
    cov_ee = np.zeros((n_bins, n_cross_A, n_cross_B))
    cov_bb = np.zeros((n_bins, n_cross_A, n_cross_B))
    cov_eb = np.zeros((n_bins, n_cross_A, n_cross_B))

    for vi in tqdm(range(n_cross_A), desc="  Covariance (coupling kernel)"):
        i = i_A[vi]
        j = j_A[vi]
        for hi in range(n_cross_B):
            p = p_B[hi]
            q = q_B[hi]

            E_ip = cl_ee_sym[:, i, p]
            E_jq = cl_ee_sym[:, j, q]
            E_iq = cl_ee_sym[:, i, q]
            E_jp = cl_ee_sym[:, j, p]
            B_ip = cl_bb_sym[:, i, p]
            B_jq = cl_bb_sym[:, j, q]
            B_iq = cl_bb_sym[:, i, q]
            B_jp = cl_bb_sym[:, j, p]

            pseudo_EB = _fast_binned_sym_cov(
                E_ip, B_jq, PsXi1_ee[i, p], P, Xi_right_1)
            pseudo_BE = _fast_binned_sym_cov(
                B_ip, E_jq, PsXi1_bb[i, p], P, Xi_right_1)
            pseudo_EE = (_fast_binned_sym_cov(
                             E_ip, E_jq, PsXi1_ee[i, p], P, Xi_right_1)
                         + _fast_binned_sym_cov(
                             E_iq, E_jp, PsXi2_ee[i, q], P, Xi_right_2))
            pseudo_BB = (_fast_binned_sym_cov(
                             B_ip, B_jq, PsXi1_bb[i, p], P, Xi_right_1)
                         + _fast_binned_sym_cov(
                             B_iq, B_jp, PsXi2_bb[i, q], P, Xi_right_2))

            pseudo_full = np.zeros((4 * nb, 4 * nb))
            pseudo_full[0*nb:1*nb, 0*nb:1*nb] = pseudo_EE
            pseudo_full[1*nb:2*nb, 1*nb:2*nb] = pseudo_EB
            pseudo_full[2*nb:3*nb, 2*nb:3*nb] = pseudo_BE
            pseudo_full[3*nb:4*nb, 3*nb:4*nb] = pseudo_BB

            final = mbb_inv_ab @ pseudo_full @ mbb_inv_cd.T

            cov_ee[:, vi, hi] = np.diag(final[0*nb:1*nb, 0*nb:1*nb])
            cov_eb[:, vi, hi] = np.diag(final[1*nb:2*nb, 1*nb:2*nb])
            cov_bb[:, vi, hi] = np.diag(final[3*nb:4*nb, 3*nb:4*nb])

    return cov_ee, cov_bb, cov_eb


# ============================================================
# Coupling kernel loading and utilities
# ============================================================

def load_coupling_data(coupling_path):
    """Load precomputed coupling data from .npz file.

    The .npz file is produced by spectrum_pipeline/compute_coupling.py
    (which uses pspy's cov_coupling_spin0and2_simple and mcm_and_bbl_spin0and2).

    Handles both same-mask (single Xi) and cross-mask (two Xi) formats.

    Args:
        coupling_path: Path to .npz file.

    Returns:
        dict with:
            'Xi_1': [n_ell, n_ell] primary coupling kernel.
            'Xi_2': [n_ell, n_ell] second kernel (= Xi_1 for same-mask).
            'mbb_inv_spin2': [4*n_bins, 4*n_bins] MCM inverse (spin2xspin2).
                             Or tuple for cross-mask.
            'P': [n_bins, n_ell] binning matrix.
            'n_bins': int.
    """
    data = np.load(coupling_path, allow_pickle=True)

    # Coupling kernel(s)
    if 'coupling_Xi_1' in data:
        Xi_1 = data['coupling_Xi_1']
        Xi_2 = data['coupling_Xi_2']
    elif 'coupling_Xi' in data:
        Xi_1 = data['coupling_Xi']
        Xi_2 = Xi_1
    else:
        raise KeyError("No coupling kernel found in .npz")

    # MCM inverse: extract spin2xspin2 block
    if 'mbb_inv' in data:
        mbb_inv_spin2 = _extract_mbb_inv_spin2(data['mbb_inv'])
    elif 'mbb_inv_ab' in data:
        mbb_inv_spin2 = (
            _extract_mbb_inv_spin2(data['mbb_inv_ab']),
            _extract_mbb_inv_spin2(data['mbb_inv_cd']))
    else:
        raise KeyError("No MCM inverse found in .npz")

    if isinstance(mbb_inv_spin2, tuple):
        n_bins = mbb_inv_spin2[0].shape[0] // 4
    else:
        n_bins = mbb_inv_spin2.shape[0] // 4

    bin_lo = data['bin_lo'][:n_bins]
    bin_hi = data['bin_hi'][:n_bins]
    n_ell = Xi_1.shape[0]
    P = _build_binning_matrix(bin_lo, bin_hi, n_ell)

    return {
        'Xi_1': Xi_1,
        'Xi_2': Xi_2,
        'mbb_inv_spin2': mbb_inv_spin2,
        'P': P,
        'n_bins': n_bins,
    }


def _build_binning_matrix(bin_lo, bin_hi, n_ell):
    """Construct the binning matrix P[b, ell] = 1/N_b for ell in bin b."""
    n_bins = len(bin_lo)
    P = np.zeros((n_bins, n_ell))
    ells = np.arange(n_ell)
    for b in range(n_bins):
        mask = (ells >= bin_lo[b]) & (ells < bin_hi[b])
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            P[b, mask] = 1.0 / n_in_bin
    return P


def _extract_mbb_inv_spin2(mbb_inv):
    """Extract spin2xspin2 block [4*n_bins, 4*n_bins] from pspy mbb_inv dict."""
    if isinstance(mbb_inv, dict):
        return mbb_inv['spin2xspin2']
    if mbb_inv.ndim == 0:
        inner = mbb_inv.item()
        if isinstance(inner, dict):
            return inner['spin2xspin2']
    return mbb_inv
