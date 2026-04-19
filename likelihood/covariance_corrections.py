"""Covariance correction terms: kspace transfer function and T→P leakage.

Kspace correction (apply_kspace_to_covariance):
    F⁻¹ sandwich + mode-count inflation for ACT's ground pickup filter.
    See Atkins et al. (2024) §4.1.2 and PSpipe get_covariance_blocks.py.

Leakage covariance (compute_leakage_covariance):
    Additive covariance from T→P leakage uncertainty (γ error modes).
    Analytically equivalent to PSpipe's MC approach (get_leakage_sim.py).
"""
import numpy as np

from power_spectrum_corrections import _load_leakage_model
from tools.spectra import camb_full_theory
from tools.data_loading import load_act_beam


def apply_kspace_to_covariance(cov_ee, cov_bb, cov_eb, kspace_F_inv,
                                n_bins, n_cross, kspace_F_inv_B=None):
    """Apply per-cross-spectrum F_b⁻¹ correction to covariance blocks.

    Supports both square blocks (single group) and rectangular cross-blocks
    (two groups, e.g. Cov(AA+AP, PP)).

    Two corrections are applied:

    1. **F⁻¹ sandwich** — propagates the kspace transfer correction from
       the spectra into the covariance:

           Cov'[b, X, vi, Y, hi] = Σ_s F_A[vi, X, s] * F_B[hi, Y, s]
                                        * Cov[b, s, vi, hi]

       where s ∈ {EE=0, BB=1, EB=2}. For square blocks F_A = F_B.
       For cross-blocks with kspace_F_inv_B=None, F_B = identity.

    2. **Mode-count correction** — the kspace filter removes Fourier modes,
       reducing the effective number of independent modes per multipole bin.
       This increases the variance beyond what the F⁻¹ sandwich captures.
       Following PSpipe (get_covariance_blocks.py, Atkins et al. 2024
       §4.1.2), the correction is:

           Cov'' = Cov' / (tf_i * tf_j)^{1/4}

       where tf is the 1D kspace transfer function (EE diagonal of the F
       matrix). The rationale is that sqrt(tf) approximates the fraction of
       modes surviving the filter, so the effective mode count is
       (2ℓ+1) × f_sky × sqrt(tf). The covariance scales as 1/sqrt(N_i * N_j),
       giving a geometric mean correction (tf_i * tf_j)^{1/4}.

       NPIPE cross-spectra have F_inv = identity (tf = 1), so they are
       unaffected.

    Args:
        cov_ee, cov_bb, cov_eb: [n_bins, n_cross_A, n_cross_B]
        kspace_F_inv: [n_cross_A, n_bins, 3, 3] — F⁻¹ for row side
                      (EE=0, BB=1, EB=2 ordering)
        n_bins, n_cross: n_cross_A dimension
        kspace_F_inv_B: [n_cross_B, n_bins, 3, 3] — F⁻¹ for column side.
                        None means identity (e.g. PP block with no kspace filter).
                        For square blocks, pass None to use kspace_F_inv for both sides.

    Returns:
        Updated (cov_ee, cov_bb, cov_eb).
    """
    n_cross_A = n_cross
    n_cross_B = cov_ee.shape[2]
    n_bins = min(cov_ee.shape[0], kspace_F_inv.shape[1])
    is_square = kspace_F_inv_B is None and n_cross_A == n_cross_B

    cov_ee_new = np.zeros_like(cov_ee)
    cov_bb_new = np.zeros_like(cov_bb)
    cov_eb_new = np.zeros_like(cov_eb)

    cov_in = [cov_ee, cov_bb, cov_eb]
    cov_out = [cov_ee_new, cov_bb_new, cov_eb_new]

    # Step 1: F⁻¹ sandwich (amplitude correction)
    for b in range(n_bins):
        F_A = kspace_F_inv[:, b, :, :]  # [n_cross_A, 3, 3]
        if is_square:
            F_B = F_A
        elif kspace_F_inv_B is not None:
            F_B = kspace_F_inv_B[:, b, :, :]  # [n_cross_B, 3, 3]
        else:
            # Identity for column side (PP): F_B[hi, Y, s] = δ_{Y,s}
            F_B = np.tile(np.eye(3), (n_cross_B, 1, 1))

        # Only X=Y diagonal needed: Cov(EE,EE), Cov(BB,BB), Cov(EB,EB).
        # Cross-type blocks like Cov(EE,BB) don't enter the M&K likelihood.
        for X in range(3):
            # contrib[vi, hi] = Σ_s F_A[vi,X,s] * F_B[hi,X,s] * cov_s[b,vi,hi]
            contrib = np.zeros((n_cross_A, n_cross_B))
            for s in range(3):
                contrib += (F_A[:, X, s][:, None] *
                            F_B[:, X, s][None, :]) * cov_in[s][b]
            cov_out[X][b] = contrib

    # Step 2: Mode-count correction (Atkins et al. 2024, §4.1.2).
    # sqrt(tf) approximates the fraction of modes surviving the filter,
    # so effective mode count is (2l+1)*f_sky*sqrt(tf). Covariance scales
    # as 1/sqrt(N_i * N_j), giving a geometric mean correction:
    #     Cov /= (tf_i * tf_j)^{1/4}
    # PSpipe uses min(tf_i, tf_j) instead, but all tf are nearly equal
    # within the AA+AP block so it makes no difference there. The geometric
    # mean is more physical for cross-blocks where tf_A < 1 and tf_B = 1.
    for b in range(n_bins):
        tf_A = 1.0 / kspace_F_inv[:, b, 0, 0]  # [n_cross_A]
        if is_square:
            tf_B = tf_A
        elif kspace_F_inv_B is not None:
            tf_B = 1.0 / kspace_F_inv_B[:, b, 0, 0]  # [n_cross_B]
        else:
            tf_B = np.ones(n_cross_B)

        tf_geom = np.sqrt(np.outer(tf_A, tf_B))  # [n_cross_A, n_cross_B]
        mode_corr = np.sqrt(tf_geom)  # (tf_i * tf_j)^{1/4}
        cov_out[0][b] /= mode_corr
        cov_out[1][b] /= mode_corr
        cov_out[2][b] /= mode_corr

    return cov_out[0], cov_out[1], cov_out[2]


def compute_leakage_covariance(ell_values, bin_def, cross_idx_map,
                                detector_labels, det_to_band, lmax):
    """Compute additive T→P leakage covariance for EE, BB, and EB.

    Analytically equivalent to PSpipe's Monte Carlo approach (10,000 sims in
    get_leakage_sim.py / get_leakage_covariance.py), but computed directly
    from the error mode decomposition of γ — no MC noise.

    The leakage model (Eq. 2 of Louis et al. 2025, 2503.14452) gives:
        E_obs = E + γ_TE × T
        B_obs = B + γ_TB × T

    Propagating the uncertainty Δγ = E_mode @ n (n ~ N(0,I)) into each
    spectrum type, with C^{TB} = C^{EB} = 0 in ΛCDM:

    For cross-spectrum k = (det_i, det_j) with band(det_i) = a, band(det_j) = b:

    **EE**:
        δC^{EE}_{ij} = Δγ_TE^a [C_TE + γ_TE^b C_TT]
                     + Δγ_TE^b [C_TE + γ_TE^a C_TT]
        Error modes from band a (or b); weight uses the OTHER band's γ_TE.

    **BB**:
        δC^{BB}_{ij} = (Δγ_TB^a γ_TB^b + γ_TB^a Δγ_TB^b) C_TT
        Error modes from band a (or b); weight uses the OTHER band's γ_TB.

    **EB**:
        δC^{EB}_{ij} = Δγ_TB^b [C_TE + γ_TE^a C_TT]
                     + Δγ_TE^a [γ_TB^b C_TT]
        TB term (dominant): error modes from B-source band b,
            weight uses E-source band a's γ_TE.
        TE term (subdominant): error modes from E-source band a,
            weight uses B-source band b's γ_TB.

    Within a single band, all ACT splits share the same γ, so the
    leakage covariance is the same for all cross-pair combinations.
    NPIPE detectors have no leakage (γ = 0).

    Args:
        ell_values: 1D array of per-ell values (ℓ grid for spectra).
        bin_def: Bin edges array (n_bins+1) or 2D (n_bins, 2).
        cross_idx_map: [n_cross, 2] int array of (E-source, B-source) indices.
        detector_labels: Ordered detector labels.
        det_to_band: Dict mapping ACT detector label -> band name.
        lmax: Maximum multipole for loading γ files.

    Returns:
        (leak_cov_ee, leak_cov_bb, leak_cov_eb): each [n_bins, n_cross, n_cross].
    """
    bin_def = np.asarray(bin_def)
    if bin_def.ndim == 2:
        bin_lo, bin_hi = bin_def[:, 0], bin_def[:, 1]
    else:
        bin_lo, bin_hi = bin_def[:-1], bin_def[1:]
    n_bins = len(bin_lo)
    n_cross = cross_idx_map.shape[0]

    theory = camb_full_theory(ell_max=lmax)
    cl_te = theory["TE"]  # C_ℓ indexed by ell
    cl_tt = theory["TT"]

    # Load error modes and means per band
    band_data = {}
    for band in set(det_to_band.values()):
        gamma_TE, err_TE, gamma_TB, err_TB = _load_leakage_model(
            band, lmax)
        band_data[band] = {
            "gamma_TE": gamma_TE, "err_TE": err_TE,
            "gamma_TB": gamma_TB, "err_TB": err_TB,
        }

    # Determine ell range from data
    n_ell = len(ell_values)

    # Precompute per-band: Gram scalars (error modes dotted with themselves)
    # and weight scalars (binned C_TE + γ_TE·C_TT, binned γ_TB·C_TT).
    #
    # The derivative for a cross-spectrum factors as d(ℓ) = err_mode(ℓ) × w(ℓ),
    # where err_modes come from one band and w depends on the OTHER band's γ.
    # Since w is a scalar per ℓ, the dot product factors:
    #     d_vi @ d_hi = w_vi · w_hi · (err @ err)
    # We precompute G = Σ_k (binned err_k)² and scalar weights separately,
    # then combine with correct band lookups in the assembly loop.
    G_TE = {}   # G_TE[band][b] = scalar Gram value for TE error modes
    G_TB = {}   # G_TB[band][b] = scalar Gram value for TB error modes
    w_cte_gtt = {}  # w_cte_gtt[band][b] = binned (C_TE + γ_TE^band · C_TT)
    w_gtb_ctt = {}  # w_gtb_ctt[band][b] = binned (γ_TB^band · C_TT)

    def _bin_scalar(vals):
        """Bin a per-ell scalar array into [n_bins] by averaging."""
        binned = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (ell_values >= bin_lo[b]) & (ell_values < bin_hi[b])
            if np.any(mask):
                binned[b] = np.mean(vals[mask])
        return binned

    def _bin_modes(modes):
        """Bin per-ell error modes [n_ell, n_modes] → [n_bins, n_modes]."""
        n_modes = modes.shape[1]
        binned = np.zeros((n_bins, n_modes))
        for b in range(n_bins):
            mask = (ell_values >= bin_lo[b]) & (ell_values < bin_hi[b])
            if np.any(mask):
                binned[b] = np.mean(modes[mask], axis=0)
        return binned

    for band, bd in band_data.items():
        n_gamma = len(bd["gamma_TE"])

        # Per-ell error modes and weights, clipped to ell grid.
        # Beam-convolved: each factor gets this band's beam b_ℓ, so that
        # G_beamed[a] × w_beamed[b_vi] × w_beamed[b_hi] gives the correct
        # total beam b_a² × b_{b_vi} × b_{b_hi} for the four-map covariance.
        bl = load_act_beam(band, lmax=lmax)
        err_TE = np.zeros((n_ell, bd["err_TE"].shape[1]))
        err_TB = np.zeros((n_ell, bd["err_TB"].shape[1]))
        w_cte = np.zeros(n_ell)  # (C_TE + γ_TE · C_TT) × b_ℓ
        w_gtb = np.zeros(n_ell)  # (γ_TB · C_TT) × b_ℓ
        for ell_idx, ell in enumerate(ell_values):
            ell_int = int(ell)
            if ell_int >= n_gamma or ell_int >= len(cl_te):
                continue
            beam = bl[ell_int] if ell_int < len(bl) else 0.0
            err_TE[ell_idx] = bd["err_TE"][ell_int] * beam
            err_TB[ell_idx] = bd["err_TB"][ell_int] * beam
            w_cte[ell_idx] = (cl_te[ell_int] + bd["gamma_TE"][ell_int] * cl_tt[ell_int]) * beam
            w_gtb[ell_idx] = bd["gamma_TB"][ell_int] * cl_tt[ell_int] * beam

        # Bin error modes and compute Gram scalars
        binned_TE = _bin_modes(err_TE)
        binned_TB = _bin_modes(err_TB)
        G_TE[band] = np.array([binned_TE[b] @ binned_TE[b] for b in range(n_bins)])
        G_TB[band] = np.array([binned_TB[b] @ binned_TB[b] for b in range(n_bins)])

        # Bin scalar weights
        w_cte_gtt[band] = _bin_scalar(w_cte)
        w_gtb_ctt[band] = _bin_scalar(w_gtb)

    # Build the leakage covariance matrices [n_bins, n_cross, n_cross]
    leak_cov_ee = np.zeros((n_bins, n_cross, n_cross))
    leak_cov_bb = np.zeros((n_bins, n_cross, n_cross))
    leak_cov_eb = np.zeros((n_bins, n_cross, n_cross))

    # For cross-pair (i,j): i is E-source (col 0), j is B-source (col 1)
    i_idx = cross_idx_map[:, 0]
    j_idx = cross_idx_map[:, 1]

    # Band for each cross-pair's E-source and B-source detector
    band_E = []
    band_B = []
    for k in range(n_cross):
        band_E.append(det_to_band.get(detector_labels[i_idx[k]]))
        band_B.append(det_to_band.get(detector_labels[j_idx[k]]))

    # Assembly: Cov[b,vi,hi] = Σ_{shared bands} G[band][b] × w[other_vi][b] × w[other_hi][b]
    #
    # δC for one cross-pair has TWO error-mode contributions (α-source band
    # and β-source band). Squaring gives FOUR bilinear branches per pair (vi, hi)
    # for EE and BB:
    #   (α^vi, α^hi), (α^vi, β^hi), (β^vi, α^hi), (β^vi, β^hi)
    # Each branch contributes only when the two indexed bands match. The
    # error modes come from the shared band; each weight uses the OTHER
    # detector's band's γ.
    #
    # EB has only TWO branches because TE and TB error modes are independent
    # mode families (drawn from different files) — the α-side carries TE modes
    # and the β-side carries TB modes, so cross-type sharing contributes zero.
    for b in range(n_bins):
        # ── EE: δC^{EE}_{ij} = Δγ_TE^a [C_TE + γ_TE^b C_TT] + (a↔b) ──
        for vi in range(n_cross):
            if band_E[vi] is None or band_B[vi] is None:
                continue
            for hi in range(n_cross):
                if band_E[hi] is None or band_B[hi] is None:
                    continue
                # vi-α / hi-α: shared band α (= band_E)
                if band_E[hi] == band_E[vi]:
                    leak_cov_ee[b, vi, hi] += (G_TE[band_E[vi]][b]
                        * w_cte_gtt[band_B[vi]][b] * w_cte_gtt[band_B[hi]][b])
                # vi-α / hi-β: band_E[vi] == band_B[hi]
                if band_B[hi] == band_E[vi]:
                    leak_cov_ee[b, vi, hi] += (G_TE[band_E[vi]][b]
                        * w_cte_gtt[band_B[vi]][b] * w_cte_gtt[band_E[hi]][b])
                # vi-β / hi-α: band_B[vi] == band_E[hi]
                if band_E[hi] == band_B[vi]:
                    leak_cov_ee[b, vi, hi] += (G_TE[band_B[vi]][b]
                        * w_cte_gtt[band_E[vi]][b] * w_cte_gtt[band_B[hi]][b])
                # vi-β / hi-β: shared band β (= band_B)
                if band_B[hi] == band_B[vi]:
                    leak_cov_ee[b, vi, hi] += (G_TE[band_B[vi]][b]
                        * w_cte_gtt[band_E[vi]][b] * w_cte_gtt[band_E[hi]][b])

        # ── BB: δC^{BB}_{ij} = (Δγ_TB^a γ_TB^b + γ_TB^a Δγ_TB^b) C_TT ──
        for vi in range(n_cross):
            if band_E[vi] is None or band_B[vi] is None:
                continue
            for hi in range(n_cross):
                if band_E[hi] is None or band_B[hi] is None:
                    continue
                if band_E[hi] == band_E[vi]:
                    leak_cov_bb[b, vi, hi] += (G_TB[band_E[vi]][b]
                        * w_gtb_ctt[band_B[vi]][b] * w_gtb_ctt[band_B[hi]][b])
                if band_B[hi] == band_E[vi]:
                    leak_cov_bb[b, vi, hi] += (G_TB[band_E[vi]][b]
                        * w_gtb_ctt[band_B[vi]][b] * w_gtb_ctt[band_E[hi]][b])
                if band_E[hi] == band_B[vi]:
                    leak_cov_bb[b, vi, hi] += (G_TB[band_B[vi]][b]
                        * w_gtb_ctt[band_E[vi]][b] * w_gtb_ctt[band_B[hi]][b])
                if band_B[hi] == band_B[vi]:
                    leak_cov_bb[b, vi, hi] += (G_TB[band_B[vi]][b]
                        * w_gtb_ctt[band_E[vi]][b] * w_gtb_ctt[band_E[hi]][b])

        # ── EB: TB contribution (dominant) ──
        # δ from Δγ_TB^β: d = E^TB_β × [C_TE + γ_TE^α C_TT]
        # Shared B-source band; weight from E-source band.
        for vi in range(n_cross):
            if band_B[vi] is None or band_E[vi] is None:
                continue
            for hi in range(n_cross):
                if band_B[hi] != band_B[vi] or band_E[hi] is None:
                    continue
                leak_cov_eb[b, vi, hi] += (G_TB[band_B[vi]][b]
                    * w_cte_gtt[band_E[vi]][b] * w_cte_gtt[band_E[hi]][b])

        # ── EB: TE contribution (subdominant) ──
        # δ from Δγ_TE^α: d = E^TE_α × γ_TB^β C_TT
        # Shared E-source band; weight from B-source band.
        for vi in range(n_cross):
            if band_E[vi] is None or band_B[vi] is None:
                continue
            for hi in range(n_cross):
                if band_E[hi] != band_E[vi] or band_B[hi] is None:
                    continue
                leak_cov_eb[b, vi, hi] += (G_TE[band_E[vi]][b]
                    * w_gtb_ctt[band_B[vi]][b] * w_gtb_ctt[band_B[hi]][b])

    n_bands = len(band_data)
    print(f"  Leakage covariance: additive term from {n_bands} band(s), "
          f"{n_cross} cross-pairs")
    for label, cov in [("EE", leak_cov_ee), ("BB", leak_cov_bb),
                        ("EB", leak_cov_eb)]:
        diag = np.array([cov[b, 0, 0] for b in range(n_bins)])
        peak_b = np.argmax(diag)
        if diag[peak_b] > 0:
            print(f"    Peak leakage σ_{label} = {np.sqrt(diag[peak_b]):.3e} "
                  f"at bin {peak_b} "
                  f"(ℓ ~ {(bin_lo[peak_b] + bin_hi[peak_b])/2:.0f})")

    return leak_cov_ee, leak_cov_bb, leak_cov_eb
