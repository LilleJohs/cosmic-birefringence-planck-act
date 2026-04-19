"""Compute psi_ell from 353 GHz TE/TB cross-spectra.

psi_l = 0.5 * arctan(TB_smooth / TE_smooth).

Following Hervías-Caimapo et al. (2025, arXiv:2408.06214, Appendix B),
we use T_353A x B_353B for TB and T_353B x E_353A for TE, rather than
averaging both orderings. NPIPE 353 GHz TE/TB spectra have ordering-
dependent biases from intensity-to-polarisation leakage, and this
specific combination is the most robust against the systematic.

Steps:
1. Load unbinned 353Ax353B spectrum
2. Bin into target bins
3. Smooth binned spectra with Gaussian kernel (sigma=1.0 on binned data)
4. Compute psi from arctan(TB/TE) using recommended ordering
5. Save as 1D array of binned psi values + plot

Run for each NPIPE mask directory that has 353Ax353B unbinned spectra.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent.parent))
import likelihood_config as lcfg
from tools.binning import load_binning_file, bin_array


def _load_binning(binning, ell_max):
    """Load bin edges from binning scheme. Returns (bin_lows, bin_highs, bin_centers)."""
    if binning == "npipe":
        ell_min = lcfg.NPIPE_ELL_MIN
        ell_max_bin = lcfg.NPIPE_ELL_MAX
        delta_ell = lcfg.NPIPE_DELTA_ELL
        bin_edges = np.arange(ell_min, ell_max_bin + delta_ell, delta_ell)
        bin_lows = bin_edges[:-1]
        bin_highs = bin_edges[1:] - 1  # inclusive upper bounds
        bin_centers = (bin_lows + bin_highs) / 2.0
        return bin_lows, bin_highs, bin_centers
    elif binning in ("unified", "joint"):
        return load_binning_file(lcfg.get_unified_binning_file(), lmax=ell_max)
    else:
        raise ValueError(f"Unknown binning: {binning!r}. Use 'npipe' or 'joint'.")


def compute_psi_ell(input_dir, sigma=1.0, binning="npipe", output_dir=None):
    """Compute and save psi_ell from 353 GHz TE/TB spectra.

    Args:
        input_dir: Directory containing npipe_353Axnpipe_353B_obs_unbinned.dat.
        sigma: Gaussian smoothing width in bins (default 1.0, applied to binned data).
        binning: "npipe" for uniform Δℓ=20, "act" for amended ACT, "unified" for unified.
        output_dir: Directory for output files. Defaults to input_dir.
    """
    if output_dir is None:
        output_dir = input_dir
    spec_path = os.path.join(input_dir, "npipe_353Axnpipe_353B_obs_unbinned.dat")
    if not os.path.exists(spec_path):
        print(f"  WARNING: {spec_path} not found, skipping psi_ell computation")
        return None, None

    print(f"Computing psi_ell from {spec_path} (binning={binning})")

    data = np.loadtxt(spec_path, comments='#')
    # File: npipe_353Axnpipe_353B_obs_unbinned.dat
    # Columns: 0=ell, 1=TT, 2=TE, 3=ET, 4=EE, 5=EB, 6=BE, 7=BB, 8=TB, 9=BT
    #   col 2 (TE) = T_353A x E_353B
    #   col 3 (ET) = E_353A x T_353B  (= T_353B x E_353A)
    #   col 8 (TB) = T_353A x B_353B
    #   col 9 (BT) = B_353A x T_353B  (= T_353B x B_353A)
    ell = data[:, 0].astype(int)

    bin_lows, bin_highs, bin_centers = _load_binning(binning, ell.max())
    n_bins = len(bin_lows)

    # Use T_353A x B_353B for TB and T_353B x E_353A for TE.
    # Hervías-Caimapo et al. (2025, Appendix B) show that NPIPE 353 GHz
    # TE/TB spectra have ordering-dependent biases from intensity-to-
    # polarisation leakage. This specific combination is the most robust.
    TE_recommended = data[:, 3]  # T_353B x E_353A
    TB_recommended = data[:, 8]  # T_353A x B_353B

    # Bin and smooth
    TE_binned = bin_array(ell, TE_recommended, bin_lows, bin_highs)
    TB_binned = bin_array(ell, TB_recommended, bin_lows, bin_highs)

    TE_smooth = gaussian_filter1d(TE_binned, sigma=sigma)
    TB_smooth = gaussian_filter1d(TB_binned, sigma=sigma)

    # Compute psi using recommended ordering only
    psi_binned = 0.5 * np.arctan(TB_smooth / TE_smooth)

    # Save data
    if binning == "npipe":
        lmax = int(bin_centers[-1]) + 1
        psi_ell = np.zeros(lmax + 1)
        for b in range(n_bins):
            ell_center = int(bin_centers[b])
            if ell_center <= lmax:
                psi_ell[ell_center] = psi_binned[b]
        out_path = os.path.join(output_dir, "psi_ell_353.npy")
        np.save(out_path, psi_ell)
        print(f"  Saved psi_ell_353.npy: lmax={lmax}, shape={psi_ell.shape}, {n_bins} bins")
    else:
        tag = binning  # "act" or "unified"
        out_path = os.path.join(output_dir, f"psi_ell_353_{tag}_binning.npz")
        np.savez(out_path, psi_ell_binned=psi_binned, bin_centers=bin_centers)
        print(f"  Saved psi_ell_353_{tag}_binning.npz: {n_bins} bins, "
              f"ell=[{bin_centers[0]:.0f}, {bin_centers[-1]:.0f}]")

    print(f"  psi_ell range: [{np.rad2deg(psi_binned.min()):.2f}, {np.rad2deg(psi_binned.max()):.2f}] deg")

    # Save plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bin_centers, np.rad2deg(psi_binned), "o-", ms=3)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\psi_\ell$ [deg]")
    ax.set_title(f"353 GHz $\\psi_\\ell$ (binning={binning})")
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"psi_ell_353_{binning}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    return psi_binned, bin_centers


