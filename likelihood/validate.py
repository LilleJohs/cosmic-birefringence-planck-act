"""Null tests and validation plots from likelihood .npz data.

Reads binned spectra and covariance from the .npz file produced by
run_likelihood.py. Error bars come directly from the covariance
diagonal — no recomputation from auto-spectra.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.spectra import cell_to_dell


def _plot_spectrum_vs_theory(ell, dl_obs, dl_theory, yerr, spec_name,
                             plot_dir, title):
    """Two-panel plot: obs vs theory, fractional residual below."""
    obs_label = 'Observed (beam-convolved)'
    theory_label = r'$\Lambda$CDM (beam-convolved)'

    # Fractional residual
    valid = np.abs(dl_theory) > 1e-10
    frac_res = np.full_like(dl_obs, np.nan)
    frac_res[valid] = (dl_obs[valid] - dl_theory[valid]) / dl_theory[valid]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})

    ax1.plot(ell, dl_theory, 'k-', lw=1, alpha=0.5, label=theory_label)
    ax1.errorbar(ell, dl_obs, yerr=yerr, fmt='o', ms=4, color='C0',
                 label=obs_label, capsize=2)
    ax1.set_ylabel(rf'$D_\ell^{{{spec_name}}}\ [\mu K^2]$')
    ax1.set_title(f'{title}: {spec_name} (beam-convolved)')
    ax1.legend(fontsize=9)

    ax2.axhline(0, color='k', ls='--', lw=0.8)
    ax2.axhspan(-0.1, 0.1, alpha=0.1, color='gray')
    ax2.plot(ell, frac_res, 'o', ms=4, color='C0')
    ax2.set_ylabel(r'$(D_\ell^{\rm obs} - D_\ell^{\rm th}) / D_\ell^{\rm th}$')
    ax2.set_xlabel(r'$\ell$')
    ax2.set_ylim(-1, 1)

    fig.savefig(os.path.join(plot_dir, f'{spec_name}_vs_theory.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_null(ell, dl_obs, yerr, dl_theory_source, spec_name,
               plot_dir, title):
    """Plot a parity-odd null spectrum with error bars.

    Overlays expected birefringence signal 2*beta*Dl_source for beta = 0.3 deg.

    Returns:
        SNR = weighted_mean / sigma(weighted_mean).
    """
    beta_deg = 0.3
    beta_rad = np.radians(beta_deg)

    # SNR using analytic errors
    valid = yerr > 0
    if np.any(valid):
        weights = np.zeros_like(yerr)
        weights[valid] = 1.0 / yerr[valid]**2
        weighted_mean = np.sum(dl_obs[valid] * weights[valid]) / np.sum(weights[valid])
        sigma_mean = 1.0 / np.sqrt(np.sum(weights[valid]))
        snr = weighted_mean / sigma_mean if sigma_mean > 0 else 0
    else:
        snr = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.errorbar(ell, dl_obs, yerr=yerr, fmt='o', ms=4, color='C0', capsize=2,
                label=f'{spec_name} (SNR={snr:.2f})')

    # Overlay expected birefringence signal
    if dl_theory_source is not None:
        if spec_name in ("EB", "BE"):
            source_label = "EE"
        elif spec_name == "TB":
            source_label = "TE"
        else:
            source_label = None
        if source_label is not None:
            signal = 2.0 * beta_rad * dl_theory_source
            ax.plot(ell, signal, '-', color='C3', lw=2, alpha=0.8,
                    label=rf'$2\beta \, D_\ell^{{{source_label}}}$ ($\beta={beta_deg}^\circ$)')

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(rf'$D_\ell^{{{spec_name}}}\ [\mu K^2]$')
    ax.set_title(f'{title}: {spec_name} null test')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{spec_name}_null.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig)

    return snr


def run_null_tests(npz_path):
    """Run null tests for all cross-spectra in a likelihood .npz file.

    Extracts binned spectra from C_obs_3d and error bars from the covariance
    diagonal. Plots EE/BB vs theory and EB/BE null tests for each cross-spectrum.

    Args:
        npz_path: Path to the likelihood .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)

    C_obs = data['C_obs_3d']       # [n_bins, n_cross, 3] (EE=0, BB=1, EB=2)
    C_theory = data['C_theory_3d'] # [n_bins, n_cross, 2] (EE=0, BB=1)
    bin_centers = data['bin_centers']
    n_bins = int(data['n_bins'])
    n_cross = int(data['n_cross'])
    cov_ee = data['cov_ee']
    cov_bb = data['cov_bb']
    cov_eb = data['cov_eb']
    if cov_eb.shape[0] != n_bins:
        raise ValueError(
            f"Covariance has {cov_eb.shape[0]} bins but data has {n_bins}. "
            "Regenerate the .npz with run_likelihood.py."
        )

    if 'active_bins' in data:
        active_bins = data['active_bins']  # [n_cross, n_bins]
    else:
        active_bins = np.ones((n_cross, n_bins), dtype=bool)

    # Build cross-spectrum labels
    det1 = data['cross_det1']
    det2 = data['cross_det2']
    labels = [f"{d1} x {d2}" for d1, d2 in zip(det1, det2)]

    plots_base = os.path.join(os.path.dirname(npz_path), "null_tests")
    os.makedirs(plots_base, exist_ok=True)

    print("=" * 60)
    print(f"Null Tests from {os.path.basename(npz_path)}")
    print(f"  {n_cross} cross-spectra, {n_bins} bins")
    print("=" * 60)

    all_snr = {}

    for vi in range(n_cross):
        label = labels[vi]
        mask = active_bins[vi].astype(bool)
        if not np.any(mask):
            continue

        ell = bin_centers[mask]

        # Extract Cl spectra on active bins
        cl_ee = C_obs[mask, vi, 0]
        cl_bb = C_obs[mask, vi, 1]
        cl_eb = C_obs[mask, vi, 2]
        cl_ee_th = C_theory[mask, vi, 0]
        cl_bb_th = C_theory[mask, vi, 1]

        # Convert Cl → Dl
        dl_ee = cell_to_dell(ell, cl_ee)
        dl_bb = cell_to_dell(ell, cl_bb)
        dl_eb = cell_to_dell(ell, cl_eb)
        dl_ee_th = cell_to_dell(ell, cl_ee_th)
        dl_bb_th = cell_to_dell(ell, cl_bb_th)

        # Error bars from covariance diagonal (Cl) → Dl
        sigma_ee = cell_to_dell(ell, np.sqrt(np.abs(cov_ee[mask, vi, vi])))
        sigma_bb = cell_to_dell(ell, np.sqrt(np.abs(cov_bb[mask, vi, vi])))
        sigma_eb = cell_to_dell(ell, np.sqrt(np.abs(cov_eb[mask, vi, vi])))

        # Plot directory for this cross-spectrum
        plot_dir = os.path.join(plots_base, label.replace(" ", ""))
        os.makedirs(plot_dir, exist_ok=True)

        # EE and BB vs theory
        _plot_spectrum_vs_theory(ell, dl_ee, dl_ee_th, sigma_ee, "EE",
                                plot_dir, label)
        _plot_spectrum_vs_theory(ell, dl_bb, dl_bb_th, sigma_bb, "BB",
                                plot_dir, label)

        # EB null test (signal source = EE theory)
        snr_eb = _plot_null(ell, dl_eb, sigma_eb, dl_ee_th, "EB",
                            plot_dir, label)

        all_snr[label] = snr_eb

        print(f"  {label}: EB SNR = {snr_eb:.2f}"
              f"  [{'PASS' if abs(snr_eb) < 3 else 'FAIL'}]")

    print(f"\n{'=' * 60}")
    print(f"Plots: {plots_base}")
    print(f"{'=' * 60}")

    return all_snr
