"""Main script for running birefringence MCMC analysis."""
import os
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

import multiprocessing
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2 as chi2_dist

from config import (
    MASK,
    SAMPLING_MODE,
    ACT_ALPHA_PRIOR_STD,
    NPIPE_HFI_ALPHA_PRIOR_STD, NPIPE_LFI_ALPHA_PRIOR_STD,
    NPIPE_FREQS, JOINT_COMPUTED_DIR, NPIPE_MASK_PERCENT,
    ACT_BANDS, USE_SACC_AA, AP_ONLY,
    NPIPE_ONLY_BINNING, NPIPE_ELL_MIN, NPIPE_ELL_MAX, NPIPE_DELTA_ELL,
    N_MCMC_WORKERS, MCMC_N_STEPS, MCMC_N_BURN,
    USE_DUST_EB_MODEL, N_DUST_BINS, ACT_DR6_ELL_MAX, ACT_DR6_OFFDIAG_COV,
)
from precomputed_loader import load_precomputed
from birefringence_likelihood import _is_hfi_label


def _optimal_n_walkers(n_params):
    """Set n_walkers = 2 * n_workers so each emcee half-step fills all workers.

    emcee's StretchMove splits walkers into two halves and evaluates each half
    in one pool.map call. With n_walkers = 2 * n_workers, each half-step
    evaluates exactly n_workers posteriors, fully utilizing all cores.
    """
    n = max(2 * N_MCMC_WORKERS, 2 * n_params)
    return n + (n % 2)  # ensure even
from tools.naming import build_plot_prefix, build_bands_tag, build_npz_filename
from mk_analysis import log_posterior_mk, get_alpha_labels, _precompute_alpha_maps
from rotation import compute_f_ell
from act_dr6 import load_act_sacc_data
from birefringence_likelihood import mk_chi2, precompute_bin_groups


# ============================================================
# Shared MCMC helpers
# ============================================================

# Global state for multiprocessing workers (avoids pickling large arrays)
_worker_fn = None
_worker_args = None


def _worker_init(fn, args):
    """Initialize worker process with shared log-posterior function and args."""
    global _worker_fn, _worker_args
    _worker_fn = fn
    _worker_args = args


def _worker_call(theta):
    """Evaluate log-posterior in worker process."""
    return _worker_fn(theta, *_worker_args)




def _to_display_units(arr, n_non_angle=0):
    """Convert rad->deg for angle parameters. Works for 2D samples or 3D chains.

    The last axis is parameters. The first (arr.shape[-1] - n_non_angle)
    parameters are angles; the rest are left unchanged.
    """
    out = arr.copy()
    n_angle = out.shape[-1] - n_non_angle
    out[..., :n_angle] = np.rad2deg(out[..., :n_angle])
    return out


def _run_mcmc(log_posterior_fn, log_posterior_args, n_params, pos,
              n_walkers, n_steps, n_burn, n_non_angle=0):
    """Run emcee MCMC and return (sampler, samples_rad, samples_display)."""
    n_workers = N_MCMC_WORKERS
    pool = None
    try:
        if n_workers > 1:
            pool = multiprocessing.Pool(
                n_workers,
                initializer=_worker_init,
                initargs=(log_posterior_fn, log_posterior_args),
            )
            sampler = emcee.EnsembleSampler(
                n_walkers, n_params, _worker_call, pool=pool
            )
            print(f"MCMC: {n_walkers} walkers, {n_workers} parallel workers")
        else:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_params, log_posterior_fn, args=log_posterior_args
            )

        for _ in tqdm(sampler.sample(pos, iterations=n_steps), total=n_steps, desc="MCMC"):
            pass
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nDiscarding {n_burn} burn-in steps...")
    samples = sampler.get_chain(discard=n_burn, flat=True)
    samples_display = _to_display_units(samples, n_non_angle=n_non_angle)
    return sampler, samples, samples_display


def _save_trace_plot(sampler, param_names, n_burn, plots_dir, filename,
                     n_non_angle=0):
    """Save trace plot for all parameters."""
    n_params = len(param_names)
    chain = sampler.get_chain()
    chain_deg = _to_display_units(chain, n_non_angle=n_non_angle)

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    for i in range(n_params):
        axes[i].plot(chain_deg[:, :, i], alpha=0.3, lw=0.5)
        axes[i].set_ylabel(param_names[i])
        axes[i].axvline(n_burn, color='r', ls='--', lw=1,
                        label='burn-in' if i == 0 else None)
    axes[-1].set_xlabel("Step")
    if n_params > 0:
        axes[0].legend(loc='upper right')
    fig.tight_layout()
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_corner_plot(samples_deg, param_names, plots_dir, filename,
                      suptitle=None):
    """Save corner plot."""
    fig = corner.corner(
        samples_deg,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.4f',
        title_kwargs={"fontsize": 12}
    )
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _compute_chi2(samples_deg, data_dict, sample_alphas, n_dust=0,
                  sample_beta=True):
    """Compute chi2, dof, PTE from median MCMC samples. Returns (chi2, ndof, pte, title_str)."""
    alpha_labels = get_alpha_labels(data_dict)
    n_alpha = len(alpha_labels)
    best_theta = np.deg2rad(np.percentile(samples_deg, 50, axis=0))
    if 'active_bins' in data_dict and 'bin_groups' not in data_dict:
        precompute_bin_groups(data_dict)
    if 'alpha_maps' not in data_dict:
        _precompute_alpha_maps(data_dict)
    maps = data_dict['alpha_maps']

    alpha_offset = 1 if sample_beta else 0
    beta = best_theta[0] if sample_beta else 0.0

    alpha_by_label = np.zeros(maps['n_alpha'])
    if sample_alphas:
        alpha_by_label[:] = best_theta[alpha_offset:alpha_offset + n_alpha]
    alpha_i = alpha_by_label[maps['alpha_idx_i']]
    alpha_j = alpha_by_label[maps['alpha_idx_j']]

    f_ell = None
    if n_dust > 0:
        A_l = best_theta[-n_dust:]
        n_cross = len(data_dict['cross_spec_list'])
        f_ell = compute_f_ell(A_l, data_dict['dust'], n_cross)

    chi2_total, n_data = mk_chi2(alpha_i, alpha_j, beta, data_dict,
                                 f_ell=f_ell)
    n_fit_params = (1 if sample_beta else 0) + (n_alpha if sample_alphas else 0) + n_dust
    ndof = n_data - n_fit_params
    pte = 1.0 - chi2_dist.cdf(chi2_total, ndof)

    print(f"\nChi2 = {chi2_total:.2f}, dof = {ndof}, "
          f"Chi2/dof = {chi2_total/ndof:.2f}, PTE = {pte:.3f}")

    title = (rf"$\chi^2 = {chi2_total:.1f}$, dof $= {ndof}$, "
             rf"$\chi^2/\mathrm{{dof}} = {chi2_total/ndof:.2f}$, "
             rf"PTE $= {pte:.3f}$")
    return chi2_total, ndof, pte, title


def _print_results(samples_display, param_names, n_non_angle=0):
    """Print median and 68% CI for each parameter."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    n_angle = len(param_names) - n_non_angle
    for i, label in enumerate(param_names):
        mcmc = np.percentile(samples_display[:, i], [16, 50, 84])
        if i < n_angle:
            print(f"{label}: {mcmc[1]:.3f} +{mcmc[2]-mcmc[1]:.3f} -{mcmc[1]-mcmc[0]:.3f} deg")
        else:
            print(f"{label}: {mcmc[1]:.4f} +{mcmc[2]-mcmc[1]:.4f} -{mcmc[1]-mcmc[0]:.4f}")


def _run_and_plot_mcmc(log_posterior_fn, log_posterior_args, n_params,
                       param_names, plots_dir, trace_fname, corner_fname,
                       pos=None, n_non_angle=0, chain_file=None):
    """Run MCMC and save trace and corner plots. Returns samples_display."""
    n_walkers = _optimal_n_walkers(n_params)
    print(f"\nSampling {n_params} parameters:")
    for i, name in enumerate(param_names):
        print(f"  {i}: {name}")
    if pos is None:
        pos = np.zeros(n_params) + 1e-2 * np.random.randn(n_walkers, n_params)
    sampler, _, samples_display = _run_mcmc(
        log_posterior_fn, log_posterior_args,
        n_params, pos, n_walkers, MCMC_N_STEPS, MCMC_N_BURN,
        n_non_angle=n_non_angle)
    _print_results(samples_display, param_names, n_non_angle=n_non_angle)
    _save_trace_plot(sampler, param_names, MCMC_N_BURN, plots_dir, trace_fname,
                     n_non_angle=n_non_angle)
    if corner_fname is not None:
        _save_corner_plot(samples_display, param_names, plots_dir, corner_fname)
    if chain_file is not None and MCMC_N_STEPS > 100:
        chain = sampler.get_chain()
        np.save(chain_file, chain)
        print(f"  Saved chain: {chain_file}")
    return samples_display


# ============================================================
# Grid search (used when SAMPLING_MODE="beta_only")
# ============================================================

def _grid_search_beta(log_posterior_fn, log_posterior_args):
    """Run 1D grid search over beta, return (beta_median_deg, beta_p16_deg, beta_p84_deg, posterior, beta_grid_deg)."""
    beta_grid_deg = np.linspace(0, 0.4, 2000)
    beta_grid_rad = np.deg2rad(beta_grid_deg)
    log_post = np.zeros(len(beta_grid_rad))

    print(f"Evaluating posterior on {len(beta_grid_rad)} grid points...")
    for i, beta_val in tqdm(enumerate(beta_grid_rad)):
        params = np.array([beta_val])
        log_post[i] = log_posterior_fn(params, *log_posterior_args)

    log_post[~np.isfinite(log_post)] = -1e30
    log_post -= np.max(log_post)
    posterior = np.exp(log_post)
    posterior /= np.trapz(posterior, beta_grid_deg)

    cdf = np.cumsum(posterior) * (beta_grid_deg[1] - beta_grid_deg[0])
    cdf /= cdf[-1]

    beta_median_deg = np.interp(0.5, cdf, beta_grid_deg)
    beta_p16_deg = np.interp(0.16, cdf, beta_grid_deg)
    beta_p84_deg = np.interp(0.84, cdf, beta_grid_deg)

    return beta_median_deg, beta_p16_deg, beta_p84_deg, posterior, beta_grid_deg


def _save_beta_posterior_plot(beta_grid_deg, posterior, beta_median_deg,
                              beta_p16_deg, beta_p84_deg, title, plots_dir, filename):
    """Save 1D beta posterior plot from grid search."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(beta_grid_deg, posterior, 'k-', lw=1.5)
    ax.axvline(beta_median_deg, color='C0', ls='--', lw=1, label='Median')
    ax.axvspan(beta_p16_deg, beta_p84_deg, alpha=0.2, color='C0', label='68% CI')
    ax.axvline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$\beta$ [degrees]')
    ax.set_ylabel(r'$P(\beta | \mathrm{data})$')
    beta_err_up = beta_p84_deg - beta_median_deg
    beta_err_lo = beta_median_deg - beta_p16_deg
    ax.set_title(f"{title}\n"
                 rf"$\beta = {beta_median_deg:.3f}"
                 rf"^{{+{beta_err_up:.3f}}}"
                 rf"_{{-{beta_err_lo:.3f}}}$°")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")



# ============================================================
# MK unified param info and walker init
# ============================================================

def _build_mk_param_info(data_dict, n_dust=0):
    """Build parameter names and count for any MK analysis.

    Returns:
        (sample_alphas, sample_beta, n_params, param_names)
    """
    sample_beta = SAMPLING_MODE != "alpha_only"
    sample_alphas = SAMPLING_MODE != "beta_only"
    alpha_labels = get_alpha_labels(data_dict)

    param_names = []
    n_params = 0

    if sample_beta:
        param_names.append(r'$\beta$')
        n_params += 1

    if sample_alphas:
        for label in alpha_labels:
            param_names.append(rf'$\alpha_{{{label}}}$')
        n_params += len(alpha_labels)

    for i in range(n_dust):
        param_names.append(rf'$A_{{{i}}}$')
    n_params += n_dust

    return sample_alphas, sample_beta, n_params, param_names


def _init_dust_walkers(pos, n_params, n_dust, rng):
    """Initialize A_l walker positions near converged values."""
    a_l_init = [0.02, 0.03, 0.03, 0.06]
    a_l_std = [0.015, 0.02, 0.02, 0.025]
    n_walkers = pos.shape[0]
    for k in range(n_dust):
        center = a_l_init[k] if k < len(a_l_init) else 0.03
        std = a_l_std[k] if k < len(a_l_std) else 0.02
        pos[:, n_params - n_dust + k] = np.abs(rng.normal(center, std, size=n_walkers))


def _init_mk_walkers(n_params, data_dict, sample_alphas, n_dust=0,
                     sample_beta=True):
    """Initialize walker positions for any MK analysis."""
    n_walkers = _optimal_n_walkers(n_params)
    rng = np.random.default_rng(42)
    pos = np.zeros((n_walkers, n_params))
    alpha_offset = 0

    if sample_beta:
        pos[:, 0] = rng.normal(0.0, 0.01, size=n_walkers)
        alpha_offset = 1

    if sample_alphas:
        alpha_labels = get_alpha_labels(data_dict)
        for k, label in enumerate(alpha_labels):
            if label in ACT_ALPHA_PRIOR_STD:
                std_deg = ACT_ALPHA_PRIOR_STD[label]
            elif _is_hfi_label(label):
                std_deg = NPIPE_HFI_ALPHA_PRIOR_STD
            else:
                std_deg = NPIPE_LFI_ALPHA_PRIOR_STD
            std_rad = np.deg2rad(std_deg)
            pos[:, alpha_offset + k] = rng.normal(0.0, std_rad * 0.5, size=n_walkers)

    if n_dust > 0:
        _init_dust_walkers(pos, n_params, n_dust, rng)

    return pos


def _get_n_dust(mask, data_dict):
    """Return number of dust parameters for a mask type."""
    if mask == "joint":
        if USE_DUST_EB_MODEL and 'dust' in data_dict:
            return N_DUST_BINS
    return 0


# ============================================================
# MK data loading
# ============================================================

def _load_mk_data():
    """Load precomputed MK likelihood data based on MASK config."""
    npipe_only = not ACT_BANDS
    fname = build_npz_filename(
        ACT_BANDS, NPIPE_FREQS,
        use_sacc_aa=USE_SACC_AA and not npipe_only,
        npipe_only_binning=NPIPE_ONLY_BINNING if npipe_only else None,
        npipe_ell_min=NPIPE_ELL_MIN if npipe_only else None,
        npipe_ell_max=NPIPE_ELL_MAX if npipe_only else None,
        npipe_delta_ell=NPIPE_DELTA_ELL if npipe_only else None,
        pp_mask_percent=NPIPE_MASK_PERCENT)
    npz_path = os.path.join(JOINT_COMPUTED_DIR, fname)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Precomputed likelihood data not found: {npz_path}\n"
            f"Run likelihood/run_likelihood.py first."
        )
    return load_precomputed(npz_path)


# ============================================================
# Unified MK analysis runner
# ============================================================

_MK_BANNERS = {
    "joint": [
        "Unified-Covariance Joint ACT+Planck Analysis",
        "Minami & Komatsu (2020)",
    ],
}

_MK_GRID_TITLES = {
    "joint": "Unified: Beta posterior",
}


def _run_mk_analysis():
    """Run MK birefringence analysis based on MASK config."""
    # Banner
    print("=" * 70)
    for line in _MK_BANNERS[MASK]:
        print(line)
    print("=" * 70)

    # Load data
    print(f"\nLoading {MASK} data...")
    data_dict = _load_mk_data()

    # Dust and param info
    n_dust = _get_n_dust(MASK, data_dict)
    if n_dust > 0:
        print(f"  Dust EB model: {n_dust} A_l parameters")
    sample_alphas, sample_beta, n_params, param_names = _build_mk_param_info(
        data_dict, n_dust)
    print(f"  Sampling mode: {SAMPLING_MODE}")

    plots_dir = os.path.join(_repo_root, "plots", MASK)
    os.makedirs(plots_dir, exist_ok=True)

    # Log-posterior function and args
    log_post_fn = log_posterior_mk
    log_post_args = (data_dict, sample_alphas, n_dust, sample_beta)

    # Filenames
    npipe_only = not ACT_BANDS
    prefix = build_plot_prefix(
        MASK, ACT_BANDS, NPIPE_FREQS,
        use_sacc_aa=USE_SACC_AA, ap_only=AP_ONLY, n_dust=n_dust,
        npipe_only_binning=NPIPE_ONLY_BINNING if npipe_only else None,
        npipe_ell_min=NPIPE_ELL_MIN if npipe_only else None,
        npipe_ell_max=NPIPE_ELL_MAX if npipe_only else None,
        npipe_delta_ell=NPIPE_DELTA_ELL if npipe_only else None,
        pp_mask_percent=NPIPE_MASK_PERCENT)
    trace_fname = f"{prefix}_trace.png"
    corner_fname = f"{prefix}_corner.png"
    grid_fname = f"{prefix}_beta_posterior.png"

    use_mcmc = sample_alphas or n_dust > 0
    if use_mcmc:
        # Chain warm-start for slow runs (dust EB, unified)
        chain_file = None
        if n_dust > 0 or MASK == "joint":
            chain_file = os.path.join(plots_dir, f"{prefix}_chain.npy")

        pos = _try_warmstart(chain_file, n_params, n_dust)
        if pos is None:
            pos = _init_mk_walkers(n_params, data_dict, sample_alphas, n_dust,
                                   sample_beta)

        samples_deg = _run_and_plot_mcmc(
            log_post_fn, log_post_args, n_params, param_names,
            plots_dir, trace_fname, None,
            pos=pos, n_non_angle=n_dust, chain_file=chain_file)

        _, _, _, chi2_title = _compute_chi2(samples_deg, data_dict,
                                            sample_alphas, n_dust,
                                            sample_beta)
        _save_corner_plot(samples_deg, param_names, plots_dir, corner_fname,
                          suptitle=chi2_title)
    else:
        print(f"\nGrid search: beta only (all alpha=0)")
        beta_med, beta_p16, beta_p84, posterior, beta_grid = _grid_search_beta(
            log_post_fn, log_post_args)

        print("\n" + "=" * 70)
        print("RESULTS (grid search)")
        print("=" * 70)
        print(f"beta: {beta_med:.4f} +{beta_p84-beta_med:.4f} -{beta_med-beta_p16:.4f} deg")

        _save_beta_posterior_plot(
            beta_grid, posterior, beta_med, beta_p16, beta_p84,
            _MK_GRID_TITLES[MASK], plots_dir, grid_fname)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def _try_warmstart(chain_file, n_params, n_dust):
    """Try to warm-start walkers from a previous chain file. Returns pos or None."""
    if chain_file is None:
        return None
    if not os.path.exists(chain_file):
        print(f"  No previous chain found at: {chain_file}")
        return None

    n_walkers = _optimal_n_walkers(n_params)
    rng = np.random.default_rng(42)
    prev_chain = np.load(chain_file)
    if prev_chain.shape[2] != n_params:
        print(f"  Chain file n_params mismatch ({prev_chain.shape[2]} vs {n_params}), using default init")
        return None

    old_walkers = prev_chain.shape[1]
    last_pos = prev_chain[-1]  # (old_walkers, n_params)
    if old_walkers < n_walkers:
        # Tile old positions to fill new walker count
        reps = (n_walkers // old_walkers) + 1
        last_pos = np.tile(last_pos, (reps, 1))[:n_walkers]
    else:
        last_pos = last_pos[:n_walkers]
    pos = last_pos + 1e-5 * rng.standard_normal((n_walkers, n_params))
    if n_dust > 0:
        pos[:, -n_dust:] = np.abs(pos[:, -n_dust:])
    print(f"  Initialized walkers from previous chain: {chain_file}")
    return pos


# ============================================================
# ACT DR6 analysis
# ============================================================

def _run_act_dr6_analysis():
    """Run ACT DR6 birefringence analysis using SACC file with full covariance."""
    print("=" * 70)
    print("ACT DR6 Cosmic Birefringence Analysis")
    print("Full cross-pair covariance from official SACC file")
    print(f"ell_max = {ACT_DR6_ELL_MAX}, off-diagonal covariance: {ACT_DR6_OFFDIAG_COV}")
    print("Using MK (2020) likelihood machinery")
    print("=" * 70)

    data_dict = load_act_sacc_data(ACT_BANDS)

    sample_alphas, sample_beta, n_params, param_names = _build_mk_param_info(
        data_dict)

    plots_dir = os.path.join(_repo_root, "plots", MASK)
    os.makedirs(plots_dir, exist_ok=True)

    log_post_fn = log_posterior_mk
    log_post_args = (data_dict, sample_alphas, 0, sample_beta)

    bands_tag = build_bands_tag(ACT_BANDS, [])
    lmax_tag = f"_lmax{ACT_DR6_ELL_MAX}" if ACT_DR6_ELL_MAX is not None else ""
    cov_tag = "_fullcov" if ACT_DR6_OFFDIAG_COV else "_diagcov"
    prefix = f"act_dr6_{bands_tag}{lmax_tag}{cov_tag}"

    pos = _init_mk_walkers(n_params, data_dict, sample_alphas,
                           sample_beta=sample_beta)

    samples_deg = _run_and_plot_mcmc(
        log_post_fn, log_post_args, n_params, param_names,
        plots_dir, f"{prefix}_trace.png", None,
        pos=pos)

    # Chi2 and corner plot
    _, _, _, chi2_title = _compute_chi2(samples_deg, data_dict, sample_alphas,
                                        sample_beta=sample_beta)
    _save_corner_plot(samples_deg, param_names, plots_dir,
                      f"{prefix}_corner.png", suptitle=chi2_title)


# ============================================================
# Main entry point
# ============================================================

_MK_MASKS = {"joint"}


def run():
    """Main analysis function."""
    print("Starting birefringence MCMC analysis...")
    print(f"Mask: {MASK}")

    plots_dir = os.path.join(_repo_root, "plots", MASK)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Output plots will be saved to: {plots_dir}")

    if MASK == "act_dr6":
        _run_act_dr6_analysis()
    elif MASK in _MK_MASKS:
        _run_mk_analysis()
    else:
        raise RuntimeError(f"Unknown MASK: {MASK}")


if __name__ == "__main__":
    run()
