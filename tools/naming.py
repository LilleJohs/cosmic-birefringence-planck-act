"""Shared naming helpers for likelihood .npz filenames and plot prefixes.

Used by beta_mcmc/config.py, beta_mcmc/run_beta_mcmc.py, and likelihood/run_likelihood.py
to ensure names are built identically on all sides.
"""


def build_bands_tag(act_bands, npipe_freqs):
    """Build a descriptive tag string from band/freq selection.

    E.g. "pa5f090_pa6f090_100_143". Used for npz filenames and plot filenames.
    """
    parts = []
    parts.extend(b.replace("_", "") for b in act_bands)
    parts.extend(str(f) for f in npipe_freqs)
    return "_".join(parts)


def _build_suffix(act_bands, npipe_freqs, use_sacc_aa=False,
                  npipe_only_binning=None, npipe_ell_min=None,
                  npipe_ell_max=None, npipe_delta_ell=None,
                  pp_mask_percent=0):
    """Build the shared suffix parts (SACC tag + binning tag + mask tag)."""
    sacc_tag = "_sacc" if use_sacc_aa else ""
    bin_tag = ""
    if not act_bands and npipe_only_binning == "uniform":
        bin_tag = f"_ell{npipe_ell_min}_{npipe_ell_max}_dl{npipe_delta_ell}"
    mask_tag = f"_{pp_mask_percent}_mask" if pp_mask_percent != 0 else ""
    return f"{sacc_tag}{bin_tag}{mask_tag}"


def build_npz_filename(act_bands, npipe_freqs, use_sacc_aa=False,
                       npipe_only_binning=None, npipe_ell_min=None,
                       npipe_ell_max=None, npipe_delta_ell=None,
                       pp_mask_percent=0):
    """Build descriptive npz filename from band/freq selection."""
    tag = build_bands_tag(act_bands, npipe_freqs)
    suffix = _build_suffix(act_bands, npipe_freqs, use_sacc_aa,
                           npipe_only_binning, npipe_ell_min,
                           npipe_ell_max, npipe_delta_ell,
                           pp_mask_percent=pp_mask_percent)
    return f"likelihood_data_{tag}{suffix}.npz"


def build_plot_prefix(mask, act_bands, npipe_freqs, use_sacc_aa=False,
                      ap_only=False, n_dust=0,
                      npipe_only_binning=None, npipe_ell_min=None,
                      npipe_ell_max=None, npipe_delta_ell=None,
                      pp_mask_percent=0):
    """Build a prefix for plot/chain filenames.

    E.g. "joint_100_143_217_353_ell51_1490_dl20_dust3"
    """
    tag = build_bands_tag(act_bands, npipe_freqs)
    ap_tag = "_ap_only" if ap_only else ""
    suffix = _build_suffix(act_bands, npipe_freqs, use_sacc_aa,
                           npipe_only_binning, npipe_ell_min,
                           npipe_ell_max, npipe_delta_ell,
                           pp_mask_percent=pp_mask_percent)
    prefix = f"{mask}{ap_tag}_{tag}{suffix}"
    if n_dust > 0:
        prefix += f"_dust{n_dust}"
    return prefix
