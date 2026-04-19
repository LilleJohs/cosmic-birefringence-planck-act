"""Entry point for the ACT/NPIPE power spectrum pipeline.

Computes EE, EB, BB power spectra from raw maps and writes unbinned
per-ell spectra (beam-convolved, MCM-deconvolved) to data/computed/.

Supports three modes:
  - ACT x ACT: cross-split spectra for all array-band pairs
  - ACT x NPIPE: ACT x NPIPE detector split cross-spectra
  - NPIPE x NPIPE: NPIPE A x B cross-spectra

Corrections, binning, validation plots, and .npz assembly are handled
by likelihood/run_likelihood.py (stage 2).
"""
import sys
import os
from multiprocessing import Pool

from pspy import sph_tools

import pipeline_config as cfg
from pspy import so_map
from tools.data_loading import act_map_path

from map_loader import (
    load_act_splits,
    load_act_split_kspace_filtered,
    project_npipe_to_car,
)
from mask_builder import (
    get_act_analysis_window, get_npipe_analysis_window, get_npipe_cross_window,
)
from compute_spectra import (
    compute_mcm,
    compute_npipe_spectrum,
    compute_act_x_npipe_spectrum,
    compute_act_cross_band_spectrum,
    compute_npipe_cross_on_act_footprint,
)
from output_formatter import (
    ensure_output_dirs,
    write_npipe_obs_and_theory,
    write_npipe_theory_file,
    write_unbinned_theory,
    write_unbinned_spectra,
)
from theory_spectra import (
    generate_npipe_theory_spectrum,
    generate_act_x_npipe_theory_spectrum,
    generate_act_x_act_theory_spectrum,
)


def _all_files_exist(out_dir, filenames):
    """Return True if all filenames exist in out_dir."""
    return all(os.path.exists(os.path.join(out_dir, f)) for f in filenames)


def _extract_spectra(binned_cl, spectrum_types=None):
    """Extract spectra of interest from pspy output dict.

    Args:
        binned_cl: Dict from pspy with keys like "TT", "EE", "EB", "BB", "ell", etc.
        spectrum_types: Which spectra to extract. Default: all from config.

    Returns:
        Dict with extracted spectra.
    """
    if spectrum_types is None:
        spectrum_types = cfg.SPECTRA_OF_INTEREST

    result = {}
    if "ell" in binned_cl:
        result["ell"] = binned_cl["ell"]
    for st in spectrum_types:
        if st in binned_cl:
            result[st] = binned_cl[st]
    return result


def run_act_x_act():
    """Compute all ACT x ACT per-split power spectra.

    For each band pair (including same-band), computes all 16 split pairs
    (4x4) and writes per-split unbinned Cl files to data/computed/act_mask/.
    """
    print("\n" + "=" * 60)
    print("ACT x ACT Power Spectra (per-split)")
    print("=" * 60)

    # Build analysis windows
    print("\nBuilding ACT analysis windows...")
    bands = cfg.ACT_ARRAY_BANDS
    act_windows = {}
    act_fskys = {}
    for band in bands:
        win, fsky = get_act_analysis_window(band)
        act_windows[band] = win
        act_fskys[band] = fsky
        print(f"  {band}: f_sky = {fsky:.4f}")

    lmax = cfg.ACT_X_ACT_COMPUTE_LMAX
    binning_file = cfg.get_unified_binning_file()
    print(f"  Binning file: {binning_file}")
    print(f"  Official PSpipe binning, lmax = {lmax}")

    # Precompute ACT alms
    print("\nPrecomputing ACT alms...")

    all_act_alms = {}
    for band in bands:
        if cfg.APPLY_KSPACE_FILTER:
            print(f"  Loading k-space filtered maps for {band}...")
            splits = [load_act_split_kspace_filtered(band, i) for i in range(cfg.ACT_N_SPLITS)]
        else:
            splits = load_act_splits(band)
        alms = []
        for i, split_map in enumerate(splits):
            alm = sph_tools.get_alms(split_map, act_windows[band],
                                      niter=cfg.ACT_X_NPIPE_NITER, lmax=lmax)
            alms.append(alm)
        all_act_alms[band] = alms
        del splits
    print(f"  Precomputed alms for {len(all_act_alms)} bands x {cfg.ACT_N_SPLITS} splits")

    # Compute all band pairs (including diagonal)
    all_results = {}
    band_pairs = []
    for i, band1 in enumerate(bands):
        for j, band2 in enumerate(bands):
            if i <= j:
                band_pairs.append((band1, band2))

    act_out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.ACT_MASK_SUBDIR)
    for band1, band2 in band_pairs:
        act_files = [f"act_{band1}_set{i}xact_{band2}_set{j}_{s}.dat"
                     for i in range(cfg.ACT_N_SPLITS) for j in range(cfg.ACT_N_SPLITS)
                     for s in ["obs_unbinned", "theory_unbinned"]]
        if _all_files_exist(act_out_dir, act_files):
            print(f"\n  Skipping {band1} x {band2} (output files already exist)")
            continue

        print(f"\n{'=' * 50}")
        print(f"Computing: {band1} x {band2} (all 16 split pairs)")
        print(f"{'=' * 50}")

        binned = compute_act_cross_band_spectrum(
            band1, band2,
            act_windows[band1], act_windows[band2],
            binning_file, lmax=lmax,
            alms1=all_act_alms[band1], alms2=all_act_alms[band2],
        )

        spectra = _extract_spectra(binned)
        all_results[f"{band1}x{band2}"] = spectra

        # Write per-split unbinned spectra
        act_label_fn = lambda i, j: f"act_{band1}_set{i}xact_{band2}_set{j}"
        print(f"  Writing per-split unbinned spectra...")
        write_unbinned_spectra(
            binned["cross_spectra"], act_label_fn,
            out_subdir=cfg.ACT_MASK_SUBDIR)

        # Generate theory (same coadd beam for all splits) and write unbinned for each pair
        print(f"  Generating beam-convolved theory spectra...")
        theory = generate_act_x_act_theory_spectrum(
            band1, band2, lmax=lmax, output_dl=True)
        for i in range(cfg.ACT_N_SPLITS):
            for j in range(cfg.ACT_N_SPLITS):
                label = f"act_{band1}_set{i}xact_{band2}_set{j}"
                write_unbinned_theory(theory, label,
                                      out_subdir=cfg.ACT_MASK_SUBDIR)

        print(f"  Wrote spectra to data/computed/{cfg.ACT_MASK_SUBDIR}/")

    print(f"\n{'=' * 60}")
    print(f"ACT x ACT complete. Computed {len(all_results)} band pairs.")
    print(f"{'=' * 60}")

    return all_results



# ============================================================
# NPIPE multiprocessing worker
# ============================================================

_npipe_worker_mask = None


def _init_npipe_worker(mask):
    """Pool initializer: store the mask once per worker process."""
    global _npipe_worker_mask
    _npipe_worker_mask = mask
    os.environ["OMP_NUM_THREADS"] = "1"


def _npipe_worker(args):
    """Compute one NPIPE spectrum and write obs + theory files."""
    freq1, split1, freq2, split2, out_subdir, lmax = args
    label = f"{freq1}{split1}x{freq2}{split2}"
    npipe_label = f"npipe_{freq1}{split1}xnpipe_{freq2}{split2}"
    npipe_files = [f"{npipe_label}_{s}.dat"
                   for s in ["obs_unbinned", "theory_unbinned"]]
    if _all_files_exist(os.path.join(cfg.OUTPUT_DIR, out_subdir), npipe_files):
        return label, None

    unbinned = compute_npipe_spectrum(freq1, split1, freq2, split2,
                                      _npipe_worker_mask, lmax=lmax)
    spectra = _extract_spectra(unbinned)
    theory = generate_npipe_theory_spectrum(
        freq1, split1, freq2, split2, lmax=lmax, output_dl=False)
    write_npipe_obs_and_theory(
        obs_dict=spectra, theory_dict=theory,
        freq1=freq1, split1=split1, freq2=freq2, split2=split2,
        out_subdir=out_subdir)
    return label, spectra


def run_npipe(npipe_mask, out_subdir, fsky):
    """Compute NPIPE x NPIPE power spectra for all frequency and split combinations.

    Computes all frequency pairs (100x100, 100x143, ...) with non-redundant
    split combinations (AA, AB, BB).

    Args:
        npipe_mask: HEALPix mask array (numpy, shape = 12*nside^2).
        out_subdir: Output subdirectory under OUTPUT_DIR.
        fsky: Sky fraction for this mask.
    """
    print("\n" + "=" * 60)
    print("NPIPE x NPIPE Power Spectra")
    print("=" * 60)

    print(f"  f_sky = {fsky:.4f}")
    print(f"  nside = {cfg.NPIPE_NSIDE}  (per-frequency)")
    print(f"  Output: {out_subdir}")

    results = {}
    lmax = cfg.NPIPE_COMPUTE_LMAX
    test_freqs = cfg.NPIPE_FREQS

    # Build job list: (freq1, split1, freq2, split2, out_subdir, lmax)
    jobs = []
    # Same-frequency pairs (AA, AB, BB)
    for freq in test_freqs:
        for i_split, split1 in enumerate(cfg.NPIPE_SPLITS):
            for split2 in cfg.NPIPE_SPLITS[i_split:]:
                jobs.append((freq, split1, freq, split2, out_subdir, lmax))
    # Cross-frequency pairs (all split combos)
    for i, freq1 in enumerate(test_freqs):
        for freq2 in test_freqs[i + 1:]:
            for split1 in cfg.NPIPE_SPLITS:
                for split2 in cfg.NPIPE_SPLITS:
                    jobs.append((freq1, split1, freq2, split2, out_subdir, lmax))

    nworkers = min(len(jobs), cfg.NPIPE_NWORKERS)
    print(f"  {len(jobs)} spectra, {nworkers} parallel workers")

    if nworkers <= 1:
        # Sequential fallback
        _init_npipe_worker(npipe_mask)
        for job in jobs:
            label, spectra = _npipe_worker(job)
            if spectra is not None:
                results[label] = spectra
                print(f"  Done: {label}")
            else:
                print(f"  Skipped: {label}")
    else:
        with Pool(nworkers, initializer=_init_npipe_worker,
                  initargs=(npipe_mask,)) as pool:
            for label, spectra in pool.imap_unordered(_npipe_worker, jobs):
                if spectra is not None:
                    results[label] = spectra
                    print(f"  Done: {label}")
                else:
                    print(f"  Skipped: {label}")

    print(f"\n{'=' * 60}")
    print(f"NPIPE computation complete. {len(results)} spectra computed.")
    print(f"{'=' * 60}")

    return results



def _all_spectrum_files_exist(out_dir, label):
    """Check if all unbinned obs/theory files exist for a given label."""
    suffixes = ["obs_unbinned.dat", "theory_unbinned.dat"]
    return all(os.path.exists(os.path.join(out_dir, f"{label}_{s}"))
               for s in suffixes)


def run_act_x_npipe():
    """Compute ACT x NPIPE cross-spectra.

    Projects NPIPE HEALPix maps to ACT CAR grid (with caching),
    then computes cross-spectra for all configured (ACT array-band, NPIPE freq)
    combinations using 4 ACT x 2 NPIPE = 8 split pairs per combination.

    Returns:
        Dict of computed spectra keyed by "{act_band}x{npipe_freq}".
    """
    print("\n" + "=" * 60)
    print("ACT x NPIPE Cross-Spectra")
    print("=" * 60)

    lmax = cfg.ACT_X_NPIPE_COMPUTE_LMAX
    act_bands = cfg.ACT_ARRAY_BANDS
    npipe_freqs = cfg.NPIPE_FREQS

    print(f"\n  ACT array-bands: {act_bands}")
    print(f"  NPIPE frequencies: {npipe_freqs}")
    print(f"  lmax = {lmax}")
    print(f"  niter = {cfg.ACT_X_NPIPE_NITER}")

    # Step 1: Load ACT template for projection
    print("\n--- Step 1: Setup ---")
    act_template = so_map.read_map(act_map_path(act_bands[0], 0))
    print(f"  ACT template geometry: {act_template.data.shape}")

    # Step 2: Set up windows
    print("\n--- Step 2: Setting up analysis windows ---")

    # ACT window for each array-band
    act_windows = {}
    act_fskys = {}
    for band in act_bands:
        win, fsky = get_act_analysis_window(band)
        act_windows[band] = win
        act_fskys[band] = fsky
        print(f"  ACT {band}: f_sky = {fsky:.4f}")

    # NPIPE windows (reuse ACT windows matched by frequency)
    npipe_windows = {}
    npipe_fskys = {}
    for freq in npipe_freqs:
        win, fsky = get_npipe_cross_window(freq)
        npipe_windows[freq] = win
        npipe_fskys[freq] = fsky
        print(f"  NPIPE {freq}: f_sky = {fsky:.4f} (using ACT {cfg.NPIPE_ACT_WINDOW} window)")

    # Step 2b: Get official PSpipe binning file (pspy MCM internal requirement)
    binning_file = cfg.get_unified_binning_file()
    print(f"\n  Binning file: {binning_file}")

    # Step 3: Precompute ACT alms (shared across cross-spectra and cross-band steps)
    print("\n--- Step 3: Precomputing ACT alms ---")

    all_act_alms = {}  # {band: [alm_split0, alm_split1, ...]}
    for band in act_bands:
        if cfg.APPLY_KSPACE_FILTER:
            print(f"  Loading k-space filtered maps for {band}...")
            splits = [load_act_split_kspace_filtered(band, i) for i in range(cfg.ACT_N_SPLITS)]
        else:
            print(f"  Loading and computing windowed alms for {band}...")
            splits = load_act_splits(band)
        alms = []
        for i, split_map in enumerate(splits):
            # PSpipe approach: apply window to map BEFORE computing alms
            alm = sph_tools.get_alms(split_map, act_windows[band], niter=cfg.ACT_X_NPIPE_NITER, lmax=lmax)
            alms.append(alm)
        all_act_alms[band] = alms
        del splits  # Free map memory
    print(f"  Precomputed windowed alms for {len(all_act_alms)} bands x {cfg.ACT_N_SPLITS} splits")

    # Step 3b: Pre-project all NPIPE maps and compute windowed alms (reused across all ACT bands)
    print("\n--- Step 3b: Precomputing NPIPE alms (projected to ACT CAR grid) ---")
    all_npipe_alms = {}  # {freq: {"A": alm, "B": alm}}
    for npipe_freq in npipe_freqs:
        print(f"  Projecting and computing windowed alms for NPIPE {npipe_freq}...")
        freq_alms = {}
        for split in cfg.NPIPE_SPLITS:
            npipe_map = project_npipe_to_car(npipe_freq, split, act_template)
            alm = sph_tools.get_alms(npipe_map, npipe_windows[npipe_freq],
                                      niter=cfg.ACT_X_NPIPE_NITER, lmax=lmax)
            freq_alms[split] = alm
            del npipe_map
        all_npipe_alms[npipe_freq] = freq_alms
    print(f"  Precomputed NPIPE alms for {len(all_npipe_alms)} frequencies x {len(cfg.NPIPE_SPLITS)} splits")

    # Step 4: NPIPE x NPIPE cross-spectra on ACT footprint
    # Must run before ACT x NPIPE so that NPIPE auto-spectra exist
    # when likelihood stage needs them.
    print("\n--- Step 4: NPIPE x NPIPE cross-spectra on ACT footprint ---")
    # All freq pairs and split combos (including same-freq auto-splits for variance)
    npipe_freq_pairs = []
    for i, f1 in enumerate(npipe_freqs):
        for f2 in npipe_freqs[i:]:
            npipe_freq_pairs.append((f1, f2))

    # Shared ACT window (same for all NPIPE freqs) — compute MCM once
    npipe_x_npipe_mbb_inv = None
    if npipe_freqs:
        npipe_act_win = npipe_windows[npipe_freqs[0]]
        print(f"  Computing shared MCM for NPIPE x NPIPE on ACT footprint...")
        npipe_x_npipe_mbb_inv, _ = compute_mcm(
            npipe_act_win, npipe_act_win, binning_file, lmax=lmax, niter=cfg.ACT_X_NPIPE_NITER)

    for freq1, freq2 in npipe_freq_pairs:
        for split1 in cfg.NPIPE_SPLITS:
            for split2 in cfg.NPIPE_SPLITS:
                if freq1 == freq2 and split1 > split2:
                    continue  # Skip redundant BA when freq1 == freq2

                act_mask_dir = os.path.join(cfg.OUTPUT_DIR, cfg.ACT_MASK_SUBDIR)
                npipe_act_label = f"npipe_{freq1}{split1}xnpipe_{freq2}{split2}"

                if _all_spectrum_files_exist(act_mask_dir, npipe_act_label):
                    print(f"\n  Skipping {npipe_act_label} (already exists)")
                    continue

                print(f"\n  Computing NPIPE {freq1}{split1} x {freq2}{split2} on ACT footprint...")

                # Get pre-computed alms if available
                alm1 = all_npipe_alms.get(freq1, {}).get(split1)
                alm2 = all_npipe_alms.get(freq2, {}).get(split2)

                unbinned = compute_npipe_cross_on_act_footprint(
                    freq1, split1, freq2, split2,
                    npipe_act_win, binning_file, lmax=lmax,
                    alm1=alm1, alm2=alm2,
                    mbb_inv=npipe_x_npipe_mbb_inv,
                )

                write_unbinned_spectra(
                    [(freq1, split1, freq2, split2, unbinned)],
                    lambda f1, s1, f2, s2: f"npipe_{f1}{s1}xnpipe_{f2}{s2}",
                    out_subdir=cfg.ACT_MASK_SUBDIR)

                print(f"  Generating theory spectrum...")
                theory = generate_npipe_theory_spectrum(
                    freq1, split1, freq2, split2,
                    lmax=lmax, output_dl=False,
                )
                write_npipe_theory_file(
                    theory, freq1, split1, freq2, split2,
                    out_subdir=cfg.ACT_MASK_SUBDIR,
                )

    # Step 5: ACT x NPIPE cross-spectra and ACT cross-band spectra
    print("\n--- Step 5: ACT x NPIPE cross-spectra ---")
    results = {}

    for act_band in act_bands:
        # --- ACT x NPIPE cross-spectra for this band ---
        for npipe_freq in npipe_freqs:
            label = f"{act_band}x{npipe_freq}"

            # Check if output already exists
            axn_files = (
                [f"act_{act_band}_set{i}xnpipe_{npipe_freq}{s}_obs_unbinned.dat"
                 for i in range(cfg.ACT_N_SPLITS) for s in cfg.NPIPE_SPLITS]
                + [f"act_{act_band}_set{i}xnpipe_{npipe_freq}{s}_theory_unbinned.dat"
                   for i in range(cfg.ACT_N_SPLITS) for s in cfg.NPIPE_SPLITS]
            )
            if _all_files_exist(os.path.join(cfg.OUTPUT_DIR, cfg.ACT_MASK_SUBDIR), axn_files):
                print(f"\n  Skipping {label} (output files already exist)")
                continue

            lmin = 2

            print(f"\n{'=' * 50}")
            print(f"Computing: {label}")
            print(f"  lmin = {lmin}, lmax = {lmax}")
            print(f"{'=' * 50}")

            binned = compute_act_x_npipe_spectrum(
                act_band, npipe_freq, npipe_maps={},
                act_window=act_windows[act_band], npipe_window=npipe_windows[npipe_freq],
                binning_file=binning_file, lmax=lmax,
                act_alms=all_act_alms[act_band],
                npipe_alms=all_npipe_alms[npipe_freq],
            )

            results[label] = _extract_spectra(binned)

            # Write individual split-level unbinned cross-spectra
            axn_label_fn = lambda i_act, split: f"act_{act_band}_set{i_act}xnpipe_{npipe_freq}{split}"
            print(f"  Writing unbinned output...")
            write_unbinned_spectra(
                binned["cross_spectra"], axn_label_fn,
                out_subdir=cfg.ACT_MASK_SUBDIR)

            # Generate theory (coadd beam) and write unbinned for each split pair
            print(f"  Generating beam-convolved theory spectra...")
            for npipe_split in cfg.NPIPE_SPLITS:
                theory = generate_act_x_npipe_theory_spectrum(
                    act_band, npipe_freq, npipe_split,
                    lmax=lmax, output_dl=True,
                )
                for i_act in range(cfg.ACT_N_SPLITS):
                    thlabel = f"act_{act_band}_set{i_act}xnpipe_{npipe_freq}{npipe_split}"
                    write_unbinned_theory(theory, thlabel,
                                          out_subdir=cfg.ACT_MASK_SUBDIR)

            print(f"  Wrote spectra to data/computed/{cfg.ACT_MASK_SUBDIR}/")

    print(f"\n{'=' * 60}")
    print(f"ACT x NPIPE computation complete. Computed {len(results)} spectra.")
    print(f"{'=' * 60}")

    return results


def run():
    """Run the full pipeline."""
    # Set up logging to file
    log_file = os.path.join(os.path.dirname(__file__), 'pipeline_run.log')

    # Open log file and redirect stdout/stderr
    log_f = open(log_file, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(original_stdout, log_f)
    sys.stderr = Tee(original_stderr, log_f)

    print("=" * 60, flush=True)
    print("Power Spectrum Pipeline", flush=True)
    print("=" * 60, flush=True)
    print(f"Logging to: {log_file}", flush=True)
    print(f"Output directory: {cfg.OUTPUT_DIR}", flush=True)
    print(f"Compute ACT mask: {cfg.COMPUTE_ACT_MASK}", flush=True)
    print(f"Compute NPIPE mask: {cfg.COMPUTE_NPIPE_FULL_SKY_MASK}", flush=True)
    print(f"K-space filter: {cfg.APPLY_KSPACE_FILTER}", flush=True)
    if cfg.COMPUTE_ACT_MASK:
        print(f"ACT bands: {cfg.ACT_ARRAY_BANDS}", flush=True)
        if not cfg.COMPUTE_ACT_X_ACT_ONLY:
            print(f"NPIPE freqs (ACT mask): {cfg.NPIPE_FREQS}", flush=True)
    if cfg.COMPUTE_NPIPE_FULL_SKY_MASK:
        print(f"NPIPE freqs: {cfg.NPIPE_FREQS}", flush=True)
    sys.stdout.flush()

    ensure_output_dirs()

    if cfg.COMPUTE_ACT_MASK:
        if cfg.COMPUTE_ACT_X_ACT_ONLY:
            print("\n>>> Starting ACT x ACT only...", flush=True)
        else:
            print("\n>>> Starting ACT mask computation (ACT x ACT + ACT x NPIPE + NPIPE x NPIPE on ACT footprint)...", flush=True)
        sys.stdout.flush()
        act_results = run_act_x_act()
        print(f">>> ACT x ACT complete. Computed {len(act_results)} spectra.", flush=True)
        sys.stdout.flush()

        if not cfg.COMPUTE_ACT_X_ACT_ONLY:
            act_x_npipe_results = run_act_x_npipe()
            print(f">>> ACT x NPIPE complete. Computed {len(act_x_npipe_results)} spectra.", flush=True)
            sys.stdout.flush()

    if cfg.COMPUTE_NPIPE_FULL_SKY_MASK:
        print("\n>>> Starting NPIPE mask computation (NPIPE x NPIPE, fsky~0.9)...", flush=True)
        sys.stdout.flush()
        npipe_window, fsky_npipe = get_npipe_analysis_window(nside=2048)
        npipe_results = run_npipe(
            npipe_mask=npipe_window[0].data,
            out_subdir=cfg.NPIPE_MASK_SUBDIR,
            fsky=fsky_npipe,
        )
        print(f">>> NPIPE x NPIPE complete. Computed {len(npipe_results)} spectra.", flush=True)
        sys.stdout.flush()

    print("\n" + "=" * 60, flush=True)
    print("Pipeline complete!", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    # Restore stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_f.close()


if __name__ == "__main__":
    run()
