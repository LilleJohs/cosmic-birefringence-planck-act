"""Format and write output files for spectrum pipeline.

Output formats:
- _obs_unbinned.dat: Per-ell C_l (for covariance computation)
- _theory_unbinned.dat: Per-ell theory C_l
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline_config as cfg
from tools.spectra import dell_to_cell

_SPEC_ORDER = ["TT", "TE", "ET", "EE", "EB", "BE", "BB", "TB", "BT"]


def ensure_output_dirs():
    """Create output directory structure."""
    for subdir in [cfg.ACT_MASK_SUBDIR, cfg.NPIPE_MASK_SUBDIR]:
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, subdir), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "auxiliary"), exist_ok=True)


def _write_spectrum_file(fpath, ell, spec_dict, header, spec_order, ell_fmt="{:.1f}"):
    """Write a spectrum file with standard column format.

    Args:
        fpath: Output file path.
        ell: Multipole array.
        spec_dict: Dict mapping spectrum names to arrays.
        header: Header string (written as-is).
        spec_order: List of spectrum keys to write as columns.
        ell_fmt: Format string for ell column (e.g. "{:.1f}" or "{:d}").
    """
    with open(fpath, "w") as f:
        f.write(header)
        for i in range(len(ell)):
            line = ell_fmt.format(int(ell[i]) if "d" in ell_fmt else ell[i])
            for spec in spec_order:
                val = spec_dict.get(spec, np.zeros_like(ell))[i]
                line += f" {val:.12e}"
            f.write(line + "\n")
    print(f"  Wrote {fpath}")


def _write_dl_as_cl_file(fpath, ell, dl_dict, header, ell_fmt="{:.1f}"):
    """Convert Dl spec_dict to Cl and write spectrum file."""
    cl_dict = {}
    for spec in _SPEC_ORDER:
        dl = dl_dict.get(spec, np.zeros_like(ell))
        cl_dict[spec] = dell_to_cell(ell, dl)
    _write_spectrum_file(fpath, ell, cl_dict, header, _SPEC_ORDER, ell_fmt=ell_fmt)


def _get_out_dir(out_subdir):
    """Get output directory, creating if needed."""
    ensure_output_dirs()
    if out_subdir is None:
        out_subdir = cfg.ACT_MASK_SUBDIR
    return os.path.join(cfg.OUTPUT_DIR, out_subdir)


# ============================================================
# Unbinned per-ell writers
# ============================================================

def write_unbinned_spectra(unbinned_list, label_fn, out_subdir=None):
    """Write per-ell (delta_l=1) spectra in Cl format.

    Args:
        unbinned_list: List of tuples. Each tuple has arbitrary leading
            indices followed by a dict with "ell" and spectrum keys (in Dl).
            E.g. [(i, j, dict), ...] or [(i, split, dict), ...].
        label_fn: Callable that takes the leading indices and returns a
            filename label string (without _obs_unbinned.dat suffix).
        out_subdir: Output subdirectory under OUTPUT_DIR.
    """
    out_dir = _get_out_dir(out_subdir)

    for entry in unbinned_list:
        spec_dict = entry[-1]
        indices = entry[:-1]
        label = label_fn(*indices)
        ell = spec_dict["ell"]

        fpath = os.path.join(out_dir, f"{label}_obs_unbinned.dat")
        header = f"### {label} Per-ell C_ell (beam-convolved, uK^2) ###\n"
        header += "# ell  " + "  ".join(_SPEC_ORDER) + "\n"

        _write_dl_as_cl_file(fpath, ell, spec_dict, header, ell_fmt="{:d}")

    print(f"  Wrote {len(unbinned_list)} unbinned spectra to {out_dir}")


def write_unbinned_theory(theory_dict, label, out_subdir=None):
    """Write per-ell theory (Dl) as Cl without binning.

    Args:
        theory_dict: Dict with "ell", "EE", "BB", etc. (unbinned Dl).
        label: File label (output: {label}_theory_unbinned.dat).
        out_subdir: Output subdirectory.

    Returns:
        Path to written file.
    """
    out_dir = _get_out_dir(out_subdir)
    fpath = os.path.join(out_dir, f"{label}_theory_unbinned.dat")
    header = f"### {label} Per-ell Theory C_ell (LCDM, beam-convolved, uK^2) ###\n"
    header += "# ell  " + "  ".join(_SPEC_ORDER) + "\n"

    _write_dl_as_cl_file(fpath, theory_dict["ell"], theory_dict, header, ell_fmt="{:d}")
    return fpath


def write_npipe_obs_and_theory(obs_dict, theory_dict, freq1, split1, freq2, split2,
                               out_subdir=None):
    """Write observed and theory spectra for NPIPE (unbinned only).

    Writes:
        - _obs_unbinned.dat: per-ell C_l (for covariance)
        - _theory_unbinned.dat: per-ell theory C_l

    Args:
        obs_dict: Dict with "ell", "TT", "EE", "EB", "BB", etc. (unbinned C_l in uK^2)
        theory_dict: Dict with "ell", "TT", "EE", "EB", "BB", etc. (unbinned C_l in uK^2)
        freq1, split1, freq2, split2: Frequency and split labels
        out_subdir: Output subdirectory under OUTPUT_DIR.

    Returns:
        Dict with paths to written files
    """
    out_dir = os.path.join(cfg.OUTPUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    label = f"npipe_{freq1}{split1}xnpipe_{freq2}{split2}"
    paths = {}

    # Unbinned per-ell observed file
    obs_fpath = os.path.join(out_dir, f"{label}_obs_unbinned.dat")
    header_unbinned = f"### {label} Per-ell C_ell (beam-convolved, uK^2) ###\n"
    header_unbinned += "# ell  " + "  ".join(_SPEC_ORDER) + "\n"
    _write_spectrum_file(obs_fpath, obs_dict["ell"], obs_dict, header_unbinned, _SPEC_ORDER, ell_fmt="{:d}")
    paths["obs_unbinned"] = obs_fpath

    # Unbinned per-ell theory file
    theory_unbinned_fpath = os.path.join(out_dir, f"{label}_theory_unbinned.dat")
    header_theory_unbinned = f"### {label} Per-ell Theory C_ell (LCDM, beam-convolved, uK^2) ###\n"
    header_theory_unbinned += "# ell  " + "  ".join(_SPEC_ORDER) + "\n"
    _write_spectrum_file(theory_unbinned_fpath, theory_dict["ell"], theory_dict, header_theory_unbinned, _SPEC_ORDER, ell_fmt="{:d}")
    paths["theory_unbinned"] = theory_unbinned_fpath

    return paths


def write_npipe_theory_file(theory_dict, freq1, split1, freq2, split2,
                            out_subdir=None):
    """Write NPIPE theory C_ell file (unbinned only).

    Args:
        theory_dict: Dict with "ell", "TT", "EE", "EB", "BB", etc. (unbinned C_l in uK^2)
        freq1, split1, freq2, split2: Frequency and split labels
        out_subdir: Output subdirectory under OUTPUT_DIR (default: NPIPE_MASK_SUBDIR)

    Returns:
        Path to the unbinned file.
    """
    ensure_output_dirs()
    if out_subdir is None:
        out_subdir = cfg.NPIPE_MASK_SUBDIR
    out_dir = os.path.join(cfg.OUTPUT_DIR, out_subdir)
    label = f"npipe_{freq1}{split1}xnpipe_{freq2}{split2}"

    # Write unbinned
    unbinned_fpath = os.path.join(out_dir, f"{label}_theory_unbinned.dat")
    header_unbinned = f"### {label} Per-ell Theory C_ell (LCDM, beam-convolved, uK^2) ###\n"
    header_unbinned += "# ell  " + "  ".join(_SPEC_ORDER) + "\n"
    _write_spectrum_file(unbinned_fpath, theory_dict["ell"], theory_dict, header_unbinned, _SPEC_ORDER, ell_fmt="{:d}")

    return unbinned_fpath
