"""Generate beam-convolved theoretical ΛCDM power spectra.

For numerical stability at high multipoles, we convolve theory spectra with
instrumental beams and pixel windows rather than deconvolving observed spectra.
"""
import numpy as np
import sys
from pathlib import Path

import healpy as hp

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.spectra import camb_full_theory, cell_to_dell
from tools.data_loading import load_act_beam, load_npipe_beam

import pipeline_config as cfg


# ACT DR6 CAR map pixel size in arcminutes (confirmed 0.5 arcmin for all arrays)
ACT_PIXEL_SIZE_ARCMIN = 0.5


def compute_car_pixel_window(pixel_size_arcmin, lmax):
    """Effective 1D pixel window for CAR pixelization.

    Computes the azimuthal average of the 2D separable sinc pixel window:
        W_eff(ell) = <sinc(ell*delta*cos(theta)/(2*pi)) * sinc(ell*delta*sin(theta)/(2*pi))>_theta

    where delta is the pixel size in radians and sinc is the normalized sinc
    (np.sinc(x) = sin(pi*x)/(pi*x)), matching pixell's calc_window convention.

    Args:
        pixel_size_arcmin: Pixel size in arcminutes.
        lmax: Maximum multipole.

    Returns:
        1D numpy array of length lmax+1, from ell=0 to lmax.
    """
    delta = pixel_size_arcmin * np.pi / (180.0 * 60.0)  # radians
    ells = np.arange(lmax + 1)
    n_theta = 400
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    f_scale = delta / (2.0 * np.pi)
    # shape: (lmax+1, n_theta)
    f = ells[:, np.newaxis] * f_scale
    cos_t = np.cos(thetas)[np.newaxis, :]
    sin_t = np.sin(thetas)[np.newaxis, :]
    pw = np.mean(np.sinc(f * cos_t) * np.sinc(f * sin_t), axis=1)
    return pw


def compute_healpix_pixel_window(nside, lmax):
    """Compute HEALPix pixel window function.

    Uses healpy.sphtfunc.pixwin for proper HEALPix pixel window calculation.

    Args:
        nside: HEALPix resolution parameter (512, 1024, or 2048)
        lmax: Maximum multipole

    Returns:
        Tuple (pw_T, pw_pol): temperature and polarization pixel windows,
        each a 1D numpy array from ell=0 to lmax.
    """
    pw_full = hp.sphtfunc.pixwin(nside, pol=True, lmax=lmax+1)
    pw_T = pw_full[0][:lmax + 1]      # Temperature window
    pw_pol = pw_full[1][:lmax + 1]    # Polarization window (E and B)

    return pw_T, pw_pol


def _convolve(cl, bl1, bl2, pw1, pw2):
    """Convolve a spectrum with beam and pixel window transfer functions.

    Implements: C_ℓ^obs = C_ℓ × b_ℓ^1 × b_ℓ^2 × w_ℓ^1 × w_ℓ^2

    All arrays are padded with ones to match length if needed.
    """
    n = len(cl)
    product = np.ones(n)
    for tf in (bl1, bl2, pw1, pw2):
        ext = np.ones(n)
        ext[:min(len(tf), n)] = tf[:min(len(tf), n)]
        product *= ext
    return cl * product


def _build_theory_result(ells, cl_dict, beams1, beams2, pw1, pw2, output_dl):
    """Build beam-convolved theory result dict.

    Args:
        ells: Multipole array.
        cl_dict: Dict with "TT", "EE", "BB", "TE" from CAMB.
        beams1: (bl_T, bl_E, bl_B) for field 1.
        beams2: (bl_T, bl_E, bl_B) for field 2.
        pw1: (pw_T, pw_pol) for field 1.
        pw2: (pw_T, pw_pol) for field 2.
        output_dl: If True, convert to D_ell.

    Returns:
        Dict with "ell", "TT", "EE", "BB", "TE", "ET", and parity-odd zeros.
    """
    cl_tt, cl_ee, cl_bb, cl_te = cl_dict["TT"], cl_dict["EE"], cl_dict["BB"], cl_dict["TE"]

    bl_T1, bl_E1, bl_B1 = beams1
    bl_T2, bl_E2, bl_B2 = beams2
    pw_T1, pw_pol1 = pw1
    pw_T2, pw_pol2 = pw2

    result = {
        "ell": ells,
        "TT": _convolve(cl_tt, bl_T1, bl_T2, pw_T1, pw_T2),
        "EE": _convolve(cl_ee, bl_E1, bl_E2, pw_pol1, pw_pol2),
        "BB": _convolve(cl_bb, bl_B1, bl_B2, pw_pol1, pw_pol2),
        "TE": _convolve(cl_te, bl_T1, bl_E2, pw_T1, pw_pol2),
        "ET": _convolve(cl_te, bl_E1, bl_T2, pw_pol1, pw_T2),
        "EB": np.zeros_like(ells, dtype=float),
        "BE": np.zeros_like(ells, dtype=float),
        "TB": np.zeros_like(ells, dtype=float),
        "BT": np.zeros_like(ells, dtype=float),
    }

    if output_dl:
        for key in ["TT", "EE", "BB", "TE", "ET", "EB", "BE", "TB", "BT"]:
            result[key] = cell_to_dell(ells, result[key])

    return result


def _load_field_beams_and_pw(type, lmax, **kwargs):
    """Load beam transfer functions and pixel window for one field.

    Args:
        type: "act" or "npipe".
        lmax: Maximum multipole.
        **kwargs: For "act": band.
                  For "npipe": freq, split.

    Returns:
        (beams, pw) where beams = (bl_T, bl_E, bl_B) and pw = (pw_T, pw_pol).
    """
    if type == "act":
        bl = load_act_beam(kwargs['band'], lmax=lmax)
        pw = compute_car_pixel_window(ACT_PIXEL_SIZE_ARCMIN, lmax)
        return (bl, bl, bl), (pw, pw)

    if type == "npipe":
        freq, split = kwargs['freq'], kwargs['split']
        beams = load_npipe_beam(freq, split, freq, split, lmax=lmax)
        nside = cfg.NPIPE_NSIDE[freq]
        pw = compute_healpix_pixel_window(nside, lmax)
        return beams, pw

    raise ValueError(f"Unknown field type: {type}")


def _field_label(field):
    """Human-readable label for a field descriptor."""
    if field["type"] == "act":
        return f"ACT {field['band']}"
    if field["type"] == "npipe":
        return f"NPIPE {field['freq']}{field['split']}"
    raise ValueError(f"Unknown field type: {field['type']}")


def generate_theory_spectrum(field1, field2, lmax=None, output_dl=True):
    """Generate beam-convolved LCDM theory spectrum for a cross-spectrum.

    Args:
        field1, field2: Field descriptors, each a dict:
            ACT:  {"type": "act", "band": "pa5_f090"}
            NPIPE: {"type": "npipe", "freq": 100, "split": "A"}
        lmax: Maximum multipole (default: ACT_X_ACT_COMPUTE_LMAX).
        output_dl: If True, return Dl. If False, return Cl.

    Returns:
        Dict with "ell", "TT", "EE", "BB", "TE", "ET", and parity-odd zeros.
    """
    if lmax is None:
        lmax = cfg.ACT_X_ACT_COMPUTE_LMAX

    label1 = _field_label(field1)
    label2 = _field_label(field2)
    print(f"  Generating theory for {label1} x {label2} [beam-convolved]...")

    theory_cl = camb_full_theory(ell_max=lmax)
    ells = theory_cl["ell"]

    # Always beam-convolve theory
    beams1, pw1 = _load_field_beams_and_pw(lmax=lmax, **field1)
    beams2, pw2 = _load_field_beams_and_pw(lmax=lmax, **field2)

    result = _build_theory_result(ells, theory_cl, beams1, beams2, pw1, pw2, output_dl)
    print(f"  Theory spectrum generated for {label1} x {label2}")
    return result


def generate_npipe_theory_spectrum(freq1, split1, freq2, split2,
                                   lmax=None, output_dl=True):
    """Generate beam-convolved ΛCDM spectrum for an NPIPE cross-spectrum."""
    return generate_theory_spectrum(
        {"type": "npipe", "freq": freq1, "split": split1},
        {"type": "npipe", "freq": freq2, "split": split2},
        lmax=lmax, output_dl=output_dl)


def generate_act_x_npipe_theory_spectrum(act_array_band, npipe_freq, npipe_split,
                                          lmax=None, output_dl=True):
    """Generate LCDM theory spectrum for an ACT x NPIPE cross-spectrum."""
    return generate_theory_spectrum(
        {"type": "act", "band": act_array_band},
        {"type": "npipe", "freq": npipe_freq, "split": npipe_split},
        lmax=lmax, output_dl=output_dl)


def generate_act_x_act_theory_spectrum(band1, band2, lmax=None, output_dl=True):
    """Generate LCDM theory spectrum for an ACT x ACT cross-spectrum."""
    return generate_theory_spectrum(
        {"type": "act", "band": band1},
        {"type": "act", "band": band2},
        lmax=lmax, output_dl=output_dl)
