"""CMB power spectrum utilities: unit conversions and CAMB theory spectra."""
import numpy as np
import camb


def dell_to_cell(ells: np.ndarray, dell: np.ndarray) -> np.ndarray:
    """Convert D_ell to C_ell: C_ell = D_ell * 2pi / (ell(ell+1)).

    Handles ell=0 safely by setting C_0 = 0.
    """
    ells = np.asarray(ells, dtype=float)
    dell = np.asarray(dell)
    cl = np.zeros_like(dell, dtype=float)
    nonzero = ells > 0
    cl[nonzero] = dell[nonzero] * 2 * np.pi / (ells[nonzero] * (ells[nonzero] + 1))
    return cl


def cell_to_dell(ells: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert C_ell to D_ell: D_ell = C_ell * ell(ell+1) / (2pi).

    Handles ell=0 safely (returns 0).
    """
    ells = np.asarray(ells, dtype=float)
    cell = np.asarray(cell)
    dl = np.zeros_like(cell, dtype=float)
    nonzero = ells > 0
    dl[nonzero] = cell[nonzero] * ells[nonzero] * (ells[nonzero] + 1) / (2 * np.pi)
    return dl


def camb_full_theory(ell_max: int = 1000) -> dict:
    """Compute all lensed CMB power spectra using CAMB.

    Returns:
        Dict with keys "ell", "TT", "EE", "BB", "TE" (all in C_ell uK^2).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, tau=0.054)
    pars.InitPower.set_params(As=2.1e-9, ns=0.965)
    pars.set_for_lmax(ell_max, lens_potential_accuracy=1)
    pars.DoLensing = True

    results = camb.get_results(pars)
    cl = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"]

    # cl columns: TT, EE, BB, TE
    ells = np.arange(cl.shape[0])
    return {
        "ell": ells,
        "TT": cl[:, 0],
        "EE": cl[:, 1],
        "BB": cl[:, 2],
        "TE": cl[:, 3],
    }


def camb_theory(ell_max: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute lensed EE and BB power spectra using CAMB.

    Returns:
        (ells, EE, BB) where EE and BB are lensed C_ell in uK^2.
        BB is non-zero due to gravitational lensing (E->B conversion).
    """
    d = camb_full_theory(ell_max)
    return d["ell"], d["EE"], d["BB"]


def bin_theory_with_window(ells: np.ndarray, cl: np.ndarray, w_ells: np.ndarray) -> np.ndarray:
    """Bin theory spectrum to the given bin centres using nearest-ell lookup."""
    result = np.zeros_like(w_ells, dtype=float)
    for i, L in enumerate(w_ells):
        idx = np.argmin(np.abs(ells - L))
        result[i] = cl[idx]
    return result
