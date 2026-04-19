"""Window function and sky fraction utilities shared across analysis folders."""
import numpy as np
from pspy import so_map

from .data_loading import act_window_path


def compute_fsky(window):
    """Compute effective sky fraction from a window.

    Args:
        window: so_map apodized mask.

    Returns:
        f_sky (float).
    """
    if hasattr(window.data, 'pixsizemap'):
        # CAR map: use pixel area weighting
        pixarea = window.data.pixsizemap()
        fsky = np.sum(window.data ** 2 * pixarea) / (4 * np.pi)
    else:
        npix = window.data.size
        fsky = np.sum(window.data ** 2) ** 2 / np.sum(window.data ** 4)
        fsky /= npix
    return fsky


def get_act_analysis_window(array_band="pa5_f090", window_type="baseline"):
    """Load official PSpipe window for ACT array-band.

    Args:
        array_band: e.g. "pa5_f090"
        window_type: "baseline" (default), "baseline_ivar", or "kspace"

    Returns:
        Tuple (window_T, window_P), f_sky
    """
    window_path = act_window_path(array_band, window_type)
    window = so_map.read_map(window_path)
    fsky = compute_fsky(window)
    return (window, window), fsky


