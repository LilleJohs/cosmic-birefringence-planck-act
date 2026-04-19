"""Map loading utilities for ACT (CAR) and NPIPE (HEALPix) maps."""
import healpy as hp
from pspy import so_map
from pspipe_utils.kspace import get_kspace_filter, filter_map

import pipeline_config as cfg
from tools.data_loading import (
    act_map_path, act_window_path, npipe_map_path, npipe_mask_path,
    PS_MASK_PATH,
)


def load_act_splits(array_band):
    """Load all splits for an ACT array-band.

    Returns:
        List of so_map objects (I/Q/U), one per split.
    """
    return [so_map.read_map(act_map_path(array_band, i))
            for i in range(cfg.ACT_N_SPLITS)]


def load_act_split_kspace_filtered(array_band, split_idx):
    """Load an ACT split map with k-space filter applied.

    Performs the full PSpipe map preparation: source add-back + k-space filter.
    Follows PSpipe/project/ACT_DR6/python/get_alms.py lines 68-89.

    Args:
        array_band: e.g. "pa5_f090"
        split_idx: 0-3

    Returns:
        so_map.so_map object with k-space filtered I/Q/U data.
    """
    # 1. Load source-free map
    srcfree = so_map.read_map(act_map_path(array_band, split_idx, product="map_srcfree"))

    # 2. Source model add-back
    original = so_map.read_map(act_map_path(array_band, split_idx, product="map"))
    source_model = original.copy()
    source_model.data -= srcfree.data
    ps_mask = so_map.read_map(PS_MASK_PATH)
    source_model.data *= ps_mask.data
    srcfree.data += source_model.data
    del original, source_model, ps_mask

    # 3. K-space filter
    win_kspace = so_map.read_map(act_window_path(array_band, window_type="kspace"))
    ks_filter = get_kspace_filter(srcfree, cfg.KSPACE_FILTER_DICT)
    filtered = filter_map(srcfree, ks_filter, win_kspace,
                          weighted_filter=cfg.KSPACE_FILTER_DICT.get("weighted", False))

    return filtered



# ============================================================
# NPIPE (Planck NPIPE6v20) loaders
# ============================================================

def load_npipe_split(freq, split):
    """Load an NPIPE detector-split map (HEALPix).

    Args:
        freq: Planck frequency in GHz (100, 143, 217, or 343)
        split: "A" or "B"

    Returns:
        so_map.so_map object (HEALPix, 3 components: I, Q, U) in μK_CMB.
    """
    npipe_map = so_map.read_map(npipe_map_path(freq, split))

    # Convert from K_CMB to μK_CMB
    npipe_map.data *= 1e6

    return npipe_map


def load_npipe_mask(nside):
    """Load NPIPE analysis mask (HEALPix).

    Args:
        nside: HEALPix resolution (1024 or 2048). Required.

    Returns:
        so_map.so_map object with the mask.
    """
    return so_map.read_map(npipe_mask_path(nside, cfg.NPIPE_MASK_PERCENT))


# ============================================================
# NPIPE -> CAR projection for ACT x NPIPE cross-spectra
# ============================================================

def project_npipe_to_car(freq, split, act_template):
    """Project an NPIPE HEALPix map to the ACT CAR grid.

    Follows PSpipe/project/ACT_DR6/python/planck/project_planck_maps.py:
    1. Load NPIPE HEALPix map (galactic coords)
    2. Convert K -> uK
    3. Reproject to ACT CAR grid via so_map.healpix2car()

    Args:
        freq: Frequency in GHz (100, 143, 217)
        split: "A" or "B"
        act_template: so_map with ACT CAR geometry to project onto.

    Returns:
        so_map with NPIPE data on the ACT CAR grid (in uK_CMB).
    """
    print(f"    Projecting NPIPE {freq}{split} to ACT CAR grid...")

    map_path = npipe_map_path(freq, split)
    print(f"    Loading {map_path}...")

    # Load as HEALPix first, then project to CAR
    # NPIPE maps are in galactic coordinates and K_CMB units
    # fields_healpix=[0,1,2] loads I, Q, U
    healpix_map = so_map.read_map(map_path, coordinate="gal",
                                   fields_healpix=[0, 1, 2])

    # Remove monopole and dipole from temperature map before projection.
    # NPIPE maps have residual monopole/dipole that leak into all ell
    # through the mode-coupling matrix when windowed by a small mask.
    healpix_map.data[0] = hp.remove_dipole(healpix_map.data[0], bad=hp.UNSEEN)
    print(f"    Removed monopole + dipole from temperature map")

    # Project to ACT CAR grid using pspy's healpix2car
    print(f"    Projecting HEALPix -> CAR (this may take several minutes)...")
    projected = so_map.healpix2car(healpix_map, act_template)

    del healpix_map

    # Convert K_CMB -> uK_CMB
    projected.data *= 1e6
    print(f"    Converted K -> uK")

    return projected
