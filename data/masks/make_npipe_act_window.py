"""Combine ACT window with Planck point source mask for ACT x NPIPE analysis.

The ACT window (pa5_f090 baseline) masks ACT-specific sources (15 mJy at
150 GHz), but Planck sees additional sources at its own frequencies. This
script multiplies the ACT window by the Planck combined PS mask to produce
a window suitable for NPIPE maps on the ACT footprint.

Steps:
    1. Load ACT window (CAR, equatorial)
    2. Load Planck PS mask (HEALPix nside=2048, Galactic)
    3. Project PS mask to the ACT CAR grid (Galactic -> equatorial)
    4. Multiply ACT window * PS mask
    5. Save combined window as FITS + diagnostic PNG

Usage:
    cd data/masks
    python make_npipe_act_window.py
"""
import os
import sys

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pixell import enmap, reproject
from pspy import so_map

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "spectrum_pipeline"))
import pipeline_config as cfg
from tools.data_loading import act_window_path

# ============================================================
# Paths
# ============================================================

ACT_WINDOW_PATH = act_window_path(cfg.NPIPE_ACT_WINDOW, "baseline")
NPIPE_PS_MASK_PATH = "path_to_planck_point_sources/combined_ps.fits"

OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FITS = os.path.join(OUTPUT_DIR, "npipe_act_window_with_ps.fits")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "npipe_act_window_with_ps.png")

# ============================================================
# Main
# ============================================================

def main():
    print("=== Combining ACT window with Planck PS mask ===")
    print()

    # 1. Load ACT window (CAR, equatorial)
    print(f"Loading ACT window: {ACT_WINDOW_PATH}")
    act_window = so_map.read_map(ACT_WINDOW_PATH)
    print(f"  Shape: {act_window.data.shape}")
    print(f"  Range: [{act_window.data.min():.4f}, {act_window.data.max():.4f}]")
    print(f"  Mean:  {act_window.data.mean():.4f}")

    # 2. Load Planck PS mask (HEALPix, Galactic)
    print(f"\nLoading Planck PS mask: {NPIPE_PS_MASK_PATH}")
    ps_mask_data = hp.read_map(NPIPE_PS_MASK_PATH)
    nside = hp.get_nside(ps_mask_data)
    print(f"  nside: {nside}")
    print(f"  Range: [{ps_mask_data.min():.6f}, {ps_mask_data.max():.6f}]")
    print(f"  Mean:  {ps_mask_data.mean():.4f}")
    print(f"  Frac < 0.5: {np.mean(ps_mask_data < 0.5):.4f}")

    # 3. Project PS mask from HEALPix Galactic -> ACT CAR equatorial
    #    Create a spin-0 so_map from the HEALPix data, then reproject
    print("\nProjecting Planck PS mask to ACT CAR grid...")
    ps_healpix = so_map.read_map(NPIPE_PS_MASK_PATH, coordinate="gal",
                                  fields_healpix=0)
    ps_car = so_map.healpix2car(ps_healpix, act_window)
    del ps_healpix

    print(f"  Projected shape: {ps_car.data.shape}")
    print(f"  Projected range: [{ps_car.data.min():.6f}, {ps_car.data.max():.6f}]")

    # Clip to [0, 1] — projection can introduce small overshoots
    ps_car.data = np.clip(ps_car.data, 0.0, 1.0)

    # 4. Multiply ACT window * PS mask
    print("\nMultiplying ACT window x Planck PS mask...")
    combined = act_window.copy()
    combined.data *= ps_car.data
    del ps_car

    # Stats
    act_nonzero = np.sum(act_window.data > 0.01)
    combined_nonzero = np.sum(combined.data > 0.01)
    n_masked = act_nonzero - combined_nonzero
    print(f"  ACT window pixels > 0.01:      {act_nonzero}")
    print(f"  Combined window pixels > 0.01:  {combined_nonzero}")
    print(f"  Newly masked pixels:            {n_masked} "
          f"({100 * n_masked / max(act_nonzero, 1):.2f}%)")
    print(f"  Combined range: [{combined.data.min():.6f}, {combined.data.max():.6f}]")
    print(f"  Combined mean:  {combined.data.mean():.6f}")

    # 5a. Save FITS
    print(f"\nSaving FITS: {OUTPUT_FITS}")
    combined.write_map(OUTPUT_FITS)

    # 5b. Save diagnostic PNG — mollview in Galactic coordinates
    #     Project CAR (equatorial) windows back to HEALPix (Galactic) for plotting
    print(f"Saving PNG:  {OUTPUT_PNG}")
    nside_plot = 1024
    rot = hp.Rotator(coord=["C", "G"])

    def _car_to_healpix_gal(car_map, nside_out):
        """Project CAR equatorial -> HEALPix Galactic for plotting."""
        hp_eq = reproject.map2healpix(car_map.data, nside=nside_out,
                                      lmax=3 * nside_out - 1, spin=0)
        return rot.rotate_map_pixel(hp_eq)

    hp_act = _car_to_healpix_gal(act_window, nside_plot)
    hp_combined = _car_to_healpix_gal(combined, nside_plot)

    plt.figure(figsize=(10, 16))
    hp.mollview(hp_act, title="ACT window (original)",
                min=0, max=1, cmap="RdYlBu_r", sub=(3, 1, 1))
    hp.mollview(ps_mask_data, title="Planck PS mask (full sky)",
                min=0, max=1, cmap="RdYlBu_r", sub=(3, 1, 2))
    hp.mollview(hp_combined, title="Combined: ACT window x Planck PS mask",
                min=0, max=1, cmap="RdYlBu_r", sub=(3, 1, 3))
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
