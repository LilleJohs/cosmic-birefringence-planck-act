"""Generate precomputed coupling kernels and MCM inverses.

Run after run_pipeline.py to produce .npz files used by the likelihood code.
Requires the analysis windows to already be available.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import healpy as hp
from pixell import reproject

import pipeline_config as cfg
from pspy import so_map
from mask_builder import (
    get_act_analysis_window, get_npipe_analysis_window,
    get_npipe_cross_window, _project_act_window_to_healpix,
)
from compute_coupling import (
    compute_and_save_coupling,
    compute_and_save_cross_coupling,
)


def main():
    coupling_dir = os.path.join(cfg.OUTPUT_DIR, "coupling")
    os.makedirs(coupling_dir, exist_ok=True)

    binning_file = cfg.get_unified_binning_file()

    # ACT same-mask coupling kernel
    act_path = os.path.join(coupling_dir, "act_mask.npz")
    print(f"\n{'='*60}")
    print(f"ACT same-mask coupling → {act_path}")
    print(f"{'='*60}")
    (win_T, _), _ = get_act_analysis_window()
    lmax_act = cfg.ACT_X_ACT_COMPUTE_LMAX
    compute_and_save_coupling(win_T, binning_file, lmax_act, act_path)

    # Shared HEALPix setup for cross-mask couplings
    nside = 2048
    rot = hp.Rotator(coord=["C", "G"])
    lmax_cross = min(cfg.ACT_X_NPIPE_COMPUTE_LMAX, 3 * nside - 2)

    npipe_win, _ = get_npipe_analysis_window(nside)
    npipe_map = npipe_win[0]

    suffix = f"_{cfg.NPIPE_MASK_PERCENT}" if cfg.NPIPE_MASK_PERCENT != 0 else ""

    # Plain ACT window projected to HEALPix Galactic
    act_healpix = _project_act_window_to_healpix(nside)
    act_hp_map = so_map.healpix_template(ncomp=1, nside=nside)
    act_hp_map.data = act_healpix

    # ACT × NPIPE cross-mask coupling (for Cov(AA, PP))
    act_npipe_path = os.path.join(coupling_dir, f"act_x_npipe{suffix}.npz")
    print(f"\n{'='*60}")
    print(f"ACT × NPIPE cross-mask coupling → {act_npipe_path}")
    print(f"{'='*60}")
    compute_and_save_cross_coupling(act_hp_map, npipe_map, binning_file,
                                    lmax_cross, act_npipe_path)

    # ACT+PS × NPIPE cross-mask coupling (for Cov(AP, PP))
    actps_npipe_path = os.path.join(coupling_dir, f"actps_x_npipe{suffix}.npz")
    print(f"\n{'='*60}")
    print(f"ACT+PS × NPIPE cross-mask coupling → {actps_npipe_path}")
    print(f"{'='*60}")
    combined_win, _ = get_npipe_cross_window(npipe_freq=100)
    combined_healpix = reproject.map2healpix(
        combined_win[0].data, nside=nside, lmax=3 * nside - 1, spin=0)
    combined_healpix = rot.rotate_map_pixel(combined_healpix)
    actps_hp_map = so_map.healpix_template(ncomp=1, nside=nside)
    actps_hp_map.data = combined_healpix
    compute_and_save_cross_coupling(actps_hp_map, npipe_map, binning_file,
                                    lmax_cross, actps_npipe_path)

    print(f"\nDone. Coupling files saved to {coupling_dir}/")


if __name__ == "__main__":
    main()
