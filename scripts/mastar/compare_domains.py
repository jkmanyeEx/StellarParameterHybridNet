import os
import sys
import numpy as np
from scipy.ndimage import median_filter

# Resolve paths relative to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.validation.mastar.eval_core import align_wavelength_resolution, read_sdss_spec

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MASTAR_FLUX = os.path.join(BASE_DIR, "data", "mastar", "processed", "X_flux_clean.npy")

def main():
    if not os.path.exists(MASTAR_FLUX):
        print(f"[Error] MaStar training file not found at: {MASTAR_FLUX}")
        return

    mastar = np.load(MASTAR_FLUX)
    print("=" * 70)
    print("MaStar TRAINING flux (what the model learned on)")
    print("=" * 70)
    print(f"  shape         : {mastar.shape}")
    print(f"  per-spectrum mean (first 3): "
          f"{[f'{np.mean(mastar[i]):.3f}' for i in range(3)]}")
    print(f"  global min/max: {mastar.min():.3f} / {mastar.max():.3f}")
    print(f"  global mean   : {mastar.mean():.3f}")
    print(f"  → continuum-normalized: values oscillate around ~1.0")

    print("\n" + "=" * 70)
    print("SDSS spec EVAL flux")
    print("=" * 70)

    dataset_dir = os.path.join(BASE_DIR, "data", "mastar", "validation_dataset")
    cands = [f for f in os.listdir(dataset_dir) if f.endswith(".fits")] if os.path.isdir(dataset_dir) else []
    sample = os.path.join(dataset_dir, cands[0]) if cands else None

    if sample and os.path.exists(sample):
        flux, loglam, truth = read_sdss_spec(sample)
        aligned = align_wavelength_resolution(loglam, flux, 4563)[0]
        print(f"  file          : {os.path.basename(sample)}")
        print(f"  raw flux min/max : {flux.min():.3f} / {flux.max():.3f}")
        print(f"  raw flux mean    : {flux.mean():.3f}")
        print(f"  → RAW physical flux, NOT normalized!")
        print(f"  → values are huge/arbitrary, NOT oscillating around 1.0")

        print("\n" + "=" * 70)
        print("SDSS spec flux AFTER continuum normalization (the fix)")
        print("=" * 70)
        safe = np.copy(flux)
        bg = median_filter(safe, size=201)
        bg = np.where(bg <= 0, 1e-5, bg)
        norm = safe / bg
        print(f"  normalized min/max : {norm.min():.3f} / {norm.max():.3f}")
        print(f"  normalized mean    : {norm.mean():.3f}")
        print(f"  → NOW it oscillates around ~1.0, matching MaStar!")
    else:
        print(f"  No spec FITS found in {dataset_dir}/")

if __name__ == "__main__":
    main()
