"""
GALAH DR4 spectral preprocessing pipeline.

Reads 4-arm reduced spectra downloaded by scripts/galah/download_spec.py
and produces:

  data/galah/processed/X_flux_clean.npy   shape (N, 4, 4000)
  data/galah/processed/star_ids.npy       shape (N,)
  data/galah/processed/standard_wave.npy  shape (4, 4000)

File naming convention (from DataCentral):
  {sobject_id}{ccd_number}.fits
  e.g.  1401130047013951.fits  (sobject_id=140113004701395, CCD1/Blue)

FITS extension layout (per official GALAH DR4 format):
  [0] Primary     : raw flux (sky-subtracted, NOT normalised)
  [1] normalized  : normalised flux          <-- we use this
  [2] relative_error
  [3] sky
  [4] teluric
  [5] scattered
  [6] cross_talk
  [7] resolution_profile

Wavelength grid: read from CRVAL1 + CDELT1 in HDU[1] header.
"""

import os
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

try:
    from src.utils.galah.config import CPU_WORKERS_PREPROCESS
except ImportError:
    CPU_WORKERS_PREPROCESS = max(1, cpu_count() - 1)

# Standard 4-arm output grids — must match extract_features.py and dataset.py
WAVE_GRIDS = [
    np.linspace(4713, 4903, 4000),  # CCD1 Blue
    np.linspace(5648, 5873, 4000),  # CCD2 Green
    np.linspace(6478, 6737, 4000),  # CCD3 Red
    np.linspace(7585, 7887, 4000),  # CCD4 NIR
]
STANDARD_WAVE = np.stack(WAVE_GRIDS, axis=0)  # (4, 4000)

# CCD number suffix for each arm (matches DataCentral filename convention)
CCD_SUFFIXES = ["1", "2", "3", "4"]

MIN_FITS_SIZE = 20_000  # bytes — real FITS are >=40 KB; VOTable stubs are ~2 KB


def _spike_clean(norm):
    """5-sigma spike rejection with linear interpolation of bad pixels."""
    med = np.nanmedian(norm)
    std = np.nanstd(norm)
    bad = np.abs(norm - med) > 5.0 * std
    bad |= ~np.isfinite(norm)
    if bad.any():
        px    = np.arange(len(norm))
        valid = ~bad
        norm  = (np.interp(px, px[valid], norm[valid])
                 if valid.sum() > 10 else np.ones_like(norm))
    return norm


def _process_single_star(args):
    """
    Worker function for multiprocessing.Pool.

    args: (sobject_id, spectra_dir)

    Reads normalised flux from HDU[1] of each arm file, resamples onto the
    standard 4-arm grid, and returns (sobject_id, ndarray shape (4, 4000)).
    Returns None if any arm is missing or unreadable.
    """
    import astropy.io.fits as fits

    sobject_id, spectra_dir = args
    arm_arrays = []

    for arm_idx, ccd in enumerate(CCD_SUFFIXES):
        fpath = os.path.join(spectra_dir, f"{sobject_id}{ccd}.fits")

        if not os.path.exists(fpath) or os.path.getsize(fpath) < MIN_FITS_SIZE:
            return None   # Missing or stub file — skip entire star

        try:
            with fits.open(fpath, memmap=False) as h:
                # HDU[1] is the normalised spectrum
                hdr  = h[1].header
                flux = h[1].data.astype(float)

                if flux.ndim != 1 or len(flux) < 100:
                    return None

                crval1 = float(hdr.get("CRVAL1", WAVE_GRIDS[arm_idx][0]))
                cdelt1 = float(hdr.get("CDELT1",
                                       (WAVE_GRIDS[arm_idx][-1] - WAVE_GRIDS[arm_idx][0])
                                       / max(len(flux) - 1, 1)))
                crpix1 = float(hdr.get("CRPIX1", 1.0))
                wave   = crval1 + (np.arange(len(flux)) - crpix1 + 1) * cdelt1

        except Exception:
            return None

        # The normalised flux from DataCentral is already continuum-divided.
        # Apply spike cleaning only.
        norm  = _spike_clean(flux.copy())
        clean = np.isfinite(norm) & np.isfinite(wave)
        if clean.sum() < 10:
            return None

        try:
            f_interp  = interp1d(wave[clean], norm[clean],
                                 kind="linear", bounds_error=False,
                                 fill_value=np.nanmedian(norm[clean]))
            resampled = f_interp(WAVE_GRIDS[arm_idx]).astype(np.float32)
        except Exception:
            return None

        arm_arrays.append(resampled)

    return sobject_id, np.stack(arm_arrays, axis=0)  # (4, 4000)


def run_galah_preprocessing_pipeline(raw_dir, out_dir):
    """
    Preprocess all GALAH DR4 spectra found in raw_dir/spectra/.

    Expected filenames: {sobject_id}{1,2,3,4}.fits
    Stars with any missing or stub arm file are silently skipped.
    """
    spectra_dir = os.path.join(raw_dir, "spectra")

    if not os.path.isdir(spectra_dir):
        raise FileNotFoundError(
            f"Spectra directory not found: {spectra_dir}\n"
            "Run scripts/galah/download_spec.py first."
        )

    # Collect candidates from CCD1 (*1.fits) files
    ccd1_files = [f for f in os.listdir(spectra_dir) if f.endswith("1.fits")]
    if not ccd1_files:
        raise FileNotFoundError(
            f"No *1.fits files found in: {spectra_dir}\n"
            "Run scripts/galah/download_spec.py first."
        )

    # Keep only stars where all 4 arms are present and large enough
    valid_ids, stub_count = [], 0
    for f1 in ccd1_files:
        sid = f1[:-6]  # strip the trailing "1.fits"
        all_ok = all(
            os.path.exists(os.path.join(spectra_dir, f"{sid}{c}.fits"))
            and os.path.getsize(os.path.join(spectra_dir, f"{sid}{c}.fits")) >= MIN_FITS_SIZE
            for c in CCD_SUFFIXES
        )
        if all_ok:
            valid_ids.append(sid)
        else:
            stub_count += 1

    total = len(valid_ids)
    print(f"[Preprocess] Stars with all 4 complete arms : {total}")
    if stub_count:
        print(f"[Preprocess] Skipped (stub/incomplete)      : {stub_count}")
        print(f"[Preprocess] Re-run download_spec.py to fetch missing arm files.")

    if total == 0:
        raise RuntimeError(
            "No complete 4-arm spectra found. "
            "All *1.fits files appear to be DataLink VOTable stubs "
            "(file size < 50 KB). Run scripts/galah/download_spec.py."
        )

    print(f"[Preprocess] Processing with {CPU_WORKERS_PREPROCESS} workers...")

    args = [(sid, spectra_dir) for sid in valid_ids]
    from tqdm import tqdm
    with Pool(processes=CPU_WORKERS_PREPROCESS) as pool:
        results = list(tqdm(
            pool.imap(_process_single_star, args, chunksize=50),
            total=total,
            desc="Preprocessing GALAH spectra",
            unit="star",
        ))

    good   = [r for r in results if r is not None]
    n_good = len(good)
    n_fail = total - n_good
    print(f"[Preprocess] Successfully processed : {n_good}")
    if n_fail:
        print(f"[Preprocess] FITS read failures     : {n_fail}")

    if n_good == 0:
        raise RuntimeError(
            "All spectra failed during preprocessing. "
            "Verify that the downloaded FITS files are valid GALAH DR4 spectra."
        )

    star_ids = np.array([r[0] for r in good])
    X_flux   = np.stack([r[1] for r in good], axis=0)  # (N, 4, 4000)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_flux_clean.npy"),  X_flux)
    np.save(os.path.join(out_dir, "star_ids.npy"),       star_ids)
    np.save(os.path.join(out_dir, "standard_wave.npy"),  STANDARD_WAVE)

    print(f"[Preprocess] Saved to : {out_dir}")
    print(f"   X_flux_clean.npy  : {X_flux.shape}")
    print(f"   star_ids.npy      : {star_ids.shape}")
    print(f"   standard_wave.npy : {STANDARD_WAVE.shape}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    raw_dir  = os.path.join(base_dir, "data", "galah", "raw")
    out_dir  = os.path.join(base_dir, "data", "galah", "processed")
    run_galah_preprocessing_pipeline(raw_dir, out_dir)
