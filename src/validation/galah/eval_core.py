import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


# GALAH standard 4-arm wave grids (must match preprocess_flux.py)
GALAH_ARM_WAVES = [
    np.linspace(4713, 4903, 4000),  # CCD1 (Blue)
    np.linspace(5648, 5873, 4000),  # CCD2 (Green)
    np.linspace(6478, 6737, 4000),  # CCD3 (Red)
    np.linspace(7585, 7887, 4000),  # CCD4 (NIR)
]
NUM_ARMS = 4
N_PIXELS = 4000


def continuum_normalize_arm(flux, window=101):
    """
    Continuum-normalize a single GALAH arm flux array.

    Identical pipeline to src/data/galah/preprocess_flux.py:
      1. Replace non-finite pixels with NaN
      2. Median-filter continuum estimate
      3. Divide by continuum
      4. 5-sigma spike rejection + linear interpolation
    """
    flux = np.array(flux, dtype=float)
    safe_flux = flux.copy()
    safe_flux[~np.isfinite(safe_flux)] = np.nan

    fill = np.nanmedian(safe_flux) if np.any(np.isfinite(safe_flux)) else 1.0
    bg_continuum = median_filter(np.nan_to_num(safe_flux, nan=fill), size=window)
    bg_continuum = np.where(bg_continuum <= 0, 1e-5, bg_continuum)
    norm_flux = safe_flux / bg_continuum

    norm_median = np.nanmedian(norm_flux)
    norm_std    = np.nanstd(norm_flux)
    spike_mask  = np.abs(norm_flux - norm_median) > 5.0 * norm_std
    spike_mask |= ~np.isfinite(norm_flux)

    if spike_mask.any():
        px    = np.arange(len(norm_flux))
        valid = ~spike_mask
        norm_flux = np.interp(px, px[valid], norm_flux[valid]) if valid.sum() > 10 \
                    else np.ones_like(norm_flux)

    return norm_flux.astype(np.float32)


def align_galah_spectrum(wave_in, flux_in, target_arm_waves=None):
    """
    Resample a raw GALAH spectrum onto the 4-arm standard grids used during training.

    Parameters
    ----------
    wave_in          : 1-D array, wavelength in Angstroms (linear)
    flux_in          : 1-D array, continuum-normalized flux matching wave_in
    target_arm_waves : list of 4 arrays (default: GALAH_ARM_WAVES)

    Returns
    -------
    norm_flux_4arm : np.ndarray shape (4, 4000), or None if data is invalid
    """
    if target_arm_waves is None:
        target_arm_waves = GALAH_ARM_WAVES

    wave_in = np.asarray(wave_in, dtype=float).flatten()
    flux_in = np.asarray(flux_in, dtype=float).flatten()

    if len(flux_in) < 10:
        return None
    if not np.any(np.isfinite(flux_in)):
        return None

    clean = np.isfinite(flux_in) & np.isfinite(wave_in)
    if clean.sum() < 10:
        return None

    try:
        f_interp = interp1d(wave_in[clean], flux_in[clean],
                            kind='linear', bounds_error=False,
                            fill_value=np.nanmedian(flux_in[clean]))
        arms = []
        for target_wave in target_arm_waves:
            arm_flux = f_interp(target_wave).astype(np.float32)
            arms.append(arm_flux)
        return np.stack(arms, axis=0)  # (4, 4000)
    except Exception as e:
        print(f"   [SKIP] GALAH arm alignment failed: {e}")
        return None


def read_galah_fits(fits_path):
    """
    Read a GALAH DR4 FITS file.

    Returns
    -------
    norm_flux_4arm : np.ndarray shape (4, 4000), continuum-normalized per arm
    star_id        : str
    is_valid       : bool
    """
    import astropy.io.fits as fits

    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"File not found: {fits_path}")

    arms_data = []
    try:
        with fits.open(fits_path, memmap=True) as hdul:
            hdr     = hdul[0].header
            star_id = str(hdr.get('OBJID', os.path.basename(fits_path))).strip()

            # GALAH DR4 stores each CCD arm as a separate extension or
            # as a 2-D array in extension 0. Handle both cases.
            data = hdul[0].data
            if data is None:
                print(f"   [SKIP] {os.path.basename(fits_path)}: empty data array.")
                return None, star_id, False

            if data.ndim == 2 and data.shape[0] == 4:
                # Shape (4, N) — one row per CCD arm
                crval1 = hdr.get('CRVAL1', 4713.0)
                cdelt1 = hdr.get('CDELT1', 0.05)
                n_pix  = data.shape[1]
                wave_full = crval1 + np.arange(n_pix) * cdelt1
                for arm_idx in range(4):
                    norm = continuum_normalize_arm(data[arm_idx], window=101)
                    arms_data.append((wave_full, norm))
            elif data.ndim == 1:
                # Single merged spectrum — split into 4 arms by wavelength
                crval1 = hdr.get('CRVAL1', 4713.0)
                cdelt1 = hdr.get('CDELT1', 0.05)
                n_pix  = len(data)
                wave_full = crval1 + np.arange(n_pix) * cdelt1
                norm_full = continuum_normalize_arm(data.astype(float), window=201)
                # Resample directly onto 4-arm grids
                norm_4arm = align_galah_spectrum(wave_full, norm_full)
                if norm_4arm is None:
                    return None, star_id, False
                return norm_4arm, star_id, True
            else:
                print(f"   [SKIP] {os.path.basename(fits_path)}: "
                      f"unexpected data shape {data.shape}.")
                return None, star_id, False

    except Exception as e:
        print(f"   [SKIP] {os.path.basename(fits_path)}: read error — {e}")
        return None, "", False

    if len(arms_data) != 4:
        return None, star_id, False

    # Resample each arm onto its standard grid
    norm_arms = []
    for arm_idx, (wave, flux) in enumerate(arms_data):
        target_wave = GALAH_ARM_WAVES[arm_idx]
        try:
            f_interp = interp1d(wave, flux, kind='linear',
                                bounds_error=False,
                                fill_value=np.nanmedian(flux))
            norm_arms.append(f_interp(target_wave).astype(np.float32))
        except Exception as e:
            print(f"   [SKIP] {os.path.basename(fits_path)}: "
                  f"arm {arm_idx} interpolation failed — {e}")
            return None, star_id, False

    return np.stack(norm_arms, axis=0), star_id, True
