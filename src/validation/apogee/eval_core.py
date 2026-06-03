import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


# APOGEE standard 3-arm wave grids (must match preprocess_flux.py)
APOGEE_ARM_WAVES = [
    np.linspace(15140, 15810, 2800),  # Blue chip
    np.linspace(15850, 16430, 2800),  # Green chip
    np.linspace(16470, 16960, 2800),  # Red chip
]
NUM_ARMS = 3
N_PIXELS = 2800


def continuum_normalize_arm(flux, window=101):
    """
    Continuum-normalize a single APOGEE arm flux array.

    Identical pipeline to src/data/apogee/preprocess_flux.py:
      1. NaN-fill bad pixels (non-finite)
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


def align_apogee_spectrum(wave_in, flux_in, target_arm_waves=None):
    """
    Resample a raw APOGEE spectrum (arbitrary wavelength grid) onto the
    3-arm standard grids used during training.

    Parameters
    ----------
    wave_in          : 1-D array, wavelength in Angstroms (linear, not log)
    flux_in          : 1-D array, continuum-normalized flux matching wave_in
    target_arm_waves : list of 3 arrays (default: APOGEE_ARM_WAVES)

    Returns
    -------
    norm_flux_3arm : np.ndarray shape (3, 2800), or None if data is bad
    """
    if target_arm_waves is None:
        target_arm_waves = APOGEE_ARM_WAVES

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
        return np.stack(arms, axis=0)  # (3, 2800)
    except Exception as e:
        print(f"   [SKIP] APOGEE arm alignment failed: {e}")
        return None


def read_apogee_fits(fits_path):
    """
    Read an APOGEE apStar or apVisit FITS file.

    Returns
    -------
    norm_flux_3arm : np.ndarray shape (3, 2800), continuum-normalized per arm
    star_id        : str
    is_valid       : bool
    """
    import astropy.io.fits as fits

    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"File not found: {fits_path}")

    with fits.open(fits_path, memmap=True) as hdul:
        hdr  = hdul[1].header
        flux = hdul[1].data.astype(float)

        # apStar files have shape (N_visits+2, N_pixels); row 0 = combined
        if flux.ndim == 2:
            flux = flux[0]

        naxis1 = hdr['NAXIS1']
        crval1 = hdr['CRVAL1']   # log10(lambda_start)
        cdelt1 = hdr['CDELT1']   # log10(delta_lambda)

        wave = 10.0 ** (crval1 + np.arange(naxis1) * cdelt1)

        star_id = str(hdr.get('OBJID', os.path.basename(fits_path))).strip()

    if len(flux) < 10 or not np.any(np.isfinite(flux)):
        print(f"   [SKIP] {os.path.basename(fits_path)}: bad flux array.")
        return None, star_id, False

    # Continuum-normalize the full combined spectrum first,
    # then resample onto per-arm grids.
    norm_full = continuum_normalize_arm(flux, window=201)
    norm_3arm = align_apogee_spectrum(wave, norm_full)

    if norm_3arm is None:
        return None, star_id, False

    return norm_3arm, star_id, True
